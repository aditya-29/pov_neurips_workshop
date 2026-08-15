"""Word-by-word MCQ experiment.

Per question, up to three condition families:

``static_image``       the whole question as one JPEG — the image baseline.
``vanishing_<speed>``  one word at a time, each followed by a blank gap.
``cumulative_<speed>`` words accumulate; the last frame equals the whole question.

The two video families are rendered **once per question** and encoded at every
configured speed: changing words-per-second changes only how long each frame is
held, never the frames themselves.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pov.config import ConfigError, Reader
from pov.experiments.base import Generator
from pov.experiments.wbw_mcq.question import Question, load_questions
from pov.experiments.wbw_mcq.render import Canvas, FrameRenderer
from pov.layout import RunLayout
from pov.manifest import ManifestWriter
from pov.video import write_timeline

MODES = ("vanishing", "cumulative")
DEFAULT_SPEEDS = {"slow": 0.5, "normal": 2.0, "fast": 5.0}


@dataclass
class WbwParams:
    questions_path: Path = Path("data/questions.jsonl")
    questions_format: str = "auto"
    limit: int | None = None
    shuffle: bool = False
    speeds: dict = field(default_factory=lambda: dict(DEFAULT_SPEEDS))
    modes: tuple[str, ...] = MODES
    static_image: bool = True
    blank_gap_frames: int = 4
    min_frames_per_word: int = 1
    jpeg_quality: int = 90
    canvas: Canvas = field(default_factory=Canvas)


def frames_per_word(fps: int, wps: float, minimum: int = 1) -> int:
    """How many output frames one word occupies at `wps` words per second."""
    if wps <= 0:
        raise ConfigError(f"speed must be > 0 words/sec, got {wps}")
    return max(minimum, round(fps / wps))


class WbwMcqGenerator(Generator):
    name = "wbw_mcq"

    # -- params ------------------------------------------------------------

    def parse_params(self, r: Reader) -> WbwParams:
        questions_path = r.path("questions_path")
        questions_format = r.str("questions_format", "auto",
                                 choices=("auto", "jsonl", "mmlu_csv"))
        limit_raw = r.raw("limit", None)
        shuffle = r.bool("shuffle", False)
        static_image = r.bool("static_image", True)
        blank_gap_frames = r.int("blank_gap_frames", 4, min=0)
        min_frames_per_word = r.int("min_frames_per_word", 1, min=1)
        jpeg_quality = r.int("jpeg_quality", 90, min=1, max=100)

        modes_raw = r.list("modes", list(MODES), item_type=str)
        speeds = self._parse_speeds(r.reader("speeds"))
        canvas = self._parse_canvas(r.reader("canvas"))
        r.done()

        if limit_raw is not None and (
            isinstance(limit_raw, bool) or not isinstance(limit_raw, int) or limit_raw < 1
        ):
            raise ConfigError(f"params.limit: expected a positive integer or null, got {limit_raw!r}")

        modes: list[str] = []
        for mode in modes_raw:
            if mode not in MODES:
                raise ConfigError(f"params.modes: {mode!r} is not one of {list(MODES)}")
            if mode not in modes:
                modes.append(mode)

        if not modes and not static_image:
            raise ConfigError(
                "params: nothing to generate — set static_image: true or list at least one mode"
            )
        if modes and not speeds:
            raise ConfigError("params.speeds: at least one speed is required for video modes")

        return WbwParams(
            questions_path=questions_path,
            questions_format=questions_format,
            limit=limit_raw,
            shuffle=shuffle,
            speeds=speeds,
            modes=tuple(modes),
            static_image=static_image,
            blank_gap_frames=blank_gap_frames,
            min_frames_per_word=min_frames_per_word,
            jpeg_quality=jpeg_quality,
            canvas=canvas,
        )

    @staticmethod
    def _parse_speeds(r: Reader) -> dict:
        names = r.keys()  # every key here is a user-chosen speed name
        if not names:
            return dict(DEFAULT_SPEEDS)
        speeds: dict = {}
        for name in sorted(names):
            if not name.replace("_", "").replace("-", "").isalnum():
                raise ConfigError(
                    f"params.speeds: {name!r} must be alphanumeric — it becomes part of a filename"
                )
            speeds[name] = r.float(name, min=0.0001)
        r.done()
        return speeds

    @staticmethod
    def _parse_canvas(r: Reader) -> Canvas:
        canvas = Canvas(
            width=r.int("width", 1200, min=64),
            height=r.int("height", 800, min=64),
            video_font_size=r.int("video_font_size", 36, min=6),
            video_padding=r.int("video_padding", 80, min=0),
            video_line_gap=r.int("video_line_gap", 8, min=0),
            static_font_size=r.int("static_font_size", 18, min=6),
            static_padding=r.int("static_padding", 60, min=0),
            static_line_gap=r.int("static_line_gap", 14, min=0),
        )
        r.done()
        try:
            return canvas.validate()
        except ValueError as exc:
            raise ConfigError(f"params.canvas: {exc}") from exc

    # -- preflight ---------------------------------------------------------

    def check_inputs(self) -> list[str]:
        path = self.params.questions_path
        if not path.exists():
            return [
                f"questions file not found: {path}  "
                "(run: python scripts/fetch_mmlu.py --limit 200)"
            ]
        try:
            load_questions(path, self.params.questions_format)
        except Exception as exc:
            return [f"questions file is unreadable: {path} — {exc}"]
        return []

    def describe_inputs(self) -> list[str]:
        count = len(load_questions(self.params.questions_path,
                                   self.params.questions_format))
        if self.params.limit:
            count = min(count, self.params.limit)
        return [f"questions  : {count} from {self.params.questions_path}"]

    # -- generation --------------------------------------------------------

    def generate(self, layout: RunLayout, manifest: ManifestWriter) -> dict:
        params = self.params
        questions = load_questions(params.questions_path, params.questions_format)

        if params.shuffle:
            random.Random(self.config.run.seed).shuffle(questions)
        if params.limit is not None:
            questions = questions[: params.limit]

        renderer = FrameRenderer(params.canvas)
        renderer.render_blank()  # warm the shared blank frame

        started = time.perf_counter()
        rows: list[Any] = []
        errors: list[tuple[str, str]] = []
        skipped = 0
        renders = 0
        output_frames = 0
        piped_frames = 0

        workers = min(self.config.run.workers, max(1, len(questions)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self._process_question, question, renderer, layout): question
                for question in questions
            }
            for future in as_completed(futures):
                question = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    errors.append((question.id, f"{type(exc).__name__}: {exc}"))
                    continue
                rows.extend(result["rows"])
                skipped += result["skipped"]
                renders += result["renders"]
                output_frames += result["output_frames"]
                piped_frames += result["piped_frames"]

        for row in sorted(rows, key=lambda r: r.sample_id):
            manifest.add(row)

        elapsed = time.perf_counter() - started
        saved = 1.0 - piped_frames / output_frames if output_frames else 0.0
        return {
            "skipped": skipped,
            "errors": errors,
            "elapsed_sec": round(elapsed, 2),
            "questions": len(questions),
            "frames_rendered": renders,
            "frames_encoded": output_frames,
            "frames_piped": piped_frames,
            "pipe_reduction": f"{saved:.1%}",
            "speeds": params.speeds,
        }

    # -- per-question work -------------------------------------------------

    def _process_question(
        self, question: Question, renderer: FrameRenderer, layout: RunLayout
    ) -> dict:
        params = self.params
        fps = self.encode.fps
        words = question.word_sequence

        rows: list[Any] = []
        skipped = 0
        renders = 0
        output_frames = 0
        piped_frames = 0

        common = dict(
            ground_truth=question.answer,
            question_id=question.id,
            word_count=question.word_count,
            stem=question.stem,
            option_a=question.options.get("A", ""),
            option_b=question.options.get("B", ""),
            option_c=question.options.get("C", ""),
            option_d=question.options.get("D", ""),
            domain=question.domain,
            difficulty=question.difficulty,
            question_source=question.source,
        )

        # ── static image ────────────────────────────────────────────────
        if params.static_image:
            sample_id = f"{question.id}_static"
            path = layout.media_file(f"{sample_id}.jpg")
            if path.exists() and self.config.run.resume:
                skipped += 1
                with Image.open(path) as existing:
                    width, height = existing.size
            else:
                image = renderer.render_full_question(question)
                renders += 1
                image.save(path, quality=params.jpeg_quality)
                width, height = image.size
            rows.append(
                self.make_row(
                    sample_id=sample_id,
                    condition="static_image",
                    media_type="image",
                    media_path=layout.relpath(path),
                    width=width,
                    height=height,
                    n_frames=1,
                    codec="jpeg",
                    file_size_bytes=path.stat().st_size,
                    mode="static",
                    speed_wps=None,
                    frames_per_word=None,
                    **common,
                )
            )

        # ── video conditions ────────────────────────────────────────────
        # Frames are built once per mode and reused across every speed.
        for mode in params.modes:
            frames, built = self._build_frames(mode, words, renderer)
            renders += built

            for speed_name, wps in sorted(params.speeds.items()):
                fpw = frames_per_word(fps, wps, params.min_frames_per_word)
                sample_id = f"{question.id}_{mode}_{speed_name}"
                path = layout.media_file(f"{sample_id}.mp4")

                if path.exists() and self.config.run.resume:
                    skipped += 1
                    from pov.video import probe_video

                    stats = probe_video(path)
                else:
                    segments = self._segments(mode, frames, fpw, renderer)
                    stats = write_timeline(path, segments, self.encode,
                                           size=params.canvas.size)
                    output_frames += stats.n_frames
                    piped_frames += stats.piped_frames

                rows.append(
                    self.make_row(
                        sample_id=sample_id,
                        condition=f"{mode}_{speed_name}",
                        media_type="video",
                        media_path=layout.relpath(path),
                        fps=stats.fps,
                        n_frames=stats.n_frames,
                        duration_sec=round(stats.duration_sec, 3),
                        width=stats.width,
                        height=stats.height,
                        codec=stats.codec,
                        file_size_bytes=stats.file_size_bytes,
                        mode=mode,
                        speed=speed_name,
                        speed_wps=wps,
                        frames_per_word=fpw,
                        **common,
                    )
                )

        return {
            "rows": rows,
            "skipped": skipped,
            "renders": renders,
            "output_frames": output_frames,
            "piped_frames": piped_frames,
        }

    @staticmethod
    def _build_frames(
        mode: str, words: list[str], renderer: FrameRenderer
    ) -> tuple[list[np.ndarray], int]:
        """Render the unique frames for one mode. Returns (frames, n_rendered)."""
        if mode == "vanishing":
            frames = [renderer.render_words([word]) for word in words]
        elif mode == "cumulative":
            frames = [renderer.render_words(words[: i + 1]) for i in range(len(words))]
        else:  # pragma: no cover - guarded by config validation
            raise ConfigError(f"unknown mode {mode!r}")
        return frames, len(frames)

    def _segments(
        self, mode: str, frames: list[np.ndarray], fpw: int, renderer: FrameRenderer
    ) -> list[tuple[np.ndarray, int]]:
        if mode == "vanishing":
            gap = self.params.blank_gap_frames
            blank = renderer.render_blank()
            segments: list[tuple[np.ndarray, int]] = []
            for frame in frames:
                segments.append((frame, fpw))
                if gap:
                    segments.append((blank, gap))
            return segments
        return [(frame, fpw) for frame in frames]
