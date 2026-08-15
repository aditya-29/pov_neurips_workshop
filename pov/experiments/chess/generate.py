"""Chess experiment: synthetic games rendered as clips of several durations.

One game produces one clip per configured duration. The clips are prefixes of
the same game, so the frames are rendered **once** and reused across durations
— rendering the 10-minute clip already produced every frame the 5-second clip
needs.

Two motion modes:

``static``   one frame per half-move, held. Every frame is unique, so the whole
             clip costs `n_half_moves` renders no matter how long it runs.
``animated`` the original look: pieces slide between squares. Slide frames are
             genuinely different, so this mode is inherently heavier.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from pov.config import ConfigError, Reader
from pov.experiments.base import Generator
from pov.experiments.chess.engine import (
    ChessGame,
    GameRecord,
    MoveRecord,
    parse_square,
    play_random_game,
)
from pov.experiments.chess.render import BoardRenderer, BoardTheme
from pov.layout import RunLayout
from pov.manifest import MAX_INLINE_GROUND_TRUTH, ManifestWriter
from pov.video import probe_video, write_timeline

MOTION_MODES = ("static", "animated")


@dataclass(frozen=True)
class Duration:
    label: str
    seconds: float


@dataclass(frozen=True)
class ClipPlan:
    """Frame budget for one clip of one target duration."""

    label: str
    target_seconds: float
    intro_frames: int
    outro_frames: int
    n_moves: int
    frames_per_move: int

    @property
    def total_frames(self) -> int:
        return self.intro_frames + self.n_moves * self.frames_per_move + self.outro_frames

    def actual_seconds(self, fps: int) -> float:
        return self.total_frames / float(fps)


@dataclass(frozen=True)
class Timing:
    """Frame budget for each part of a clip.

    Intro and outro are *maximums*, not fixed costs. A 6-second title card and
    outro would swallow a clip labelled ``5s`` whole — leaving no room for a
    single move and producing a 7-second file. :meth:`plan` therefore caps them
    at a fraction of the target duration so short clips stay honest.
    """

    intro_frames: int = 60
    hold_frames: int = 36
    slide_frames: int = 14
    pause_frames: int = 22
    outro_frames: int = 120

    #: Largest share of a clip the intro / outro may consume.
    intro_share: float = 0.10
    outro_share: float = 0.20

    def frames_per_move(self, motion: str) -> int:
        if motion == "animated":
            return self.slide_frames + self.pause_frames
        return self.hold_frames

    def plan(self, label: str, seconds: float, fps: int, motion: str) -> ClipPlan:
        """Budget one clip: how much intro/outro it can afford, and how many moves fit."""
        budget = max(1, int(round(seconds * fps)))
        intro = min(self.intro_frames, int(budget * self.intro_share))
        outro = min(self.outro_frames, int(budget * self.outro_share))
        per_move = self.frames_per_move(motion)
        n_moves = max(1, (budget - intro - outro) // per_move)
        return ClipPlan(
            label=label,
            target_seconds=seconds,
            intro_frames=intro,
            outro_frames=outro,
            n_moves=n_moves,
            frames_per_move=per_move,
        )

    def moves_for_seconds(self, seconds: float, fps: int, motion: str) -> int:
        return self.plan("", seconds, fps, motion).n_moves


@dataclass
class ChessParams:
    num_games: int = 10
    durations: tuple[Duration, ...] = ()
    motion: str = "static"
    max_half_moves: int = 400
    max_game_attempts: int = 12
    timing: Timing = field(default_factory=Timing)
    theme: BoardTheme = field(default_factory=BoardTheme)
    use_svg_pieces: bool = True
    show_labels: bool = True


class ChessGenerator(Generator):
    name = "chess"

    # -- params ------------------------------------------------------------

    def parse_params(self, r: Reader) -> ChessParams:
        num_games = r.int("num_games", 10, min=1)
        motion = r.str("motion", "static", choices=MOTION_MODES)
        max_half_moves = r.int("max_half_moves", 400, min=1)
        max_game_attempts = r.int("max_game_attempts", 12, min=1)
        use_svg = r.bool("use_svg_pieces", True)
        show_labels = r.bool("show_labels", True)

        durations = self._parse_durations(r)
        timing = self._parse_timing(r.reader("timing"))
        theme = self._parse_theme(r.reader("theme"))
        r.done()

        return ChessParams(
            num_games=num_games,
            durations=durations,
            motion=motion,
            max_half_moves=max_half_moves,
            max_game_attempts=max_game_attempts,
            timing=timing,
            theme=theme,
            use_svg_pieces=use_svg,
            show_labels=show_labels,
        )

    @staticmethod
    def _parse_durations(r: Reader) -> tuple[Duration, ...]:
        raw = r.list("durations", [], min_len=1)
        durations: list[Duration] = []
        seen: set[str] = set()
        for i, item in enumerate(raw):
            item_reader = Reader(item, path=f"params.durations[{i}]")
            label = item_reader.str("label")
            seconds = item_reader.float("seconds", min=0.001)
            item_reader.done()
            if label in seen:
                raise ConfigError(f"params.durations: duplicate label {label!r}")
            seen.add(label)
            durations.append(Duration(label=label, seconds=seconds))
        return tuple(durations)

    @staticmethod
    def _parse_timing(r: Reader) -> Timing:
        timing = Timing(
            intro_frames=r.int("intro_frames", 60, min=0),
            hold_frames=r.int("hold_frames", 36, min=1),
            slide_frames=r.int("slide_frames", 14, min=1),
            pause_frames=r.int("pause_frames", 22, min=0),
            outro_frames=r.int("outro_frames", 120, min=0),
            intro_share=r.float("intro_share", 0.10, min=0.0, max=1.0),
            outro_share=r.float("outro_share", 0.20, min=0.0, max=1.0),
        )
        r.done()
        if timing.intro_share + timing.outro_share >= 1.0:
            raise ConfigError(
                "params.timing: intro_share + outro_share must be < 1.0 so at least "
                "one move fits in the clip"
            )
        return timing

    @staticmethod
    def _parse_theme(r: Reader) -> BoardTheme:
        theme = BoardTheme(
            square=r.int("square", 48, min=8),
            padding=r.int("padding", 16, min=0),
            panel=r.int("panel", 64, min=0),
        )
        r.done()
        return theme.validate()

    # -- generation --------------------------------------------------------

    def generate(self, layout: RunLayout, manifest: ManifestWriter) -> dict:
        params = self.params
        renderer = BoardRenderer(params.theme, use_svg=params.use_svg_pieces).prepare()

        plans = {
            duration.label: params.timing.plan(
                duration.label, duration.seconds, self.encode.fps, params.motion
            )
            for duration in params.durations
        }
        required_moves = max(plan.n_moves for plan in plans.values())

        started = time.perf_counter()
        rows: list[Any] = []
        errors: list[tuple[str, str]] = []
        skipped = 0
        total_renders = 0
        total_output_frames = 0
        total_piped_frames = 0

        workers = min(self.config.run.workers, params.num_games)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._process_game, index, renderer, plans, required_moves, layout
                ): index
                for index in range(params.num_games)
            }
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # keep one bad game from killing the run
                    errors.append((f"game{index:04d}", f"{type(exc).__name__}: {exc}"))
                    continue
                rows.extend(result["rows"])
                skipped += result["skipped"]
                total_renders += result["renders"]
                total_output_frames += result["output_frames"]
                total_piped_frames += result["piped_frames"]

        for row in sorted(rows, key=lambda r: r.sample_id):
            manifest.add(row)

        elapsed = time.perf_counter() - started
        saved = (
            1.0 - total_piped_frames / total_output_frames if total_output_frames else 0.0
        )
        return {
            "skipped": skipped,
            "errors": errors,
            "elapsed_sec": round(elapsed, 2),
            "frames_rendered": total_renders,
            "frames_encoded": total_output_frames,
            "frames_piped": total_piped_frames,
            "pipe_reduction": f"{saved:.1%}",
            "piece_art": renderer.art.source,
            "moves_per_duration": {label: plan.n_moves for label, plan in plans.items()},
        }

    # -- per-game work -----------------------------------------------------

    def _process_game(
        self,
        index: int,
        renderer: BoardRenderer,
        plans: dict[str, ClipPlan],
        required_moves: int,
        layout: RunLayout,
    ) -> dict:
        params = self.params
        record = self._play_long_enough_game(index, required_moves)

        # Render every frame the longest clip needs, once; shorter clips are
        # prefixes of the same list.
        longest = min(required_moves, record.n_half_moves)
        cache = _FrameCache(renderer, record, params, show_labels=params.show_labels)
        cache.build_through(longest)

        rows = []
        skipped = 0
        output_frames = 0
        piped_frames = 0

        for label, plan in sorted(plans.items()):
            n_moves = min(plan.n_moves, record.n_half_moves)
            sample_id = f"game{index:04d}_{label}"
            filename = f"{sample_id}.mp4"
            media_path = layout.media_file(filename)

            clip = record.truncated(n_moves)
            transcript = clip.transcript()
            gt_file = layout.ground_truth_file(f"{sample_id}.txt")
            gt_file.write_text(transcript, encoding="utf-8")

            if media_path.exists() and self.config.run.resume:
                skipped += 1
                stats = probe_video(media_path)
            else:
                segments = cache.segments(
                    n_moves,
                    intro_frames=plan.intro_frames,
                    outro_frames=plan.outro_frames,
                )
                stats = write_timeline(media_path, segments, self.encode,
                                       size=renderer.size)
                output_frames += stats.n_frames
                piped_frames += stats.piped_frames

            compact = clip.compact_moves()
            cell = (
                compact
                if len(compact) <= MAX_INLINE_GROUND_TRUTH
                else compact[:MAX_INLINE_GROUND_TRUTH].rstrip()
                + " …[truncated, see ground_truth_path]"
            )

            rows.append(
                self.make_row(
                    sample_id=sample_id,
                    condition=f"video_{label}",
                    media_type="video",
                    media_path=layout.relpath(media_path),
                    ground_truth=cell,
                    ground_truth_path=layout.relpath(gt_file),
                    fps=stats.fps,
                    n_frames=stats.n_frames,
                    duration_sec=round(stats.duration_sec, 3),
                    width=stats.width,
                    height=stats.height,
                    codec=stats.codec,
                    file_size_bytes=stats.file_size_bytes,
                    seed=record.seed,
                    game_index=index,
                    duration_label=label,
                    target_seconds=plan.target_seconds,
                    n_half_moves=n_moves,
                    game_total_half_moves=record.n_half_moves,
                    game_result=clip.result,
                    motion=params.motion,
                )
            )

        return {
            "rows": rows,
            "skipped": skipped,
            "renders": cache.render_count,
            "output_frames": output_frames,
            "piped_frames": piped_frames,
        }

    def _play_long_enough_game(self, index: int, required_moves: int) -> GameRecord:
        """Replay with fresh seeds until a game is long enough (or attempts run out)."""
        params = self.params
        base_seed = self.config.run.seed
        best: GameRecord | None = None

        for attempt in range(params.max_game_attempts):
            seed = base_seed + index + attempt * 1_000_003
            record = play_random_game(
                seed=seed,
                max_half_moves=params.max_half_moves,
                game_index=index,
            )
            if best is None or record.n_half_moves > best.n_half_moves:
                best = record
            if record.n_half_moves >= required_moves:
                return record

        assert best is not None
        return best


# ──────────────────────────────────────────────────────────────────────────────
# Frame building
# ──────────────────────────────────────────────────────────────────────────────


class _FrameCache:
    """Builds and memoises the timeline segments for one game."""

    def __init__(self, renderer: BoardRenderer, record: GameRecord,
                 params: ChessParams, show_labels: bool = True):
        self.renderer = renderer
        self.record = record
        self.params = params
        self.show_labels = show_labels
        self.render_count = 0

        #: _per_move[i] holds the segments for half-move i (0-based).
        self._intro_frame: np.ndarray | None = None
        self._per_move: list[list[tuple[np.ndarray, int]]] = []
        self._built_through = 0

    # -- public ------------------------------------------------------------

    def build_through(self, n_moves: int) -> None:
        if self._intro_frame is None:
            self._intro_frame = self._initial_frame()
        while self._built_through < n_moves:
            self._per_move.append(self._segments_for_move(self._built_through))
            self._built_through += 1

    def segments(
        self,
        n_moves: int,
        intro_frames: int | None = None,
        outro_frames: int | None = None,
    ) -> list[tuple[np.ndarray, int]]:
        """Timeline for the first `n_moves` half-moves, including intro/outro.

        Intro and outro lengths come from the clip's plan, so the same cached
        frames serve clips of every target duration.
        """
        if n_moves < 1:
            raise ValueError(f"n_moves must be >= 1, got {n_moves}")
        intro = self.params.timing.intro_frames if intro_frames is None else intro_frames
        outro = self.params.timing.outro_frames if outro_frames is None else outro_frames
        self.build_through(n_moves)

        segments: list[tuple[np.ndarray, int]] = []
        if intro > 0:
            assert self._intro_frame is not None
            segments.append((self._intro_frame, intro))
        for index in range(n_moves):
            segments.extend(self._per_move[index])

        # Hold the final position for the outro rather than emitting a new
        # segment with the same frame — one fewer pipe write.
        if outro > 0:
            frame, count = segments[-1]
            segments[-1] = (frame, count + outro)
        return segments

    # -- frame builders ----------------------------------------------------

    def _initial_frame(self) -> np.ndarray:
        self.render_count += 1
        return self.renderer.render(
            ChessGame().board,
            label="Starting position" if self.show_labels else "",
            sublabel=f"Game #{self.record.game_index}" if self.show_labels else "",
        )

    def _board_before(self, index: int):
        if index == 0:
            return ChessGame().board
        return self.record.moves[index - 1].board_after

    def _labels(self, move: MoveRecord) -> tuple[str, str]:
        if not self.show_labels:
            return "", ""
        # ASCII only: the panel font is whatever the OS provides, and a missing
        # arrow glyph renders as a tofu box in the middle of the label.
        label = f"Move {move.move_no} - {move.color}: {move.notation}"
        sublabel = f"{move.piece}  {move.src} -> {move.dst}"
        return label, sublabel

    def _segments_for_move(self, index: int) -> list[tuple[np.ndarray, int]]:
        move = self.record.moves[index]
        timing = self.params.timing
        label, sublabel = self._labels(move)
        label_color = self.renderer.theme.check if move.is_check else None
        src = parse_square(move.src)
        dst = parse_square(move.dst)

        if self.params.motion == "static":
            self.render_count += 1
            frame = self.renderer.render(
                move.board_after,
                highlight_from=src,
                highlight_to=dst,
                label=label,
                sublabel=sublabel,
                label_color=label_color,
            )
            return [(frame, timing.hold_frames)]

        return self._animated_segments(move, index, src, dst, label, sublabel, label_color)

    def _animated_segments(
        self,
        move: MoveRecord,
        index: int,
        src: tuple[int, int],
        dst: tuple[int, int],
        label: str,
        sublabel: str,
        label_color: tuple[int, int, int] | None,
    ) -> list[tuple[np.ndarray, int]]:
        timing = self.params.timing
        board_before = self._board_before(index)
        piece = board_before[src[0]][src[1]]
        x0, y0 = self.renderer.square_origin(*src)
        x1, y1 = self.renderer.square_origin(*dst)

        segments: list[tuple[np.ndarray, int]] = []
        steps = max(1, timing.slide_frames)
        for step in range(steps):
            t = step / steps
            self.render_count += 1
            frame = self.renderer.render(
                board_before,
                highlight_from=src,
                label=label,
                sublabel=sublabel,
                label_color=label_color,
                skip=src,
                floating=(piece, x0 + (x1 - x0) * t, y0 + (y1 - y0) * t) if piece else None,
            )
            segments.append((frame, 1))

        self.render_count += 1
        settled = self.renderer.render(
            move.board_after,
            highlight_from=src,
            highlight_to=dst,
            label=label,
            sublabel=sublabel,
            label_color=label_color,
        )
        segments.append((settled, timing.pause_frames))
        return segments
