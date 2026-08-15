"""Fast ffmpeg-backed video writing, shared by every experiment.

Why this exists
---------------
The original pipelines encoded each frame to a JPEG on disk and then handed the
directory to ffmpeg. For a word-by-word video at 0.5 words/sec that is 60
identical JPEGs *per word* — thousands of PIL encodes and disk writes to
produce a video whose content changes a few dozen times.

This module removes both costs:

1. **Raw frames are piped straight into ffmpeg's stdin.** No temp files, no
   JPEG round-trip, no re-decode.
2. **Held frames are never re-rendered or re-piped.** :func:`write_timeline`
   takes ``(frame, n_output_frames)`` segments and divides every hold count by
   their GCD, piping the reduced set at ``fps/gcd`` and letting ffmpeg expand
   them back to the target rate. A 60-frame hold is piped once, not 60 times.

The result is byte-identical output with an order-of-magnitude less work.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from pov.errors import PovError


class VideoError(PovError, RuntimeError):
    """Raised when ffmpeg/ffprobe is missing or fails."""


# ──────────────────────────────────────────────────────────────────────────────
# Tool discovery
# ──────────────────────────────────────────────────────────────────────────────


@lru_cache(maxsize=None)
def ffmpeg_path() -> str | None:
    return shutil.which("ffmpeg")


@lru_cache(maxsize=None)
def ffprobe_path() -> str | None:
    return shutil.which("ffprobe")


def ffmpeg_available() -> bool:
    return ffmpeg_path() is not None


def ffprobe_available() -> bool:
    return ffprobe_path() is not None


def require_ffmpeg() -> str:
    path = ffmpeg_path()
    if path is None:
        raise VideoError(
            "ffmpeg not found on PATH.\n"
            "  macOS : brew install ffmpeg\n"
            "  Ubuntu: sudo apt-get install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Encoder settings
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EncodeSettings:
    """Encoder knobs, mirroring :class:`pov.config.VideoConfig`."""

    fps: int = 30
    crf: int = 18
    preset: str = "veryfast"
    tune: str | None = "stillimage"
    pix_fmt: str = "yuv420p"
    codec: str = "libx264"
    threads: int = 0

    @classmethod
    def from_config(cls, video_config: Any) -> "EncodeSettings":
        return cls(
            fps=video_config.fps,
            crf=video_config.crf,
            preset=video_config.preset,
            tune=video_config.tune,
            pix_fmt=video_config.pix_fmt,
            codec=video_config.codec,
            threads=video_config.threads,
        )


#: ffmpeg encoder name → the codec name `ffprobe` reports for its output.
#: The manifest's `codec` column is documented as what is really in the file
#: (`h264`), and every path that probes a finished file reports that name. A
#: freshly written clip must not disagree with the same clip after a resume.
_STREAM_CODEC_NAMES = {
    "libx264": "h264",
    "h264_videotoolbox": "h264",
    "libopenh264": "h264",
    "libx265": "hevc",
    "hevc_videotoolbox": "hevc",
    "libvpx": "vp8",
    "libvpx-vp9": "vp9",
    "libaom-av1": "av1",
    "libsvtav1": "av1",
    "librav1e": "av1",
    "mjpeg": "mjpeg",
}


def stream_codec_name(encoder: str) -> str:
    """Name `ffprobe` will report for a stream written by `encoder`.

    Falls back to the encoder name itself, which is already correct for
    encoders named after their codec (e.g. ``mpeg4``, ``mjpeg``).
    """
    return _STREAM_CODEC_NAMES.get(encoder, encoder)


# ──────────────────────────────────────────────────────────────────────────────
# Frame normalisation
# ──────────────────────────────────────────────────────────────────────────────


def as_rgb_array(frame: Any) -> np.ndarray:
    """Coerce a PIL image or array-like into a contiguous uint8 RGB array."""
    if hasattr(frame, "convert") and hasattr(frame, "size"):  # PIL.Image
        frame = np.asarray(frame.convert("RGB"))
    array = np.asarray(frame)

    if array.ndim == 2:  # greyscale
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3:
        raise VideoError(f"frame must be 2-D or 3-D, got shape {array.shape}")
    if array.shape[2] == 4:  # drop alpha
        array = array[:, :, :3]
    if array.shape[2] != 3:
        raise VideoError(f"frame must have 3 channels, got {array.shape[2]}")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


# ──────────────────────────────────────────────────────────────────────────────
# Writer
# ──────────────────────────────────────────────────────────────────────────────


class VideoWriter:
    """Streams raw RGB frames into ffmpeg.

    Usage::

        with VideoWriter(path, (w, h), EncodeSettings()) as writer:
            writer.write(frame, repeat=15)

    `input_fps` decouples the piped rate from the output rate: pipe one frame
    per word at 2 fps and ask for a 30 fps output, and ffmpeg does the
    duplication internally. :func:`write_timeline` uses this automatically.
    """

    def __init__(
        self,
        path: str | Path,
        size: tuple[int, int],
        settings: EncodeSettings | None = None,
        *,
        input_fps: str | float | None = None,
    ):
        self.path = Path(path)
        self.width, self.height = int(size[0]), int(size[1])
        self.settings = settings or EncodeSettings()
        self.input_fps = input_fps if input_fps is not None else self.settings.fps

        if self.width <= 0 or self.height <= 0:
            raise VideoError(f"invalid frame size {size}")
        if self.width % 2 or self.height % 2:
            raise VideoError(
                f"frame size {self.width}x{self.height} must have even dimensions "
                f"for {self.settings.pix_fmt} — pad the canvas"
            )

        self.frames_written = 0
        self._proc: subprocess.Popen | None = None
        self._stderr: list[bytes] = []
        self._stderr_thread: threading.Thread | None = None
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    def _command(self) -> list[str]:
        s = self.settings
        cmd = [
            require_ffmpeg(),
            "-y",
            "-v", "error",
            "-nostdin",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-framerate", str(self.input_fps),
            "-i", "-",
            "-an",
            "-c:v", s.codec,
            "-crf", str(s.crf),
            "-preset", s.preset,
            "-pix_fmt", s.pix_fmt,
            "-r", str(s.fps),
        ]
        if s.tune:
            cmd += ["-tune", s.tune]
        if s.threads:
            cmd += ["-threads", str(s.threads)]
        cmd.append(str(self.path))
        return cmd

    def open(self) -> "VideoWriter":
        if self._proc is not None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Drain stderr concurrently: a full pipe buffer would deadlock the
        # writer mid-stream, and that failure mode is miserable to debug.
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        return self

    def _drain_stderr(self) -> None:
        assert self._proc is not None and self._proc.stderr is not None
        for line in self._proc.stderr:
            self._stderr.append(line)

    def __enter__(self) -> "VideoWriter":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()

    # -- writing -----------------------------------------------------------

    def write(self, frame: Any, repeat: int = 1) -> None:
        """Write `frame` `repeat` times.

        The frame is converted to bytes **once**; repeats reuse that buffer.
        """
        if repeat < 0:
            raise VideoError(f"repeat must be >= 0, got {repeat}")
        if repeat == 0:
            return
        if self._proc is None:
            self.open()
        assert self._proc is not None and self._proc.stdin is not None

        array = as_rgb_array(frame)
        if array.shape[0] != self.height or array.shape[1] != self.width:
            raise VideoError(
                f"frame is {array.shape[1]}x{array.shape[0]}, "
                f"writer expects {self.width}x{self.height}"
            )

        payload = array.tobytes()
        try:
            for _ in range(repeat):
                self._proc.stdin.write(payload)
        except BrokenPipeError as exc:
            raise VideoError(
                f"ffmpeg exited early while writing {self.path}:\n{self._stderr_text()}"
            ) from exc
        self.frames_written += repeat

    # -- teardown ----------------------------------------------------------

    def close(self) -> Path:
        if self._closed:
            return self.path
        self._closed = True
        if self._proc is None:
            raise VideoError(f"no frames were written to {self.path}")
        if self.frames_written == 0:
            self.abort()
            raise VideoError(f"no frames were written to {self.path}")

        assert self._proc.stdin is not None
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        returncode = self._proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=5)
        if returncode != 0:
            raise VideoError(
                f"ffmpeg failed ({returncode}) writing {self.path}:\n{self._stderr_text()}"
            )
        return self.path

    def abort(self) -> None:
        """Kill ffmpeg and remove the partial file."""
        self._closed = True
        if self._proc is None:
            return
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self._proc.kill()
            self._proc.wait(timeout=5)
        except Exception:  # pragma: no cover - best effort cleanup
            pass
        self.path.unlink(missing_ok=True)

    def _stderr_text(self) -> str:
        return b"".join(self._stderr).decode("utf-8", errors="replace").strip()


# ──────────────────────────────────────────────────────────────────────────────
# Timeline writing (the fast path)
# ──────────────────────────────────────────────────────────────────────────────


def reduce_holds(counts: Sequence[int]) -> int:
    """GCD of hold counts — the factor by which piped frames can be reduced."""
    divisor = 0
    for count in counts:
        if count < 0:
            raise VideoError(f"hold count must be >= 0, got {count}")
        divisor = math.gcd(divisor, int(count))
    return divisor or 1


def write_timeline(
    path: str | Path,
    segments: Iterable[tuple[Any, int]],
    settings: EncodeSettings | None = None,
    *,
    size: tuple[int, int] | None = None,
) -> "VideoStats":
    """Encode ``(frame, n_output_frames)`` segments into a video.

    Each frame is rendered by the caller **once** and held for the requested
    number of output frames. Hold counts are divided by their GCD so identical
    frames are piped as few times as possible; ffmpeg restores the real timing.

    Returns :class:`VideoStats` describing what was produced.
    """
    settings = settings or EncodeSettings()
    materialised = [(frame, int(count)) for frame, count in segments if int(count) > 0]
    if not materialised:
        raise VideoError(f"cannot write {path}: timeline has no frames")

    counts = [count for _, count in materialised]
    divisor = reduce_holds(counts)
    total_output_frames = sum(counts)

    if size is None:
        first = as_rgb_array(materialised[0][0])
        size = (first.shape[1], first.shape[0])

    # Pipe at fps/divisor; ffmpeg's rate conversion re-expands each frame.
    input_fps = f"{settings.fps}/{divisor}" if divisor > 1 else str(settings.fps)

    writer = VideoWriter(path, size, settings, input_fps=input_fps)
    with writer:
        for frame, count in materialised:
            writer.write(frame, repeat=count // divisor)

    return VideoStats(
        path=Path(path),
        width=size[0],
        height=size[1],
        fps=float(settings.fps),
        n_frames=total_output_frames,
        duration_sec=total_output_frames / float(settings.fps),
        codec=stream_codec_name(settings.codec),
        piped_frames=writer.frames_written,
        file_size_bytes=Path(path).stat().st_size if Path(path).exists() else 0,
    )


@dataclass(frozen=True)
class VideoStats:
    """Properties of a written video, recorded in the manifest."""

    path: Path
    width: int
    height: int
    fps: float
    n_frames: int
    duration_sec: float
    codec: str
    piped_frames: int = 0
    file_size_bytes: int = 0

    @property
    def pipe_savings(self) -> float:
        """Fraction of frame writes avoided by hold reduction."""
        if not self.n_frames:
            return 0.0
        return 1.0 - (self.piped_frames / self.n_frames)


# ──────────────────────────────────────────────────────────────────────────────
# Probing / transcoding
# ──────────────────────────────────────────────────────────────────────────────


def probe_video(path: str | Path) -> VideoStats:
    """Read real properties of an existing video via ffprobe."""
    path = Path(path)
    if not path.exists():
        raise VideoError(f"video not found: {path}")
    probe = ffprobe_path()
    if probe is None:
        raise VideoError("ffprobe not found on PATH (install ffmpeg)")

    cmd = [
        probe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name,duration",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise VideoError(f"ffprobe failed for {path}:\n{result.stderr.strip()}")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise VideoError(f"{path}: no video stream found")
    stream = streams[0]

    fps = _parse_rate(stream.get("avg_frame_rate")) or _parse_rate(stream.get("r_frame_rate")) or 0.0
    duration = _parse_float(stream.get("duration"))
    if duration is None:
        duration = _parse_float((data.get("format") or {}).get("duration")) or 0.0

    n_frames = _parse_int(stream.get("nb_frames"))
    if n_frames is None:
        n_frames = int(round(duration * fps)) if fps else 0

    return VideoStats(
        path=path,
        width=int(stream.get("width") or 0),
        height=int(stream.get("height") or 0),
        fps=fps,
        n_frames=n_frames,
        duration_sec=duration,
        codec=str(stream.get("codec_name") or ""),
        file_size_bytes=path.stat().st_size,
    )


def cut_clip(
    source: str | Path,
    dest: str | Path,
    settings: EncodeSettings | None = None,
    *,
    start: float | None = None,
    duration: float | None = None,
    scale_height: int | None = None,
    target_fps: int | None = None,
) -> VideoStats:
    """Cut / normalise a clip out of an existing video (used by the ASL pipeline).

    Re-encodes so every clip in a run shares one codec, pixel format and frame
    rate; ``-ss`` before ``-i`` keeps the seek fast.
    """
    settings = settings or EncodeSettings()
    source, dest = Path(source), Path(dest)
    if not source.exists():
        raise VideoError(f"source video not found: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [require_ffmpeg(), "-y", "-v", "error", "-nostdin"]
    if start is not None:
        if start < 0:
            raise VideoError(f"start must be >= 0, got {start}")
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(source)]
    if duration is not None:
        if duration <= 0:
            raise VideoError(f"duration must be > 0, got {duration}")
        cmd += ["-t", f"{duration:.3f}"]

    filters = []
    if scale_height:
        # -2 keeps the aspect ratio and forces an even width for yuv420p.
        filters.append(f"scale=-2:{int(scale_height)}")
    if filters:
        cmd += ["-vf", ",".join(filters)]

    fps = int(target_fps or settings.fps)
    cmd += [
        "-an",
        "-c:v", settings.codec,
        "-crf", str(settings.crf),
        "-preset", settings.preset,
        "-pix_fmt", settings.pix_fmt,
        "-r", str(fps),
    ]
    if settings.threads:
        cmd += ["-threads", str(settings.threads)]
    cmd.append(str(dest))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        dest.unlink(missing_ok=True)
        raise VideoError(f"ffmpeg failed cutting {source} → {dest}:\n{result.stderr.strip()}")

    return probe_video(dest)


def _parse_rate(value: Any) -> float | None:
    """Parse an ffprobe rational rate like '30/1'."""
    if not value or value in ("0/0", "N/A"):
        return None
    text = str(value)
    if "/" in text:
        num, _, den = text.partition("/")
        try:
            numerator, denominator = float(num), float(den)
        except ValueError:
            return None
        return numerator / denominator if denominator else None
    return _parse_float(text)


def _parse_float(value: Any) -> float | None:
    if value is None or value == "N/A":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    if value is None or value == "N/A":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
