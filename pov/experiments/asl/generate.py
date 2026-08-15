"""ASL experiment: duration-bucketed sign-language clips.

Reproduces the original notebook's curation — read the How2Sign metadata,
measure every video, bucket by duration, sample evenly — and then goes one step
further by cutting and normalising the sampled clips into the run directory, so
a run is self-contained rather than a set of pointers into a scratch folder.

Measuring thousands of videos is the slow part, so durations are cached on disk
keyed by path, size and mtime; a second run reuses them.
"""

from __future__ import annotations

import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pov.config import ConfigError, Reader
from pov.experiments.asl.sampling import (
    Bucket,
    SamplingError,
    build_buckets,
    sample_buckets,
)
from pov.experiments.base import Generator
from pov.layout import RunLayout
from pov.manifest import ManifestWriter
from pov.video import cut_clip, probe_video

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class AslParams:
    metadata_csv: Path = Path("how2sign_val.csv")
    video_dir: Path = Path("raw_videos")
    delimiter: str = "auto"
    id_column: str = "SENTENCE_NAME"
    text_column: str = "SENTENCE"
    duration_column: str | None = None
    video_extension: str = ".mp4"
    buckets: tuple[Bucket, ...] = ()
    samples_per_bucket: int | None = None
    total_samples: int | None = None
    scale_height: int | None = 480
    target_fps: int | None = None
    max_clip_seconds: float | None = None
    use_time_range: bool = False
    start_column: str = "START"
    end_column: str = "END"
    duration_cache: Path | None = None
    auto_duration_cache: bool = True
    probe_workers: int = 16


class AslGenerator(Generator):
    name = "asl"

    # -- params ------------------------------------------------------------

    def parse_params(self, r: Reader) -> AslParams:
        metadata_csv = r.path("metadata_csv")
        video_dir = r.path("video_dir")
        delimiter = r.str("delimiter", "auto")
        id_column = r.str("id_column", "SENTENCE_NAME")
        text_column = r.str("text_column", "SENTENCE")
        duration_column = r.optional_str("duration_column", None)
        video_extension = r.str("video_extension", ".mp4")
        samples_per_bucket = _optional_positive_int(r, "samples_per_bucket")
        total_samples = _optional_positive_int(r, "total_samples")
        scale_height = _optional_positive_int(r, "scale_height", default=480)
        target_fps = _optional_positive_int(r, "target_fps")
        max_clip_seconds = _optional_positive_float(r, "max_clip_seconds")
        use_time_range = r.bool("use_time_range", False)
        start_column = r.str("start_column", "START")
        end_column = r.str("end_column", "END")
        cache_raw = r.optional_str("duration_cache", "auto")
        probe_workers = r.int("probe_workers", 16, min=1, max=256)

        buckets = self._parse_buckets(r.reader("buckets"))
        r.done()

        if (samples_per_bucket is None) == (total_samples is None):
            raise ConfigError(
                "params: specify exactly one of samples_per_bucket or total_samples"
            )
        if len(delimiter) != 1 and delimiter != "auto":
            raise ConfigError(
                f"params.delimiter: must be a single character or 'auto', got {delimiter!r}"
            )
        if not video_extension.startswith("."):
            raise ConfigError(
                f"params.video_extension: must start with a dot, got {video_extension!r}"
            )

        # 'auto' is resolved against the output root at generation time so the
        # cache never lands inside the user's source video directory.
        auto_duration_cache = cache_raw == "auto"
        if auto_duration_cache or cache_raw in (None, "", "none", "off"):
            duration_cache = None
        else:
            duration_cache = Path(cache_raw).expanduser()

        return AslParams(
            metadata_csv=metadata_csv,
            video_dir=video_dir,
            delimiter=delimiter,
            id_column=id_column,
            text_column=text_column,
            duration_column=duration_column,
            video_extension=video_extension,
            buckets=tuple(buckets),
            samples_per_bucket=samples_per_bucket,
            total_samples=total_samples,
            scale_height=scale_height,
            target_fps=target_fps,
            max_clip_seconds=max_clip_seconds,
            use_time_range=use_time_range,
            start_column=start_column,
            end_column=end_column,
            duration_cache=duration_cache,
            auto_duration_cache=auto_duration_cache,
            probe_workers=probe_workers,
        )

    def _cache_path(self) -> Path | None:
        """Where measured durations are remembered between runs."""
        if self.params.auto_duration_cache:
            return self.config.run.output_root / self.name / ".pov_durations.json"
        return self.params.duration_cache

    @staticmethod
    def _parse_buckets(r: Reader) -> list[Bucket]:
        edges = r.list("edges", item_type=(int, float), min_len=2)
        labels_raw = r.raw("labels", None)
        r.done()
        labels = None
        if labels_raw is not None:
            if not isinstance(labels_raw, list) or not all(
                isinstance(item, str) for item in labels_raw
            ):
                raise ConfigError("params.buckets.labels: expected a list of strings")
            labels = labels_raw
        try:
            return build_buckets([float(edge) for edge in edges], labels)
        except SamplingError as exc:
            raise ConfigError(f"params.buckets: {exc}") from exc

    # -- preflight ---------------------------------------------------------

    def check_inputs(self) -> list[str]:
        params = self.params
        problems: list[str] = []
        hint = "(run: ./scripts/download_how2sign.sh)"

        if not params.metadata_csv.exists():
            problems.append(f"metadata_csv not found: {params.metadata_csv}  {hint}")
        if not params.video_dir.is_dir():
            problems.append(f"video_dir not found: {params.video_dir}  {hint}")
        else:
            if not any(params.video_dir.glob(f"*{params.video_extension}")):
                problems.append(
                    f"video_dir contains no {params.video_extension} files: "
                    f"{params.video_dir}  {hint}"
                )
        return problems

    def describe_inputs(self) -> list[str]:
        params = self.params
        n_videos = sum(1 for _ in params.video_dir.glob(f"*{params.video_extension}"))
        return [f"source     : {n_videos} clips in {params.video_dir}"]

    # -- generation --------------------------------------------------------

    def generate(self, layout: RunLayout, manifest: ManifestWriter) -> dict:
        params = self.params
        started = time.perf_counter()

        records = self._read_metadata()
        records = self._attach_paths(records)
        missing = [rec for rec in records if not rec["_path"].exists()]
        records = [rec for rec in records if rec["_path"].exists()]
        if not records:
            raise ConfigError(
                f"params.video_dir: none of the {len(missing)} videos listed in "
                f"{params.metadata_csv} were found under {params.video_dir}\n"
                f"  Expected files like: {missing[0]['_path'].name}\n"
                "  Download them with:  ./scripts/download_how2sign.sh\n"
                "  (the sentence-level 'Green Screen RGB Clips', not the "
                "full-length videos)"
            )

        probed_at = time.perf_counter()
        n_probed = self._attach_durations(records)
        records = [rec for rec in records if rec["_duration"] is not None]
        probe_elapsed = time.perf_counter() - probed_at

        try:
            sampled, counts = sample_buckets(
                records,
                params.buckets,
                key=lambda rec: rec["_duration"],
                per_bucket=params.samples_per_bucket,
                total=params.total_samples,
                seed=self.config.run.seed,
            )
        except SamplingError as exc:
            raise ConfigError(f"params: {exc}") from exc

        rows: list[Any] = []
        errors: list[tuple[str, str]] = []
        skipped = 0

        workers = min(self.config.run.workers, max(1, len(sampled)))
        with self.progress(len(sampled), unit="clip") as bar:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._process_clip, record, layout): record
                    for record in sampled
                }
                for future in as_completed(futures):
                    record = futures[future]
                    bar.update()
                    try:
                        row, was_skipped = future.result()
                    except Exception as exc:
                        errors.append((record["_sample_id"], f"{type(exc).__name__}: {exc}"))
                        continue
                    rows.append(row)
                    skipped += int(was_skipped)

        for row in sorted(rows, key=lambda r: r.sample_id):
            manifest.add(row)

        return {
            "skipped": skipped,
            "errors": errors,
            "elapsed_sec": round(time.perf_counter() - started, 2),
            "metadata_rows": len(records) + len(missing),
            "videos_missing": len(missing),
            "videos_probed": n_probed,
            "probe_sec": round(probe_elapsed, 2),
            "bucket_counts": counts,
        }

    # -- metadata ----------------------------------------------------------

    def _read_metadata(self) -> list[dict]:
        params = self.params
        path = params.metadata_csv
        if not path.exists():
            raise ConfigError(
                f"params.metadata_csv: file not found: {path}\n"
                "  Download the How2Sign source data with:\n"
                "      pip install gdown\n"
                "      ./scripts/download_how2sign.sh\n"
                "  Or point params.metadata_csv / params.video_dir at an "
                "existing How2Sign copy."
            )

        with open(path, encoding="utf-8", newline="") as f:
            sample = f.read(8192)
            f.seek(0)
            delimiter = params.delimiter
            if delimiter == "auto":
                delimiter = _sniff_delimiter(sample)
            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None:
                raise ConfigError(f"params.metadata_csv: {path} is empty")
            for column in (params.id_column, params.text_column):
                if column not in reader.fieldnames:
                    raise ConfigError(
                        f"params.metadata_csv: column {column!r} not found in {path}. "
                        f"Columns: {reader.fieldnames}"
                    )
            records = [dict(row) for row in reader]

        if not records:
            raise ConfigError(f"params.metadata_csv: {path} has no data rows")
        return records

    def _attach_paths(self, records: list[dict]) -> list[dict]:
        params = self.params
        seen: set[str] = set()
        out: list[dict] = []
        for record in records:
            name = (record.get(params.id_column) or "").strip()
            if not name:
                continue
            if name in seen:  # How2Sign repeats sentence names across splits
                continue
            seen.add(name)
            record["_name"] = name
            record["_sample_id"] = _safe_id(name)
            record["_path"] = params.video_dir / f"{name}{params.video_extension}"
            out.append(record)
        return out

    def _attach_durations(self, records: list[dict]) -> int:
        """Fill in `_duration` for each record. Returns how many were probed."""
        params = self.params

        if params.duration_column:
            for record in records:
                record["_duration"] = _as_float(record.get(params.duration_column))
            return 0

        cache = _DurationCache(self._cache_path())
        cache.load()

        pending = []
        for record in records:
            cached = cache.get(record["_path"])
            record["_duration"] = cached
            if cached is None:
                pending.append(record)

        if pending:
            # Measuring thousands of videos is the slow half of an uncached ASL
            # run, so it gets its own bar rather than looking like a hang.
            with self.progress(len(pending), unit="probe") as bar:
                bar.set_description(f"{self.name}: measuring durations")
                with ThreadPoolExecutor(max_workers=params.probe_workers) as pool:
                    futures = {
                        pool.submit(_probe_duration, record["_path"]): record
                        for record in pending
                    }
                    for future in as_completed(futures):
                        record = futures[future]
                        duration = future.result()
                        record["_duration"] = duration
                        if duration is not None:
                            cache.put(record["_path"], duration)
                        bar.update()
            cache.save()

        return len(pending)

    # -- clip production ---------------------------------------------------

    def _process_clip(self, record: dict, layout: RunLayout) -> tuple[Any, bool]:
        params = self.params
        sample_id = record["_sample_id"]
        source = record["_path"]
        dest = layout.media_file(f"{sample_id}{params.video_extension}")

        start = None
        duration = None
        if params.use_time_range:
            start = _as_float(record.get(params.start_column))
            end = _as_float(record.get(params.end_column))
            if start is not None and end is not None and end > start:
                duration = end - start
        if params.max_clip_seconds is not None:
            capped = params.max_clip_seconds
            duration = capped if duration is None else min(duration, capped)

        was_skipped = False
        if dest.exists() and self.config.run.resume:
            was_skipped = True
            stats = probe_video(dest)
        else:
            stats = cut_clip(
                source,
                dest,
                self.encode,
                start=start,
                duration=duration,
                scale_height=params.scale_height,
                target_fps=params.target_fps,
            )

        sentence = (record.get(params.text_column) or "").strip()
        bucket = _bucket_label(record["_duration"], params.buckets)

        row = self.make_row(
            sample_id=sample_id,
            condition=f"video_{bucket}" if bucket else "video",
            media_type="video",
            media_path=layout.relpath(dest),
            ground_truth=sentence,
            fps=stats.fps,
            n_frames=stats.n_frames,
            duration_sec=round(stats.duration_sec, 3),
            width=stats.width,
            height=stats.height,
            codec=stats.codec,
            file_size_bytes=stats.file_size_bytes,
            bucket=bucket or "",
            source_name=record["_name"],
            source_duration_sec=round(float(record["_duration"]), 3),
            source_path=str(source),
            video_id=record.get("VIDEO_ID", ""),
            sentence_id=record.get("SENTENCE_ID", ""),
        )
        return row, was_skipped


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


class _DurationCache:
    """Path → duration, invalidated by file size and mtime."""

    def __init__(self, path: Path | None):
        self.path = path
        self._data: dict[str, list] = {}
        self._dirty = False

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._data = data
        except (json.JSONDecodeError, OSError):
            self._data = {}  # a corrupt cache is not worth failing a run over

    def get(self, path: Path) -> float | None:
        entry = self._data.get(str(path))
        if not entry or len(entry) != 3:
            return None
        size, mtime, duration = entry
        try:
            stat = path.stat()
        except OSError:
            return None
        if int(size) != stat.st_size or abs(float(mtime) - stat.st_mtime) > 1e-6:
            return None
        return float(duration)

    def put(self, path: Path, duration: float) -> None:
        try:
            stat = path.stat()
        except OSError:
            return
        self._data[str(path)] = [stat.st_size, stat.st_mtime, duration]
        self._dirty = True

    def save(self) -> None:
        if self.path is None or not self._dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f)
            tmp.replace(self.path)
        except OSError:
            pass  # caching is an optimisation, never a hard requirement


def _probe_duration(path: Path) -> float | None:
    try:
        return probe_video(path).duration_sec
    except Exception:
        return None


def _bucket_label(seconds: float | None, buckets: tuple[Bucket, ...]) -> str | None:
    if seconds is None:
        return None
    for bucket in buckets:
        if bucket.contains(seconds):
            return bucket.label
    return None


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
    except csv.Error:
        # How2Sign ships tab-separated; fall back to whichever appears more.
        return "\t" if sample.count("\t") > sample.count(",") else ","


def _safe_id(name: str) -> str:
    cleaned = _UNSAFE.sub("_", name).strip("._-")
    return cleaned or "clip"


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_positive_int(r: Reader, key: str, default: Any = None) -> int | None:
    value = r.raw(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"params.{key}: expected a positive integer or null, got {value!r}")
    return value


def _optional_positive_float(r: Reader, key: str, default: Any = None) -> float | None:
    value = r.raw(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigError(f"params.{key}: expected a positive number or null, got {value!r}")
    return float(value)
