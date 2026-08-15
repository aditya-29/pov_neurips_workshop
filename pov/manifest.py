"""The manifest CSV — the single hand-off between generation and evaluation.

Generation writes one row per media artifact. Every row carries the ground
truth, the media's real properties (fps, frame count, duration, resolution),
and a flattened copy of the config that produced it, so a row is
self-describing even if it is pulled out of its run folder.

The last column is `model_output`, written empty. Fill it in with your model's
prediction and hand the same file to `pov eval`.

Column order is stable: core columns, then experiment-specific columns
(alphabetical), then `cfg_*` columns (alphabetical), then `model_output`.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pov.errors import PovError

# Ground truth longer than this is written to ground_truth/ and referenced by
# path instead of being inlined; the CSV cell keeps a truncated preview.
MAX_INLINE_GROUND_TRUTH = 2000

MODEL_OUTPUT_COLUMN = "model_output"

#: Columns present for every experiment, in order.
CORE_FIELDS: tuple[str, ...] = (
    "sample_id",
    "experiment",
    "condition",
    "media_type",
    "media_path",
    "media_filename",
    "ground_truth",
    "ground_truth_path",
    "fps",
    "n_frames",
    "duration_sec",
    "width",
    "height",
    "codec",
    "file_size_bytes",
    "seed",
    "run_id",
    "config_hash",
    "pov_version",
    "generated_at",
)

_MEDIA_TYPES = ("video", "image")


class ManifestError(PovError, ValueError):
    """Raised for malformed manifest rows or files."""


@dataclass
class ManifestRow:
    """One generated artifact.

    `extra` holds experiment-specific columns (e.g. `n_half_moves` for chess,
    `bucket` for ASL, `speed_wps` for word-by-word MCQ).
    """

    sample_id: str
    experiment: str
    condition: str
    media_type: str
    media_path: str
    ground_truth: str = ""
    ground_truth_path: str = ""
    fps: float | None = None
    n_frames: int | None = None
    duration_sec: float | None = None
    width: int | None = None
    height: int | None = None
    codec: str = ""
    file_size_bytes: int | None = None
    seed: int | None = None
    run_id: str = ""
    config_hash: str = ""
    pov_version: str = ""
    generated_at: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ManifestError("sample_id must not be empty")
        if not self.media_path:
            raise ManifestError(f"{self.sample_id}: media_path must not be empty")
        if self.media_type not in _MEDIA_TYPES:
            raise ManifestError(
                f"{self.sample_id}: media_type must be one of {list(_MEDIA_TYPES)}, "
                f"got {self.media_type!r}"
            )
        reserved = set(CORE_FIELDS) | {MODEL_OUTPUT_COLUMN}
        clashes = sorted(reserved & set(self.extra))
        if clashes:
            raise ManifestError(
                f"{self.sample_id}: extra column(s) {clashes} collide with reserved "
                "manifest columns"
            )

    @property
    def media_filename(self) -> str:
        return Path(self.media_path).name

    def to_dict(self) -> dict:
        row: dict = {
            "sample_id": self.sample_id,
            "experiment": self.experiment,
            "condition": self.condition,
            "media_type": self.media_type,
            "media_path": self.media_path,
            "media_filename": self.media_filename,
            "ground_truth": self.ground_truth,
            "ground_truth_path": self.ground_truth_path,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "duration_sec": self.duration_sec,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "file_size_bytes": self.file_size_bytes,
            "seed": self.seed,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "pov_version": self.pov_version,
            "generated_at": self.generated_at,
        }
        row.update(self.extra)
        return row


class ManifestWriter:
    """Collects rows and writes the manifest atomically.

    Rows are buffered so the column union is known before anything is written;
    the file is then written to a temp file and renamed, so an interrupted run
    never leaves a half-written manifest that eval would silently accept.
    """

    def __init__(self, path: str | Path, config_columns: Mapping[str, Any] | None = None):
        self.path = Path(path)
        self.config_columns = dict(config_columns or {})
        bad = sorted(k for k in self.config_columns if not k.startswith("cfg_"))
        if bad:
            raise ManifestError(f"config columns must start with 'cfg_': {bad}")
        self._rows: list[ManifestRow] = []
        self._ids: set[str] = set()

    # -- collection --------------------------------------------------------

    def add(self, row: ManifestRow) -> None:
        if row.sample_id in self._ids:
            raise ManifestError(f"duplicate sample_id {row.sample_id!r}")
        self._ids.add(row.sample_id)
        self._rows.append(row)

    def extend(self, rows: Iterable[ManifestRow]) -> None:
        for row in rows:
            self.add(row)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> list[ManifestRow]:
        return list(self._rows)

    # -- output ------------------------------------------------------------

    def fieldnames(self) -> list[str]:
        extra_keys: set[str] = set()
        for row in self._rows:
            extra_keys.update(row.extra)
        return [
            *CORE_FIELDS,
            *sorted(extra_keys),
            *sorted(self.config_columns),
            MODEL_OUTPUT_COLUMN,
        ]

    def write(self, sort_key=None) -> Path:
        """Write the manifest. Returns the path written."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = self.fieldnames()

        rows = sorted(self._rows, key=sort_key) if sort_key else self._rows

        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                record = {name: "" for name in fieldnames}
                record.update(self.config_columns)
                record.update(
                    {k: _cell(v) for k, v in row.to_dict().items()}
                )
                record[MODEL_OUTPUT_COLUMN] = ""
                writer.writerow(record)
        os.replace(tmp, self.path)
        return self.path


def _cell(value: Any) -> Any:
    """Render a value for a CSV cell (None becomes empty, not the text 'None')."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        # Keep floats short and stable across platforms.
        return f"{value:.6g}"
    return value


def read_manifest(path: str | Path) -> list[dict]:
    """Read a manifest CSV into a list of dicts (all values are strings)."""
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")
    # Ground-truth cells can be large; raise the field limit for long transcripts.
    _raise_csv_field_limit()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ManifestError(f"{path}: file is empty")
        missing = [c for c in ("sample_id", "experiment") if c not in reader.fieldnames]
        if missing:
            raise ManifestError(
                f"{path}: not a pov manifest — missing column(s) {missing}"
            )
        rows = [dict(row) for row in reader]
    return rows


def store_ground_truth(
    text: str,
    *,
    filename: str,
    ground_truth_dir: Path,
    run_dir: Path,
    max_inline: int = MAX_INLINE_GROUND_TRUTH,
) -> tuple[str, str]:
    """Decide whether ground truth is inlined or written to a file.

    Returns ``(inline_text, relative_path)``. Short ground truth (an ASL
    sentence, an MCQ answer letter) is inlined and `relative_path` is empty.
    Long ground truth (a full chess transcript) is written to disk; the inline
    cell keeps a truncated preview so the CSV stays readable on its own.
    """
    if len(text) <= max_inline:
        return text, ""

    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    target = ground_truth_dir / filename
    target.write_text(text, encoding="utf-8")
    rel = target.resolve().relative_to(run_dir.resolve()).as_posix()
    preview = text[:max_inline].rstrip() + "\n…[truncated, see ground_truth_path]"
    return preview, rel


def load_ground_truth(row: Mapping[str, Any], run_dir: str | Path) -> str:
    """Return the full ground truth for a row, reading the file if referenced."""
    rel = (row.get("ground_truth_path") or "").strip()
    if rel:
        path = Path(run_dir) / rel
        if not path.exists():
            raise ManifestError(
                f"{row.get('sample_id')}: ground_truth_path {rel!r} does not exist "
                f"under {run_dir}"
            )
        return path.read_text(encoding="utf-8")
    return row.get("ground_truth", "") or ""


def _raise_csv_field_limit() -> None:
    """csv defaults to 128 KiB per field; chess transcripts can exceed it."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit // 2)
