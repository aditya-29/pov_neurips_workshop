"""On-disk layout for a generation run.

Every run of every experiment produces the same shape::

    <output_root>/<experiment>/<run_id>/
        media/                  generated .mp4 / .jpg files
        ground_truth/           full-text ground truth too large for a CSV cell
        config.resolved.yaml    exactly what produced this run
        manifest.jsonl          one JSON object per media artifact  ← eval reads this

Media paths recorded in the manifest are **relative to the run directory**, so a
run folder can be moved or shared without invalidating it.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from pov.errors import PovError

MEDIA_DIRNAME = "media"
GROUND_TRUTH_DIRNAME = "ground_truth"
MANIFEST_FILENAME = "manifest.jsonl"
CONFIG_FILENAME = "config.resolved.yaml"


class LayoutError(PovError, RuntimeError):
    """Raised when a run directory is in an unusable state."""


@dataclass(frozen=True)
class RunLayout:
    """Resolved paths for one generation run."""

    output_root: Path
    experiment: str
    run_id: str

    # -- construction ------------------------------------------------------

    @classmethod
    def build(cls, output_root: str | Path, experiment: str, run_id: str) -> "RunLayout":
        if not experiment:
            raise LayoutError("experiment name must not be empty")
        if not run_id:
            raise LayoutError("run_id must not be empty")
        if "/" in run_id or "\\" in run_id or run_id in (".", ".."):
            raise LayoutError(f"run_id {run_id!r} must be a single path segment")
        return cls(Path(output_root), experiment, run_id)

    # -- paths -------------------------------------------------------------

    @property
    def experiment_dir(self) -> Path:
        return self.output_root / self.experiment

    @property
    def run_dir(self) -> Path:
        return self.experiment_dir / self.run_id

    @property
    def media_dir(self) -> Path:
        return self.run_dir / MEDIA_DIRNAME

    @property
    def ground_truth_dir(self) -> Path:
        return self.run_dir / GROUND_TRUTH_DIRNAME

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / MANIFEST_FILENAME

    @property
    def config_path(self) -> Path:
        return self.run_dir / CONFIG_FILENAME

    # -- lifecycle ---------------------------------------------------------

    def create(self, overwrite: bool = False) -> "RunLayout":
        """Create the run directory tree.

        With `overwrite`, an existing run directory is deleted first. Without
        it, an existing directory is reused (resume); this is not an error
        because generators skip artifacts that already exist.
        """
        if overwrite and self.run_dir.exists():
            if not self.run_dir.is_dir():
                raise LayoutError(f"{self.run_dir} exists but is not a directory")
            shutil.rmtree(self.run_dir)
        if self.run_dir.exists() and not self.run_dir.is_dir():
            raise LayoutError(f"{self.run_dir} exists but is not a directory")

        # ground_truth/ is deliberately NOT created here. Only ground truth too
        # long for a CSV cell is written to a file — chess transcripts. An MCQ
        # answer ("B") or an ASL sentence lives in the manifest's ground_truth
        # column, so creating the directory up front would leave an empty one
        # sitting in every run of those experiments forever.
        self.media_dir.mkdir(parents=True, exist_ok=True)
        return self

    def exists(self) -> bool:
        return self.run_dir.is_dir()

    # -- helpers -----------------------------------------------------------

    def media_file(self, filename: str) -> Path:
        _check_filename(filename)
        return self.media_dir / filename

    def ground_truth_file(self, filename: str) -> Path:
        """Path for a ground-truth file, creating the directory on demand.

        Created here rather than in :meth:`create` so experiments whose ground
        truth is short enough to inline never grow an empty directory.
        """
        _check_filename(filename)
        self.ground_truth_dir.mkdir(parents=True, exist_ok=True)
        return self.ground_truth_dir / filename

    def relpath(self, path: str | Path) -> str:
        """Path relative to the run directory, in posix form for the manifest."""
        path = Path(path)
        if not path.is_absolute():
            # Already relative — normalise separators and return as-is.
            return path.as_posix()
        try:
            return path.resolve().relative_to(self.run_dir.resolve()).as_posix()
        except ValueError as exc:
            raise LayoutError(
                f"{path} is outside the run directory {self.run_dir}"
            ) from exc

    def resolve(self, relative_path: str | Path) -> Path:
        """Inverse of :meth:`relpath`."""
        return self.run_dir / Path(relative_path)


def _check_filename(filename: str) -> None:
    if not filename:
        raise LayoutError("filename must not be empty")
    if "/" in filename or "\\" in filename:
        raise LayoutError(f"filename {filename!r} must not contain a path separator")
    if filename in (".", ".."):
        raise LayoutError(f"filename {filename!r} is not a valid name")
