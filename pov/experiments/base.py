"""Shared generator scaffolding.

A generator's only job is to produce media and describe it. Everything around
that — creating the run directory, snapshotting the config, stamping provenance
onto each row, writing the manifest atomically — happens here, identically for
every experiment.
"""

from __future__ import annotations

import datetime as _dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pov import __version__
from pov.config import Config, Reader
from pov.layout import RunLayout
from pov.manifest import ManifestRow, ManifestWriter, store_ground_truth
from pov.video import EncodeSettings


@dataclass
class GenerationResult:
    """What a completed run produced."""

    experiment: str
    run_id: str
    run_dir: Path
    manifest_path: Path
    n_rows: int
    n_skipped: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"{self.experiment}: {self.n_rows} artifact(s) → {self.run_dir}",
            f"  manifest: {self.manifest_path}",
        ]
        if self.n_skipped:
            lines.append(f"  skipped (already present): {self.n_skipped}")
        for key, value in self.stats.items():
            lines.append(f"  {key}: {value}")
        if self.errors:
            lines.append(f"  ERRORS: {len(self.errors)}")
            for sample_id, message in self.errors[:10]:
                lines.append(f"    {sample_id}: {message}")
        return "\n".join(lines)


class Generator(ABC):
    """Base class for the three experiment generators.

    Subclasses implement :meth:`parse_params` (validate `params:` from the
    YAML) and :meth:`generate` (produce media, add manifest rows).
    """

    #: Experiment name as it appears in `experiment:` and in the output path.
    name: str = ""

    def __init__(self, config: Config):
        if not self.name:
            raise NotImplementedError(f"{type(self).__name__} must set a class-level name")
        if config.experiment != self.name:
            raise ValueError(
                f"config is for experiment {config.experiment!r}, "
                f"but {type(self).__name__} generates {self.name!r}"
            )
        self.config = config
        self.run_id = config.resolve_run_id()
        self.config_hash = config.config_hash()
        self.encode = EncodeSettings.from_config(config.video)
        self.params = self.parse_params(Reader(config.params, path="params"))
        self._generated_at = _dt.datetime.now().isoformat(timespec="seconds")

    # -- subclass hooks ----------------------------------------------------

    @abstractmethod
    def parse_params(self, r: Reader) -> Any:
        """Validate and return the experiment-specific `params:` block."""

    @abstractmethod
    def generate(self, layout: RunLayout, manifest: ManifestWriter) -> dict:
        """Produce media and add manifest rows. Returns a stats dict."""

    def check_inputs(self) -> list[str]:
        """Problems that would stop this run, as human-readable strings.

        Used by ``--dry-run``: a plan that cannot execute is not a plan, so the
        dry run reports missing source data instead of printing a confident
        summary for a command that fails a second later. Experiments with no
        external inputs (chess) inherit the empty default.
        """
        return []

    # -- orchestration -----------------------------------------------------

    def build_layout(self) -> RunLayout:
        return RunLayout.build(
            self.config.run.output_root, self.name, self.run_id
        )

    def run(self) -> GenerationResult:
        layout = self.build_layout().create(overwrite=self.config.run.overwrite)
        self.config.write_snapshot(layout.config_path)

        manifest = ManifestWriter(
            layout.manifest_path, config_columns=self.config.flatten()
        )
        stats = self.generate(layout, manifest) or {}
        manifest.write(sort_key=lambda row: row.sample_id)

        return GenerationResult(
            experiment=self.name,
            run_id=self.run_id,
            run_dir=layout.run_dir,
            manifest_path=layout.manifest_path,
            n_rows=len(manifest),
            n_skipped=int(stats.pop("skipped", 0) or 0),
            errors=list(stats.pop("errors", []) or []),
            stats=stats,
        )

    # -- helpers for subclasses -------------------------------------------

    def make_row(
        self,
        *,
        sample_id: str,
        condition: str,
        media_type: str,
        media_path: str,
        ground_truth: str = "",
        ground_truth_path: str = "",
        fps: float | None = None,
        n_frames: int | None = None,
        duration_sec: float | None = None,
        width: int | None = None,
        height: int | None = None,
        codec: str = "",
        file_size_bytes: int | None = None,
        seed: int | None = None,
        **extra: Any,
    ) -> ManifestRow:
        """Build a manifest row with provenance already filled in."""
        return ManifestRow(
            sample_id=sample_id,
            experiment=self.name,
            condition=condition,
            media_type=media_type,
            media_path=media_path,
            ground_truth=ground_truth,
            ground_truth_path=ground_truth_path,
            fps=fps,
            n_frames=n_frames,
            duration_sec=duration_sec,
            width=width,
            height=height,
            codec=codec,
            file_size_bytes=file_size_bytes,
            seed=self.config.run.seed if seed is None else seed,
            run_id=self.run_id,
            config_hash=self.config_hash,
            pov_version=__version__,
            generated_at=self._generated_at,
            extra=extra,
        )

    def store_ground_truth(self, layout: RunLayout, text: str, filename: str) -> tuple[str, str]:
        """Inline short ground truth; write long ground truth to a file."""
        return store_ground_truth(
            text,
            filename=filename,
            ground_truth_dir=layout.ground_truth_dir,
            run_dir=layout.run_dir,
        )
