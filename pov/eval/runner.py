"""Evaluation driver.

Reads a manifest CSV that has been given a `model_output` column, scores every
row with the scorer for its experiment, and writes:

    scored.csv    every input column plus `score_*` columns
    summary.csv   mean of each metric, grouped by condition (and `model` if present)

`pov eval` never calls a model. It only reads what you put in `model_output`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pov.eval.asl import AslScorer
from pov.eval.base import SCORE_PREFIX, Scorer
from pov.eval.chess import ChessScorer
from pov.eval.mcq import McqScorer
from pov.manifest import MODEL_OUTPUT_COLUMN, ManifestError, load_ground_truth, read_manifest

from pov.errors import PovError

#: experiment name → scorer
SCORERS: dict[str, type[Scorer]] = {
    ChessScorer.experiment: ChessScorer,
    AslScorer.experiment: AslScorer,
    McqScorer.experiment: McqScorer,
}

#: Grouping columns applied when present in the input.
_OPTIONAL_GROUPS = ("model", "run_id")


class EvalError(PovError, ValueError):
    """Raised for unusable evaluation input."""


def get_scorer(experiment: str) -> Scorer:
    try:
        return SCORERS[experiment]()
    except KeyError:
        raise EvalError(
            f"no scorer for experiment {experiment!r}. Known: {sorted(SCORERS)}"
        ) from None


@dataclass
class EvalReport:
    """Scored rows plus aggregate tables."""

    rows: list[dict] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    group_columns: list[str] = field(default_factory=list)
    summary: list[dict] = field(default_factory=list)
    experiments: list[str] = field(default_factory=list)
    n_missing_output: int = 0
    scored_path: Path | None = None
    summary_path: Path | None = None

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def overall(self) -> dict:
        """Mean of each metric across every row."""
        return {metric: _mean(_values(self.rows, metric)) for metric in self.metrics}

    def text_summary(self) -> str:
        lines = [
            f"Scored {self.n_rows} row(s) across {', '.join(self.experiments) or '—'}",
        ]
        if self.n_missing_output:
            lines.append(
                f"  {self.n_missing_output} row(s) had an empty model_output (scored 0)"
            )
        overall = self.overall()
        if overall:
            lines.append("  overall: " + "  ".join(
                f"{name}={value:.4f}" for name, value in overall.items() if value is not None
            ))
        if self.summary:
            lines.append("")
            lines.append(_format_table(self.summary, self.group_columns, self.metrics))
        return "\n".join(lines)


def evaluate(
    input_csv: str | Path,
    output_dir: str | Path | None = None,
    *,
    run_dir: str | Path | None = None,
    group_by: Sequence[str] | None = None,
    write: bool = True,
) -> EvalReport:
    """Score `input_csv` and (optionally) write scored.csv and summary.csv."""
    input_path = Path(input_csv)
    rows = read_manifest(input_path)
    if not rows:
        raise EvalError(f"{input_path}: no rows to score")
    if MODEL_OUTPUT_COLUMN not in rows[0]:
        raise EvalError(
            f"{input_path}: no {MODEL_OUTPUT_COLUMN!r} column. Add your model's "
            "predictions to that column before running eval."
        )

    # Ground-truth files are recorded relative to the run directory, which is
    # where the manifest lives unless told otherwise.
    base_dir = Path(run_dir) if run_dir is not None else input_path.parent

    scorers: dict[str, Scorer] = {}
    scored_rows: list[dict] = []
    metric_names: list[str] = []
    missing_output = 0

    for index, row in enumerate(rows, start=1):
        experiment = (row.get("experiment") or "").strip()
        if not experiment:
            raise EvalError(f"{input_path}: row {index} has no 'experiment' value")
        if experiment not in scorers:
            scorers[experiment] = get_scorer(experiment)
        scorer = scorers[experiment]

        try:
            ground_truth = load_ground_truth(row, base_dir)
        except ManifestError as exc:
            raise EvalError(str(exc)) from exc

        # An empty prediction is still scored by the real scorer rather than
        # being zeroed. Zeroing every metric is wrong twice over: it reports
        # `wer: 0.0` — a *perfect* transcription — for a model that produced
        # nothing, and it flattens ground-truth counts like `moves_expected`
        # to 0. The scorers already handle empty input correctly.
        if not (row.get(MODEL_OUTPUT_COLUMN) or "").strip():
            missing_output += 1
        scores = scorer.score(row, ground_truth)

        scored = dict(row)
        for name, value in scores.items():
            column = f"{SCORE_PREFIX}{name}"
            scored[column] = value
            if column not in metric_names and _is_numeric(value):
                metric_names.append(column)
        scored_rows.append(scored)

    # A metric only some rows produced (a judge column on part of the data)
    # must not read as 0 elsewhere — leave those cells blank.
    for row in scored_rows:
        for column in metric_names:
            row.setdefault(column, "")

    group_columns = _resolve_groups(rows[0], group_by)
    summary = summarise(scored_rows, group_columns, metric_names)

    report = EvalReport(
        rows=scored_rows,
        metrics=metric_names,
        group_columns=group_columns,
        summary=summary,
        experiments=sorted({(r.get("experiment") or "") for r in rows}),
        n_missing_output=missing_output,
    )

    if write:
        target = Path(output_dir) if output_dir else input_path.parent
        target.mkdir(parents=True, exist_ok=True)
        report.scored_path = _write_csv(target / "scored.csv", scored_rows)
        report.summary_path = _write_csv(target / "summary.csv", summary)

    return report


def summarise(
    rows: Sequence[Mapping], group_columns: Sequence[str], metrics: Sequence[str]
) -> list[dict]:
    """Mean of each metric per group, plus a row count."""
    if not rows:
        return []

    grouped: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(column, "")) for column in group_columns)
        grouped[key].append(row)

    summary: list[dict] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        record: dict = dict(zip(group_columns, key))
        record["n"] = len(group_rows)
        for metric in metrics:
            mean = _mean(_values(group_rows, metric))
            record[metric] = "" if mean is None else round(mean, 6)
        summary.append(record)
    return summary


def _resolve_groups(sample_row: Mapping, group_by: Sequence[str] | None) -> list[str]:
    if group_by:
        missing = [column for column in group_by if column not in sample_row]
        if missing:
            raise EvalError(f"group-by column(s) not in input: {missing}")
        return list(group_by)

    groups = ["experiment", "condition"]
    groups.extend(column for column in _OPTIONAL_GROUPS if column in sample_row)
    return [column for column in groups if column in sample_row]


def _values(rows: Iterable[Mapping], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(column, "")
        if value == "" or value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _is_numeric(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str) and value.strip():
        try:
            float(value)
            return True
        except ValueError:
            return False
    return False


def _write_csv(path: Path, rows: Sequence[Mapping]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    tmp.replace(path)
    return path


def _format_table(summary: Sequence[Mapping], groups: Sequence[str],
                  metrics: Sequence[str]) -> str:
    columns = [*groups, "n", *metrics]
    widths = {
        column: max(len(_short(column)), *(len(_cell(row.get(column, ""))) for row in summary))
        for column in columns
    }
    lines = ["  " + "  ".join(_short(c).ljust(widths[c]) for c in columns)]
    lines.append("  " + "  ".join("-" * widths[c] for c in columns))
    for row in summary:
        lines.append(
            "  " + "  ".join(_cell(row.get(c, "")).ljust(widths[c]) for c in columns)
        )
    return "\n".join(lines)


def _short(column: str) -> str:
    return column[len(SCORE_PREFIX):] if column.startswith(SCORE_PREFIX) else column


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
