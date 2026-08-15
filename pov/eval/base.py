"""Scorer interface shared by the three experiment scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

#: Prefix for every column the evaluator adds, so scores can never collide
#: with a manifest column.
SCORE_PREFIX = "score_"


class Scorer(ABC):
    """Scores one row of a manifest against its `model_output`."""

    #: Experiment name this scorer handles.
    experiment: str = ""

    #: Numeric metrics produced, in report order.
    metrics: tuple[str, ...] = ()

    #: The metric used when a single number is wanted (e.g. sorting a report).
    primary_metric: str = ""

    #: Non-numeric diagnostic columns produced (not averaged in summaries).
    details: tuple[str, ...] = ()

    @abstractmethod
    def score(self, row: Mapping[str, Any], ground_truth: str) -> dict:
        """Return ``{metric: value}`` for one row.

        `ground_truth` is already resolved — inline for short values, read from
        `ground_truth_path` for long ones.

        This must handle an empty or missing `model_output` sensibly: the
        evaluator does **not** special-case it. Zeroing every metric would be
        wrong for error-style metrics (`wer: 0.0` means a perfect transcription)
        and for ground-truth counts (`moves_expected`), so each scorer reports
        what an empty prediction really earns.
        """
