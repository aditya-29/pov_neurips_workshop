"""Evaluation half of pov.

Independent of :mod:`pov.experiments` — it reads a CSV and nothing else, and
never calls a model. Give it a manifest with a filled-in `model_output` column
and it produces per-row scores and aggregate tables.
"""

from pov.eval.asl import AslScorer
from pov.eval.base import SCORE_PREFIX, Scorer
from pov.eval.chess import ChessScorer
from pov.eval.mcq import McqScorer
from pov.eval.runner import (
    SCORERS,
    EvalError,
    EvalReport,
    evaluate,
    get_scorer,
    summarise,
)

__all__ = [
    "SCORE_PREFIX",
    "SCORERS",
    "AslScorer",
    "ChessScorer",
    "EvalError",
    "EvalReport",
    "McqScorer",
    "Scorer",
    "evaluate",
    "get_scorer",
    "summarise",
]
