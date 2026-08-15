"""ASL translation scoring.

Two layers, both optional to use:

* **Deterministic metrics** — exact match, token F1, smoothed BLEU-4, character
  similarity and WER, computed from the ground-truth sentence. No network, no
  model calls, fully reproducible.
* **Judge passthrough** — if the input CSV carries `judge_strict` /
  `judge_loose` columns (filled in by whatever judge you like), they are parsed
  and aggregated alongside the deterministic metrics. `pov` never calls a judge
  itself; it only reports what you supply.
"""

from __future__ import annotations

from typing import Any, Mapping

from pov.eval.base import Scorer
from pov.eval.text import bleu, char_similarity, exact_match, token_f1, word_error_rate

JUDGE_COLUMNS = ("judge_strict", "judge_loose")

_TRUE = {"1", "true", "yes", "y", "t", "correct", "pass"}
_FALSE = {"0", "false", "no", "n", "f", "incorrect", "fail"}


def parse_judge(value: Any) -> float | None:
    """Interpret a judge cell as 1.0 / 0.0, or None when absent/unparseable."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in _TRUE:
        return 1.0
    if text in _FALSE:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return None


class AslScorer(Scorer):
    experiment = "asl"
    metrics = ("exact_match", "token_f1", "bleu", "char_similarity", "wer")
    primary_metric = "token_f1"

    def score(self, row: Mapping[str, Any], ground_truth: str) -> dict:
        hypothesis = str(row.get("model_output") or "")
        reference = ground_truth or ""

        scores: dict = {
            "exact_match": round(exact_match(reference, hypothesis), 6),
            "token_f1": round(token_f1(reference, hypothesis), 6),
            "bleu": round(bleu(reference, hypothesis), 6),
            "char_similarity": round(char_similarity(reference, hypothesis), 6),
            "wer": round(word_error_rate(reference, hypothesis), 6),
        }

        # Judge columns are reported only when the input actually has them.
        for column in JUDGE_COLUMNS:
            if column in row:
                parsed = parse_judge(row[column])
                if parsed is not None:
                    scores[column] = parsed
        return scores
