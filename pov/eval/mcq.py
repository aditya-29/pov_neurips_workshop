"""Multiple-choice scoring for the word-by-word MCQ experiment.

Answer extraction follows the original benchmark's rules:

1. A structured ``ANSWER: X`` wins, wherever it appears.
2. Otherwise a leading standalone letter, e.g. ``B) Cell division``.
3. Otherwise the first standalone A–D in the text.

A response that reads as a refusal *and* contains no letter is flagged rather
than scored as wrong, so refusals can be excluded from accuracy the way the
original evaluator did.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from pov.eval.base import Scorer

LETTERS = ("A", "B", "C", "D")

_ANSWER_TAG = re.compile(r"ANSWER\s*[:\-=]\s*\(?\s*([A-D])\b", re.IGNORECASE)
_LEADING_LETTER = re.compile(r"^\s*\(?([A-D])\)?[\s.:,)-]", re.IGNORECASE)
_STANDALONE = re.compile(r"\b([A-D])\b")
_OPTION_FORM = re.compile(r"\(\s*([A-D])\s*\)", re.IGNORECASE)

_REFUSAL_SIGNALS = (
    "i'm sorry", "i am sorry", "i cannot", "i can't", "i can not",
    "as an ai", "not able to assist", "i apologize", "i apologise",
    "unable to help", "can't help with that",
)


def parse_answer(text: str) -> str | None:
    """Extract the chosen letter from a model response, or None."""
    if not text:
        return None

    match = _ANSWER_TAG.search(text)
    if match:
        return match.group(1).upper()

    match = _LEADING_LETTER.match(text)
    if match:
        return match.group(1).upper()

    # A parenthesised option like "(C)" is a stronger signal than a bare letter
    # that might just be the article "a".
    match = _OPTION_FORM.search(text)
    if match:
        return match.group(1).upper()

    for candidate in _STANDALONE.findall(text):
        letter = candidate.upper()
        if letter in LETTERS:
            # "A" alone is usually the article; require it to be uppercase.
            if letter == "A" and candidate != "A":
                continue
            return letter
    return None


def is_refusal(text: str) -> bool:
    """A refusal is a refusal phrase with no answer letter anywhere."""
    if not text:
        return False
    lowered = text.lower()
    if not any(signal in lowered for signal in _REFUSAL_SIGNALS):
        return False
    return parse_answer(text) is None


class McqScorer(Scorer):
    experiment = "wbw_mcq"
    metrics = ("correct", "answered", "refusal")
    primary_metric = "correct"
    details = ("predicted_answer",)

    def score(self, row: Mapping[str, Any], ground_truth: str) -> dict:
        raw = str(row.get("model_output") or "")
        predicted = parse_answer(raw)
        refusal = is_refusal(raw)
        expected = (ground_truth or "").strip().upper()

        return {
            "correct": float(predicted is not None and predicted == expected),
            "answered": float(predicted is not None),
            "refusal": float(refusal),
            "predicted_answer": predicted or "",
        }
