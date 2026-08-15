"""Chess move-transcription scoring.

Three scores, all with the transcript length as denominator, ported from the
original `chess_game_creator/eval.py`:

``strict``  longest correct prefix — stops at the first mistake.
``loose``   longest common subsequence — right moves in the right order, gaps allowed.
``hybrid``  only moves inside a run of >= 2 consecutive correct moves count.

By construction ``strict <= hybrid <= loose``.

Moves are matched on **(colour, from-square, to-square)**. Piece names are
parsed but deliberately ignored: models frequently name the piece wrongly while
reading the squares correctly, and the squares are what the task is about.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from pov.eval.base import Scorer

# "1      White     b2        b3"  — the generated transcript table
_TABLE_ROW = re.compile(
    r"^\s*(\d+)\s+(white|black)\s+([a-h][1-8])\s+([a-h][1-8])\s*$", re.IGNORECASE
)
# "1 White b2 b3; 1 Black c7 c5"  — the compact manifest cell
_COMPACT = re.compile(r"(\d+)\s+(white|black)\s+([a-h][1-8])\s+([a-h][1-8])", re.IGNORECASE)

# Model output, most specific pattern first.
_MODEL_PATTERNS = (
    re.compile(
        r"move\s+(\d+)\s*[:\-.]?\s*(white|black)\s+(?:[a-z]+\s+)?"
        r"([a-h][1-8])\s*(?:->|to|-|—|\s)?\s*([a-h][1-8])",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d+)\s*[.)]\s*(white|black)\s+(?:[a-z]+\s+)?"
        r"([a-h][1-8])\s*(?:->|to|-|—|\s)?\s*([a-h][1-8])",
        re.IGNORECASE,
    ),
    re.compile(
        r"(white|black)\s+(?:[a-z]+\s+)?([a-h][1-8])\s*(?:->|to|-|—|\s)?\s*([a-h][1-8])",
        re.IGNORECASE,
    ),
)


def parse_transcript(text: str, n_moves: int | None = None) -> list[dict]:
    """Parse ground truth in either the table or the compact format."""
    moves: list[dict] = []
    for line in text.splitlines():
        match = _TABLE_ROW.match(line)
        if match:
            moves.append(_move(match.group(1), match.group(2), match.group(3), match.group(4)))

    if not moves:
        for match in _COMPACT.finditer(text):
            moves.append(_move(*match.groups()))

    if n_moves is not None:
        moves = [m for m in moves if m["move_num"] <= n_moves]

    moves.sort(key=lambda m: (m["move_num"], 0 if m["color"] == "White" else 1))
    return moves


def parse_model_output(text: str) -> list[dict]:
    """Parse a model's free-form move list.

    Tries the most explicit pattern first and stops at the first one that
    matches, so a numbered list is never re-parsed by the looser fallback.
    """
    if not text:
        return []
    for pattern in _MODEL_PATTERNS[:2]:
        found = pattern.findall(text)
        if found:
            return [_move(num, color, src, dst) for num, color, src, dst in found]

    found = _MODEL_PATTERNS[2].findall(text)
    return [
        _move(index, color, src, dst)
        for index, (color, src, dst) in enumerate(found, start=1)
    ]


def _move(move_num: Any, color: str, src: str, dst: str) -> dict:
    return {
        "move_num": int(move_num),
        "color": str(color).capitalize(),
        "from": str(src).lower(),
        "to": str(dst).lower(),
    }


def _key(move: Mapping) -> tuple:
    return (move["color"], move["from"], move["to"])


def lcs_alignment(a: Sequence[Mapping], b: Sequence[Mapping]) -> list[tuple[int, int]]:
    """Longest common subsequence alignment as (index_in_a, index_in_b) pairs."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []

    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if _key(a[i - 1]) == _key(b[j - 1]):
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])

    alignment: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if _key(a[i - 1]) == _key(b[j - 1]):
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif table[i - 1][j] >= table[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return list(reversed(alignment))


def strict_score(truth: Sequence[Mapping], predicted: Sequence[Mapping]) -> float:
    """Fraction of the transcript matched as an unbroken prefix."""
    if not truth:
        return 0.0
    matched = 0
    for index, move in enumerate(truth):
        if index < len(predicted) and _key(move) == _key(predicted[index]):
            matched += 1
        else:
            break
    return matched / len(truth)


def loose_score(truth: Sequence[Mapping], predicted: Sequence[Mapping]) -> float:
    """LCS length over transcript length."""
    if not truth:
        return 0.0
    return len(lcs_alignment(truth, predicted)) / len(truth)


def hybrid_score(truth: Sequence[Mapping], predicted: Sequence[Mapping]) -> float:
    """Only moves in a run of >= 2 consecutive correct moves count."""
    if not truth:
        return 0.0
    alignment = lcs_alignment(truth, predicted)
    if not alignment:
        return 0.0

    runs: list[int] = []
    length = 1
    for previous, current in zip(alignment, alignment[1:]):
        if current[0] == previous[0] + 1:
            length += 1
        else:
            runs.append(length)
            length = 1
    runs.append(length)

    paired = sum(run for run in runs if run >= 2)
    return paired / len(truth)


class ChessScorer(Scorer):
    experiment = "chess"
    metrics = ("strict", "loose", "hybrid", "moves_matched", "moves_expected", "moves_predicted")
    primary_metric = "loose"

    def score(self, row: Mapping[str, Any], ground_truth: str) -> dict:
        n_moves = _as_int(row.get("n_half_moves"))
        truth = parse_transcript(ground_truth)
        # n_half_moves counts half-moves; the transcript is capped by clip length.
        if n_moves is not None and len(truth) > n_moves:
            truth = truth[:n_moves]

        predicted = parse_model_output(str(row.get("model_output") or ""))

        return {
            "strict": round(strict_score(truth, predicted), 6),
            "loose": round(loose_score(truth, predicted), 6),
            "hybrid": round(hybrid_score(truth, predicted), 6),
            "moves_matched": len(lcs_alignment(truth, predicted)),
            "moves_expected": len(truth),
            "moves_predicted": len(predicted),
        }


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
