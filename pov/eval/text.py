"""Deterministic text metrics — pure python, no network, no model calls.

Used by the ASL scorer, and available to any other free-text comparison. Every
function here is a pure function of two strings, so results are reproducible
and unit-testable.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Sequence

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE = re.compile(r"\s+")
_ARTICLES = frozenset({"a", "an", "the"})


def normalise(text: str, *, strip_articles: bool = True) -> str:
    """Lowercase, strip punctuation/accents/articles, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    tokens = _WHITESPACE.sub(" ", text).strip().split()
    if strip_articles:
        tokens = [t for t in tokens if t not in _ARTICLES]
    return " ".join(tokens)


def tokenize(text: str, *, strip_articles: bool = True) -> list[str]:
    normalised = normalise(text, strip_articles=strip_articles)
    return normalised.split() if normalised else []


def exact_match(reference: str, hypothesis: str) -> float:
    """1.0 if the two normalise to the same string."""
    return float(normalise(reference) == normalise(hypothesis))


def token_f1(reference: str, hypothesis: str) -> float:
    """SQuAD-style token overlap F1 (multiset intersection)."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0

    common = Counter(ref_tokens) & Counter(hyp_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Sentence-level BLEU with add-one smoothing on higher-order n-grams.

    Smoothing matters here: single ASL sentences are short, and an unsmoothed
    BLEU-4 collapses to 0 whenever any order has no match, which throws away
    the signal from the orders that did match.
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0

    log_precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(_ngrams(ref_tokens, n))
        hyp_ngrams = Counter(_ngrams(hyp_tokens, n))
        total = sum(hyp_ngrams.values())
        if total == 0:
            # Hypothesis is shorter than n; nothing to measure at this order.
            continue
        overlap = sum((hyp_ngrams & ref_ngrams).values())
        if n == 1:
            if overlap == 0:
                return 0.0
            precision = overlap / total
        else:
            precision = (overlap + 1) / (total + 1)  # add-one smoothing
        log_precisions.append(math.log(precision))

    if not log_precisions:
        return 0.0

    geometric_mean = math.exp(sum(log_precisions) / len(log_precisions))
    return _brevity_penalty(len(ref_tokens), len(hyp_tokens)) * geometric_mean


def _brevity_penalty(ref_len: int, hyp_len: int) -> float:
    if hyp_len == 0:
        return 0.0
    if hyp_len > ref_len:
        return 1.0
    return math.exp(1 - ref_len / hyp_len)


def _ngrams(tokens: Sequence[str], n: int) -> list[tuple]:
    if n <= 0:
        raise ValueError(f"n must be >= 1, got {n}")
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings (character level)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        current = [i]
        for j, ch_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + (ch_a != ch_b),  # substitution
                )
            )
        previous = current
    return previous[-1]


def char_similarity(reference: str, hypothesis: str) -> float:
    """1 - normalised edit distance, on normalised text."""
    ref = normalise(reference)
    hyp = normalise(hypothesis)
    if not ref and not hyp:
        return 1.0
    longest = max(len(ref), len(hyp))
    if longest == 0:
        return 1.0
    return max(0.0, 1.0 - edit_distance(ref, hyp) / longest)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = edit distance over tokens / reference length. Not capped at 1."""
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    previous = list(range(len(hyp_tokens) + 1))
    for i, ref_token in enumerate(ref_tokens, start=1):
        current = [i]
        for j, hyp_token in enumerate(hyp_tokens, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_token != hyp_token),
                )
            )
        previous = current
    return previous[-1] / len(ref_tokens)
