"""Duration-bucketed sampling for the ASL corpus.

Pure logic, no I/O — the bucketing rules from the original notebook made
testable. Buckets are half-open ``[low, high)`` intervals, matching
``pandas.cut(..., right=False)``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


class SamplingError(ValueError):
    """Raised for malformed bucket definitions."""


@dataclass(frozen=True)
class Bucket:
    """A half-open duration interval ``[low, high)``."""

    label: str
    low: float
    high: float

    def contains(self, seconds: float) -> bool:
        return self.low <= seconds < self.high


def build_buckets(edges: Sequence[float], labels: Sequence[str] | None = None) -> list[Bucket]:
    """Turn ``[0, 3, 5]`` into buckets ``[0,3)`` and ``[3,5)``.

    `edges` must be strictly increasing; `labels` (if given) must have exactly
    one fewer entry than `edges`.
    """
    if len(edges) < 2:
        raise SamplingError(f"need at least 2 bucket edges, got {len(edges)}")
    for low, high in zip(edges, edges[1:]):
        if high <= low:
            raise SamplingError(
                f"bucket edges must strictly increase, got {low} then {high}"
            )

    if labels is None:
        labels = [_default_label(high) for high in edges[1:]]
    elif len(labels) != len(edges) - 1:
        raise SamplingError(
            f"expected {len(edges) - 1} bucket labels for {len(edges)} edges, got {len(labels)}"
        )

    if len(set(labels)) != len(labels):
        raise SamplingError(f"bucket labels must be unique, got {list(labels)}")

    return [
        Bucket(label=label, low=float(low), high=float(high))
        for label, low, high in zip(labels, edges, edges[1:])
    ]


def _default_label(high: float) -> str:
    if high >= 60 and high % 60 == 0:
        return f"<{int(high // 60)}min"
    return f"<{int(high)}s" if float(high).is_integer() else f"<{high}s"


def assign_bucket(seconds: float, buckets: Sequence[Bucket]) -> Bucket | None:
    """The bucket containing `seconds`, or None if it falls outside them all."""
    for bucket in buckets:
        if bucket.contains(seconds):
            return bucket
    return None


def bucket_items(
    items: Iterable[Any], buckets: Sequence[Bucket], key
) -> dict[str, list]:
    """Group items by bucket label. Items outside every bucket are dropped."""
    grouped: dict[str, list] = {bucket.label: [] for bucket in buckets}
    for item in items:
        bucket = assign_bucket(key(item), buckets)
        if bucket is not None:
            grouped[bucket.label].append(item)
    return grouped


def allocate(total: int, n_buckets: int) -> list[int]:
    """Split `total` samples across `n_buckets`, spreading the remainder.

    Matches the original notebook: base allocation, then one extra to each of
    the first `remainder` buckets.
    """
    if n_buckets <= 0:
        raise SamplingError(f"n_buckets must be > 0, got {n_buckets}")
    if total < 0:
        raise SamplingError(f"total must be >= 0, got {total}")
    base, remainder = divmod(total, n_buckets)
    return [base + (1 if i < remainder else 0) for i in range(n_buckets)]


def sample_buckets(
    items: Sequence[Any],
    buckets: Sequence[Bucket],
    key,
    *,
    per_bucket: int | None = None,
    total: int | None = None,
    seed: int = 1234,
) -> tuple[list, dict[str, int]]:
    """Sample items evenly across duration buckets.

    Exactly one of `per_bucket` or `total` must be given. A bucket with fewer
    items than its quota contributes everything it has (it is not topped up
    from other buckets — that would skew the duration distribution the
    experiment is built to vary).

    Returns ``(sampled_items, {bucket_label: count})``.
    """
    if (per_bucket is None) == (total is None):
        raise SamplingError("specify exactly one of per_bucket or total")

    grouped = bucket_items(items, buckets, key)
    quotas = (
        [per_bucket] * len(buckets)
        if per_bucket is not None
        else allocate(int(total), len(buckets))
    )

    rng = random.Random(seed)
    sampled: list = []
    counts: dict[str, int] = {}
    for bucket, quota in zip(buckets, quotas):
        pool = grouped[bucket.label]
        take = min(int(quota), len(pool))
        # Sort first so the sample depends only on the seed, never on the
        # order the filesystem happened to hand us the rows.
        ordered = sorted(pool, key=lambda item: str(_identity(item)))
        chosen = rng.sample(ordered, take) if take else []
        chosen.sort(key=lambda item: str(_identity(item)))
        sampled.extend(chosen)
        counts[bucket.label] = len(chosen)
    return sampled, counts


def _identity(item: Any) -> Any:
    """Stable sort key: mappings sort by their id-ish field, else by repr."""
    if isinstance(item, dict):
        for field in ("sample_id", "id", "SENTENCE_NAME", "name"):
            if field in item:
                return item[field]
        return sorted(item.items())
    return item
