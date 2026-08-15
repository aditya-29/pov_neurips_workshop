"""ASL duration bucketing and sampling."""

from __future__ import annotations

import pytest

from pov.config import Config, ConfigError
from pov.experiments.asl.generate import AslGenerator, _safe_id, _sniff_delimiter
from pov.experiments.asl.sampling import (
    Bucket,
    SamplingError,
    allocate,
    assign_bucket,
    bucket_items,
    build_buckets,
    sample_buckets,
)

EDGES = [0, 3, 5, 10]


# ── Buckets ───────────────────────────────────────────────────────────────────


class TestBuildBuckets:
    def test_edges_become_half_open_intervals(self):
        buckets = build_buckets(EDGES)
        assert [(b.low, b.high) for b in buckets] == [(0, 3), (3, 5), (5, 10)]

    def test_default_labels(self):
        assert [b.label for b in build_buckets(EDGES)] == ["<3s", "<5s", "<10s"]

    def test_minute_labels(self):
        assert build_buckets([0, 60, 120])[1].label == "<2min"

    def test_custom_labels(self):
        buckets = build_buckets(EDGES, ["short", "mid", "long"])
        assert [b.label for b in buckets] == ["short", "mid", "long"]

    def test_rejects_too_few_edges(self):
        with pytest.raises(SamplingError, match="at least 2"):
            build_buckets([5])

    def test_rejects_non_increasing_edges(self):
        with pytest.raises(SamplingError, match="strictly increase"):
            build_buckets([0, 5, 5])

    def test_rejects_decreasing_edges(self):
        with pytest.raises(SamplingError, match="strictly increase"):
            build_buckets([0, 10, 5])

    def test_rejects_label_count_mismatch(self):
        with pytest.raises(SamplingError, match="expected 3 bucket labels"):
            build_buckets(EDGES, ["a", "b"])

    def test_rejects_duplicate_labels(self):
        with pytest.raises(SamplingError, match="unique"):
            build_buckets(EDGES, ["x", "x", "y"])


class TestAssignBucket:
    def setup_method(self):
        self.buckets = build_buckets(EDGES)

    @pytest.mark.parametrize(
        "seconds,label",
        [(0, "<3s"), (2.9, "<3s"), (3, "<5s"), (4.99, "<5s"), (5, "<10s"), (9.99, "<10s")],
    )
    def test_boundaries_are_half_open(self, seconds, label):
        assert assign_bucket(seconds, self.buckets).label == label

    @pytest.mark.parametrize("seconds", [-1, 10, 100])
    def test_outside_range_is_none(self, seconds):
        assert assign_bucket(seconds, self.buckets) is None

    def test_bucket_items_drops_out_of_range(self):
        items = [{"id": "a", "d": 1}, {"id": "b", "d": 99}]
        grouped = bucket_items(items, self.buckets, key=lambda i: i["d"])
        assert len(grouped["<3s"]) == 1
        assert sum(len(v) for v in grouped.values()) == 1


class TestAllocate:
    @pytest.mark.parametrize(
        "total,n,expected",
        [
            (100, 4, [25, 25, 25, 25]),
            (10, 3, [4, 3, 3]),
            (2, 5, [1, 1, 0, 0, 0]),
            (0, 3, [0, 0, 0]),
            (7, 7, [1] * 7),
        ],
    )
    def test_spreads_remainder_to_earliest_buckets(self, total, n, expected):
        assert allocate(total, n) == expected

    def test_sum_equals_total(self):
        assert sum(allocate(200, 7)) == 200

    def test_rejects_bad_inputs(self):
        with pytest.raises(SamplingError):
            allocate(10, 0)
        with pytest.raises(SamplingError):
            allocate(-1, 3)


class TestSampleBuckets:
    def items(self, n: int = 30):
        return [{"id": f"v{i:03d}", "d": i % 10} for i in range(n)]

    def test_per_bucket_quota(self):
        sampled, counts = sample_buckets(
            self.items(), build_buckets(EDGES), key=lambda i: i["d"],
            per_bucket=2, seed=1,
        )
        assert counts == {"<3s": 2, "<5s": 2, "<10s": 2}
        assert len(sampled) == 6

    def test_total_is_split_across_buckets(self):
        _, counts = sample_buckets(
            self.items(), build_buckets(EDGES), key=lambda i: i["d"],
            total=7, seed=1,
        )
        assert sum(counts.values()) == 7
        assert counts["<3s"] == 3  # remainder goes to the earliest bucket

    def test_small_bucket_is_not_topped_up(self):
        # Only one item lands in [0,3); the quota of 5 must not steal from elsewhere.
        items = [{"id": "a", "d": 1}] + [{"id": f"b{i}", "d": 7} for i in range(10)]
        _, counts = sample_buckets(
            items, build_buckets(EDGES), key=lambda i: i["d"], per_bucket=5, seed=1
        )
        assert counts["<3s"] == 1 and counts["<10s"] == 5

    def test_same_seed_same_sample(self):
        args = dict(buckets=build_buckets(EDGES), key=lambda i: i["d"], per_bucket=3)
        a, _ = sample_buckets(self.items(), seed=42, **args)
        b, _ = sample_buckets(self.items(), seed=42, **args)
        assert [i["id"] for i in a] == [i["id"] for i in b]

    def test_different_seed_different_sample(self):
        args = dict(buckets=build_buckets(EDGES), key=lambda i: i["d"], per_bucket=3)
        a, _ = sample_buckets(self.items(60), seed=1, **args)
        b, _ = sample_buckets(self.items(60), seed=2, **args)
        assert [i["id"] for i in a] != [i["id"] for i in b]

    def test_input_order_does_not_change_the_sample(self):
        # Filesystem ordering must not leak into which clips get chosen.
        items = self.items(40)
        args = dict(buckets=build_buckets(EDGES), key=lambda i: i["d"],
                    per_bucket=3, seed=5)
        a, _ = sample_buckets(items, **args)
        b, _ = sample_buckets(list(reversed(items)), **args)
        assert [i["id"] for i in a] == [i["id"] for i in b]

    def test_requires_exactly_one_quota(self):
        buckets = build_buckets(EDGES)
        with pytest.raises(SamplingError, match="exactly one"):
            sample_buckets(self.items(), buckets, key=lambda i: i["d"])
        with pytest.raises(SamplingError, match="exactly one"):
            sample_buckets(self.items(), buckets, key=lambda i: i["d"],
                           per_bucket=1, total=5)

    def test_empty_input(self):
        sampled, counts = sample_buckets(
            [], build_buckets(EDGES), key=lambda i: i["d"], per_bucket=2
        )
        assert sampled == [] and set(counts.values()) == {0}


# ── Generator helpers and params ──────────────────────────────────────────────


class TestHelpers:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("a_b-c.mp4", "a_b-c.mp4"),
            ("has space", "has_space"),
            ("we!rd/na*me", "we_rd_na_me"),
            ("...", "clip"),
            ("", "clip"),
        ],
    )
    def test_safe_id(self, name, expected):
        assert _safe_id(name) == expected

    def test_sniff_tab(self):
        assert _sniff_delimiter("a\tb\tc\n1\t2\t3\n") == "\t"

    def test_sniff_comma(self):
        assert _sniff_delimiter("a,b,c\n1,2,3\n") == ","


class TestAslParams:
    def build(self, params: dict):
        base = {
            "metadata_csv": "meta.csv",
            "video_dir": "videos",
            "buckets": {"edges": EDGES},
            "samples_per_bucket": 2,
        }
        base.update(params)
        return AslGenerator(Config.from_mapping({"experiment": "asl", "params": base}))

    def test_defaults(self):
        params = self.build({}).params
        assert params.id_column == "SENTENCE_NAME"
        assert params.text_column == "SENTENCE"
        assert len(params.buckets) == 3

    def test_requires_a_quota(self):
        with pytest.raises(ConfigError, match="exactly one"):
            self.build({"samples_per_bucket": None})

    def test_rejects_both_quotas(self):
        with pytest.raises(ConfigError, match="exactly one"):
            self.build({"total_samples": 10})

    def test_requires_edges(self):
        with pytest.raises(ConfigError, match="edges"):
            self.build({"buckets": {}})

    def test_rejects_bad_edges(self):
        with pytest.raises(ConfigError, match="strictly increase"):
            self.build({"buckets": {"edges": [5, 1]}})

    def test_rejects_bad_delimiter(self):
        with pytest.raises(ConfigError, match="single character"):
            self.build({"delimiter": "||"})

    def test_rejects_extension_without_dot(self):
        with pytest.raises(ConfigError, match="start with a dot"):
            self.build({"video_extension": "mp4"})

    def test_rejects_unknown_param(self):
        with pytest.raises(ConfigError, match="unknown key"):
            self.build({"sample_per_bucket": 3})

    def test_auto_cache_lives_under_output_root(self, tmp_path):
        generator = AslGenerator(Config.from_mapping({
            "experiment": "asl",
            "run": {"output_root": str(tmp_path)},
            "params": {
                "metadata_csv": "m.csv", "video_dir": "v",
                "buckets": {"edges": EDGES}, "samples_per_bucket": 1,
            },
        }))
        cache = generator._cache_path()
        assert cache == tmp_path / "asl" / ".pov_durations.json"

    def test_cache_can_be_disabled(self, tmp_path):
        generator = self.build({"duration_cache": "none"})
        assert generator._cache_path() is None

    def test_explicit_cache_path(self, tmp_path):
        target = tmp_path / "durations.json"
        generator = self.build({"duration_cache": str(target)})
        assert generator._cache_path() == target
