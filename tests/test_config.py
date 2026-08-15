"""Config loading, validation, and resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pov.config import (
    Config,
    ConfigError,
    Reader,
    RunConfig,
    VideoConfig,
    deep_merge,
    load_yaml,
    parse_override,
)


# ── Reader ────────────────────────────────────────────────────────────────────


class TestReader:
    def test_reads_typed_values(self):
        r = Reader({"name": "x", "count": 3, "ratio": 0.5, "flag": True}, path="p")
        assert r.str("name") == "x"
        assert r.int("count") == 3
        assert r.float("ratio") == 0.5
        assert r.bool("flag") is True
        r.done()

    def test_missing_required_key_names_the_path(self):
        r = Reader({}, path="params")
        with pytest.raises(ConfigError, match="params.missing: required key is missing"):
            r.str("missing")

    def test_default_used_when_absent(self):
        r = Reader({}, path="p")
        assert r.int("count", 5) == 5
        assert r.optional_str("name") is None

    def test_explicit_null_falls_back_to_default(self):
        # `fps: ~` in YAML should mean "use the default", not crash.
        r = Reader({"count": None}, path="p")
        assert r.int("count", 7) == 7

    def test_unknown_key_is_an_error(self):
        r = Reader({"fps": 30, "fsp": 24}, path="video")
        r.int("fps", 30)
        with pytest.raises(ConfigError, match=r"unknown key\(s\) \['fsp'\]"):
            r.done()

    def test_wrong_type_reports_actual_type(self):
        r = Reader({"count": "twelve"}, path="p")
        with pytest.raises(ConfigError, match="expected an integer, got str"):
            r.int("count")

    def test_bool_is_not_accepted_as_int(self):
        # `fps: true` must not silently become 1.
        r = Reader({"fps": True}, path="video")
        with pytest.raises(ConfigError, match="expected an integer, got bool"):
            r.int("fps")

    def test_bool_is_not_accepted_as_float(self):
        r = Reader({"ratio": False}, path="p")
        with pytest.raises(ConfigError, match="expected a number, got bool"):
            r.float("ratio")

    @pytest.mark.parametrize("value,low,high", [(0, 1, 10), (11, 1, 10)])
    def test_range_is_enforced(self, value, low, high):
        r = Reader({"n": value}, path="p")
        with pytest.raises(ConfigError, match="must be"):
            r.int("n", min=low, max=high)

    def test_choices_are_enforced(self):
        r = Reader({"mode": "sideways"}, path="p")
        with pytest.raises(ConfigError, match="is not one of"):
            r.str("mode", choices=("static", "animated"))

    def test_list_validates_item_type_and_length(self):
        r = Reader({"edges": [1, "two"]}, path="p")
        with pytest.raises(ConfigError, match=r"p.edges\[1\]"):
            r.list("edges", item_type=(int, float))

        r2 = Reader({"edges": [1]}, path="p")
        with pytest.raises(ConfigError, match="needs at least 2"):
            r2.list("edges", item_type=int, min_len=2)

    def test_string_is_not_a_list(self):
        r = Reader({"modes": "vanishing"}, path="p")
        with pytest.raises(ConfigError, match="expected a list, got str"):
            r.list("modes")

    def test_non_mapping_input_rejected(self):
        with pytest.raises(ConfigError, match="expected a mapping, got list"):
            Reader([1, 2], path="params")

    def test_empty_path_is_rejected(self):
        r = Reader({"p": ""}, path="x")
        with pytest.raises(ConfigError, match="must not be empty"):
            r.path("p")

    def test_keys_and_contains(self):
        r = Reader({"a": 1, "b": 2}, path="p")
        assert sorted(r.keys()) == ["a", "b"]
        assert "a" in r and len(r) == 2


# ── Sections ──────────────────────────────────────────────────────────────────


class TestVideoConfig:
    def test_defaults(self):
        cfg = VideoConfig.from_reader(Reader({}, path="video"))
        assert (cfg.fps, cfg.crf, cfg.codec) == (30, 18, "libx264")

    def test_rejects_unknown_preset(self):
        with pytest.raises(ConfigError, match="is not one of"):
            VideoConfig.from_reader(Reader({"preset": "turbo"}, path="video"))

    def test_tune_may_be_null(self):
        cfg = VideoConfig.from_reader(Reader({"tune": None}, path="video"))
        assert cfg.tune is None

    def test_rejects_unknown_tune(self):
        with pytest.raises(ConfigError, match="video.tune"):
            VideoConfig.from_reader(Reader({"tune": "sharp"}, path="video"))

    def test_crf_range(self):
        with pytest.raises(ConfigError):
            VideoConfig.from_reader(Reader({"crf": 99}, path="video"))


class TestRunConfig:
    def test_defaults(self):
        cfg = RunConfig.from_reader(Reader({}, path="run"))
        assert cfg.output_root == Path("data")
        assert cfg.seed == 1234 and cfg.resume is True

    @pytest.mark.parametrize("bad", ["../escape", "with/slash", ".hidden", "-leading"])
    def test_rejects_unsafe_run_ids(self, bad):
        with pytest.raises(ConfigError, match="not a valid directory name"):
            RunConfig.from_reader(Reader({"run_id": bad}, path="run"))

    def test_accepts_reasonable_run_ids(self):
        cfg = RunConfig.from_reader(Reader({"run_id": "run_1.2-final"}, path="run"))
        assert cfg.run_id == "run_1.2-final"

    def test_overwrite_and_resume_conflict(self):
        with pytest.raises(ConfigError, match="cannot both be true"):
            RunConfig.from_reader(
                Reader({"overwrite": True, "resume": True}, path="run")
            )


# ── Config ────────────────────────────────────────────────────────────────────


class TestConfig:
    def test_minimal_config(self):
        cfg = Config.from_mapping({"experiment": "chess"})
        assert cfg.experiment == "chess"
        assert cfg.params == {}

    def test_missing_experiment(self):
        with pytest.raises(ConfigError, match="experiment: required"):
            Config.from_mapping({"run": {}})

    def test_unknown_top_level_key(self):
        with pytest.raises(ConfigError, match=r"unknown key\(s\) \['prams'\]"):
            Config.from_mapping({"experiment": "chess", "prams": {}})

    def test_params_must_be_mapping(self):
        with pytest.raises(ConfigError, match="params: expected a mapping"):
            Config.from_mapping({"experiment": "chess", "params": [1, 2]})

    def test_params_null_becomes_empty(self):
        cfg = Config.from_mapping({"experiment": "chess", "params": None})
        assert cfg.params == {}

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(ConfigError, match="config file not found"):
            Config.load(tmp_path / "nope.yaml")

    def test_load_applies_overrides(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text(yaml.safe_dump({"experiment": "chess", "run": {"seed": 1}}))
        cfg = Config.load(path, overrides={"run": {"seed": 99}})
        assert cfg.run.seed == 99

    def test_hash_is_stable_and_content_sensitive(self):
        base = {"experiment": "chess", "params": {"num_games": 2}}
        a = Config.from_mapping(base)
        b = Config.from_mapping({"experiment": "chess", "params": {"num_games": 2}})
        c = Config.from_mapping({"experiment": "chess", "params": {"num_games": 3}})
        assert a.config_hash() == b.config_hash()
        assert a.config_hash() != c.config_hash()

    def test_hash_ignores_non_content_run_fields(self):
        a = Config.from_mapping({"experiment": "chess", "run": {"workers": 1}})
        b = Config.from_mapping({"experiment": "chess", "run": {"workers": 16}})
        assert a.config_hash() == b.config_hash()

    def test_hash_tracks_seed(self):
        a = Config.from_mapping({"experiment": "chess", "run": {"seed": 1}})
        b = Config.from_mapping({"experiment": "chess", "run": {"seed": 2}})
        assert a.config_hash() != b.config_hash()

    def test_run_id_from_config_wins(self):
        cfg = Config.from_mapping({"experiment": "chess", "run": {"run_id": "fixed"}})
        assert cfg.resolve_run_id() == "fixed"

    def test_generated_run_id_has_timestamp_and_hash(self):
        import datetime

        cfg = Config.from_mapping({"experiment": "chess"})
        run_id = cfg.resolve_run_id(now=datetime.datetime(2026, 8, 14, 20, 30, 5))
        assert run_id.startswith("20260814-203005-")
        assert run_id.endswith(cfg.config_hash())

    def test_flatten_produces_cfg_columns(self):
        cfg = Config.from_mapping({"experiment": "chess", "params": {"num_games": 4}})
        flat = cfg.flatten()
        assert flat["cfg_experiment"] == "chess"
        assert flat["cfg_params.num_games"] == 4
        assert all(key.startswith("cfg_") for key in flat)

    def test_flatten_serialises_lists_as_json(self):
        cfg = Config.from_mapping({"experiment": "asl", "params": {"edges": [1, 2]}})
        assert cfg.flatten()["cfg_params.edges"] == "[1, 2]"

    def test_snapshot_round_trips(self, tmp_path):
        cfg = Config.from_mapping({"experiment": "chess", "params": {"num_games": 3}})
        target = tmp_path / "snap" / "config.resolved.yaml"
        cfg.write_snapshot(target)
        loaded = yaml.safe_load(target.read_text())
        assert loaded["experiment"] == "chess"
        assert loaded["params"]["num_games"] == 3
        assert loaded["_meta"]["config_hash"] == cfg.config_hash()


# ── YAML helpers ──────────────────────────────────────────────────────────────


class TestYamlHelpers:
    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "empty.yaml"
        path.write_text("")
        with pytest.raises(ConfigError, match="file is empty"):
            load_yaml(path)

    def test_non_mapping_top_level_rejected(self, tmp_path):
        path = tmp_path / "list.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ConfigError, match="top level must be a mapping"):
            load_yaml(path)

    def test_invalid_yaml_reports_path(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("a: [1, 2\n")
        with pytest.raises(ConfigError, match="invalid YAML"):
            load_yaml(path)

    def test_deep_merge_merges_mappings(self):
        merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
        assert merged == {"a": {"x": 1, "y": 3}}

    def test_deep_merge_replaces_lists_wholesale(self):
        merged = deep_merge({"a": [1, 2, 3]}, {"a": [9]})
        assert merged == {"a": [9]}

    def test_deep_merge_does_not_mutate_inputs(self):
        base = {"a": {"x": 1}}
        deep_merge(base, {"a": {"x": 2}})
        assert base == {"a": {"x": 1}}

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("run.workers=4", {"run": {"workers": 4}}),
            ("video.tune=null", {"video": {"tune": None}}),
            ("run.resume=false", {"run": {"resume": False}}),
            ("params.speeds.fast=5.5", {"params": {"speeds": {"fast": 5.5}}}),
            ("experiment=chess", {"experiment": "chess"}),
        ],
    )
    def test_parse_override(self, text, expected):
        assert parse_override(text) == expected

    def test_parse_override_requires_equals(self):
        with pytest.raises(ConfigError, match="expected the form"):
            parse_override("run.workers")

    def test_parse_override_rejects_empty_key(self):
        with pytest.raises(ConfigError, match="empty key"):
            parse_override("=4")
