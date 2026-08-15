"""Run layout and manifest CSV."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pov.layout import RunLayout, LayoutError
from pov.manifest import (
    CORE_FIELDS,
    MODEL_OUTPUT_COLUMN,
    ManifestError,
    ManifestRow,
    ManifestWriter,
    load_ground_truth,
    read_manifest,
    store_ground_truth,
)


@pytest.fixture
def layout(tmp_path: Path) -> RunLayout:
    return RunLayout.build(tmp_path / "data", "chess", "run1").create()


def make_row(**overrides) -> ManifestRow:
    fields = dict(
        sample_id="s1",
        experiment="chess",
        condition="video_5s",
        media_type="video",
        media_path="media/s1.mp4",
    )
    fields.update(overrides)
    return ManifestRow(**fields)


# ── Layout ────────────────────────────────────────────────────────────────────


class TestRunLayout:
    def test_directory_structure(self, layout):
        assert layout.run_dir.name == "run1"
        assert layout.run_dir.parent.name == "chess"
        assert layout.media_dir.is_dir()
        assert layout.manifest_path.name == "manifest.csv"
        assert layout.config_path.name == "config.resolved.yaml"

    def test_ground_truth_dir_is_not_created_up_front(self, layout):
        # Experiments whose ground truth fits in a CSV cell (an MCQ answer
        # letter, an ASL sentence) must not grow an empty directory.
        assert not layout.ground_truth_dir.exists()

    def test_ground_truth_dir_appears_when_first_used(self, layout):
        path = layout.ground_truth_file("game0000.txt")
        assert layout.ground_truth_dir.is_dir()
        path.write_text("transcript", encoding="utf-8")
        assert path.read_text() == "transcript"

    def test_create_is_idempotent(self, layout):
        marker = layout.media_dir / "keep.txt"
        marker.write_text("x")
        layout.create()
        assert marker.exists()

    def test_overwrite_clears_the_run(self, layout):
        marker = layout.media_dir / "old.txt"
        marker.write_text("x")
        layout.create(overwrite=True)
        assert not marker.exists()
        assert layout.media_dir.is_dir()

    @pytest.mark.parametrize("bad", ["a/b", "..", ".", "a\\b", ""])
    def test_rejects_bad_run_ids(self, tmp_path, bad):
        with pytest.raises(LayoutError):
            RunLayout.build(tmp_path, "chess", bad)

    def test_rejects_empty_experiment(self, tmp_path):
        with pytest.raises(LayoutError, match="experiment"):
            RunLayout.build(tmp_path, "", "run1")

    def test_relpath_is_posix_and_relative(self, layout):
        media = layout.media_file("clip.mp4")
        assert layout.relpath(media) == "media/clip.mp4"

    def test_relpath_rejects_outside_paths(self, layout, tmp_path):
        with pytest.raises(LayoutError, match="outside the run directory"):
            layout.relpath(tmp_path / "elsewhere.mp4")

    def test_resolve_inverts_relpath(self, layout):
        media = layout.media_file("clip.mp4")
        assert layout.resolve(layout.relpath(media)) == media

    @pytest.mark.parametrize("bad", ["a/b.mp4", "..", "", "a\\b.mp4"])
    def test_media_file_rejects_path_separators(self, layout, bad):
        with pytest.raises(LayoutError):
            layout.media_file(bad)

    def test_run_dir_that_is_a_file(self, tmp_path):
        target = tmp_path / "data" / "chess" / "run1"
        target.parent.mkdir(parents=True)
        target.write_text("not a directory")
        with pytest.raises(LayoutError, match="not a directory"):
            RunLayout.build(tmp_path / "data", "chess", "run1").create()


# ── ManifestRow ───────────────────────────────────────────────────────────────


class TestManifestRow:
    def test_media_filename_derived_from_path(self):
        assert make_row(media_path="media/game_1.mp4").media_filename == "game_1.mp4"

    def test_rejects_empty_sample_id(self):
        with pytest.raises(ManifestError, match="sample_id"):
            make_row(sample_id="")

    def test_rejects_empty_media_path(self):
        with pytest.raises(ManifestError, match="media_path"):
            make_row(media_path="")

    def test_rejects_unknown_media_type(self):
        with pytest.raises(ManifestError, match="media_type"):
            make_row(media_type="audio")

    def test_rejects_extra_column_colliding_with_core(self):
        with pytest.raises(ManifestError, match="collide"):
            make_row(extra={"fps": 10})

    def test_rejects_extra_column_named_model_output(self):
        with pytest.raises(ManifestError, match="collide"):
            make_row(extra={MODEL_OUTPUT_COLUMN: "x"})

    def test_extra_columns_appear_in_dict(self):
        row = make_row(extra={"n_half_moves": 12})
        assert row.to_dict()["n_half_moves"] == 12


# ── ManifestWriter ────────────────────────────────────────────────────────────


class TestManifestWriter:
    def test_writes_core_columns_in_order(self, tmp_path):
        writer = ManifestWriter(tmp_path / "manifest.csv")
        writer.add(make_row())
        path = writer.write()
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        assert header[: len(CORE_FIELDS)] == list(CORE_FIELDS)
        assert header[-1] == MODEL_OUTPUT_COLUMN

    def test_model_output_column_is_present_and_empty(self, tmp_path):
        writer = ManifestWriter(tmp_path / "m.csv")
        writer.add(make_row())
        rows = read_manifest(writer.write())
        assert rows[0][MODEL_OUTPUT_COLUMN] == ""

    def test_config_columns_are_repeated_on_every_row(self, tmp_path):
        writer = ManifestWriter(tmp_path / "m.csv", config_columns={"cfg_experiment": "chess"})
        writer.add(make_row(sample_id="a"))
        writer.add(make_row(sample_id="b"))
        rows = read_manifest(writer.write())
        assert [row["cfg_experiment"] for row in rows] == ["chess", "chess"]

    def test_config_columns_must_be_prefixed(self, tmp_path):
        with pytest.raises(ManifestError, match="must start with 'cfg_'"):
            ManifestWriter(tmp_path / "m.csv", config_columns={"experiment": "chess"})

    def test_duplicate_sample_id_rejected(self, tmp_path):
        writer = ManifestWriter(tmp_path / "m.csv")
        writer.add(make_row())
        with pytest.raises(ManifestError, match="duplicate sample_id"):
            writer.add(make_row())

    def test_union_of_extra_columns_across_rows(self, tmp_path):
        writer = ManifestWriter(tmp_path / "m.csv")
        writer.add(make_row(sample_id="a", extra={"alpha": 1}))
        writer.add(make_row(sample_id="b", extra={"beta": 2}))
        rows = read_manifest(writer.write())
        assert rows[0]["beta"] == ""      # absent for row a
        assert rows[1]["alpha"] == ""     # absent for row b
        assert rows[0]["alpha"] == "1"

    def test_none_becomes_empty_not_the_string_none(self, tmp_path):
        writer = ManifestWriter(tmp_path / "m.csv")
        writer.add(make_row(fps=None, duration_sec=None))
        rows = read_manifest(writer.write())
        assert rows[0]["fps"] == "" and rows[0]["duration_sec"] == ""

    def test_bools_render_lowercase(self, tmp_path):
        writer = ManifestWriter(tmp_path / "m.csv")
        writer.add(make_row(extra={"flag": True, "other": False}))
        rows = read_manifest(writer.write())
        assert rows[0]["flag"] == "true" and rows[0]["other"] == "false"

    def test_sort_key_orders_rows(self, tmp_path):
        writer = ManifestWriter(tmp_path / "m.csv")
        writer.add(make_row(sample_id="z"))
        writer.add(make_row(sample_id="a"))
        rows = read_manifest(writer.write(sort_key=lambda r: r.sample_id))
        assert [row["sample_id"] for row in rows] == ["a", "z"]

    def test_embedded_commas_newlines_and_quotes_survive(self, tmp_path):
        nasty = 'He said "hi", then\nleft; a,b,c'
        writer = ManifestWriter(tmp_path / "m.csv")
        writer.add(make_row(ground_truth=nasty))
        rows = read_manifest(writer.write())
        assert rows[0]["ground_truth"] == nasty

    def test_unicode_survives(self, tmp_path):
        writer = ManifestWriter(tmp_path / "m.csv")
        writer.add(make_row(ground_truth="♔ e4 — naïve café 日本語"))
        rows = read_manifest(writer.write())
        assert rows[0]["ground_truth"] == "♔ e4 — naïve café 日本語"

    def test_very_long_field_is_readable(self, tmp_path):
        # csv defaults to a 128 KiB field cap; long transcripts exceed it.
        long_text = "1 White a2 a3; " * 20000
        writer = ManifestWriter(tmp_path / "m.csv")
        writer.add(make_row(ground_truth=long_text))
        rows = read_manifest(writer.write())
        assert rows[0]["ground_truth"] == long_text

    def test_write_is_atomic_leaving_no_temp_file(self, tmp_path):
        writer = ManifestWriter(tmp_path / "m.csv")
        writer.add(make_row())
        writer.write()
        assert list(tmp_path.glob("*.tmp")) == []

    def test_len_and_rows(self, tmp_path):
        writer = ManifestWriter(tmp_path / "m.csv")
        writer.add(make_row())
        assert len(writer) == 1 and writer.rows[0].sample_id == "s1"


# ── read_manifest ─────────────────────────────────────────────────────────────


class TestReadManifest:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ManifestError, match="not found"):
            read_manifest(tmp_path / "nope.csv")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("")
        with pytest.raises(ManifestError, match="file is empty"):
            read_manifest(path)

    def test_rejects_non_manifest_csv(self, tmp_path):
        path = tmp_path / "other.csv"
        path.write_text("a,b\n1,2\n")
        with pytest.raises(ManifestError, match="missing column"):
            read_manifest(path)


# ── Ground truth storage ──────────────────────────────────────────────────────


class TestGroundTruth:
    def test_short_text_stays_inline(self, layout):
        inline, rel = store_ground_truth(
            "B", filename="s1.txt",
            ground_truth_dir=layout.ground_truth_dir, run_dir=layout.run_dir,
        )
        assert inline == "B" and rel == ""

    def test_long_text_moves_to_a_file(self, layout):
        text = "x" * 5000
        inline, rel = store_ground_truth(
            text, filename="s1.txt",
            ground_truth_dir=layout.ground_truth_dir, run_dir=layout.run_dir,
        )
        assert rel == "ground_truth/s1.txt"
        assert "truncated" in inline
        assert (layout.run_dir / rel).read_text() == text

    def test_load_prefers_the_file(self, layout):
        text = "y" * 5000
        inline, rel = store_ground_truth(
            text, filename="s1.txt",
            ground_truth_dir=layout.ground_truth_dir, run_dir=layout.run_dir,
        )
        row = {"sample_id": "s1", "ground_truth": inline, "ground_truth_path": rel}
        assert load_ground_truth(row, layout.run_dir) == text

    def test_load_falls_back_to_inline(self, layout):
        row = {"sample_id": "s1", "ground_truth": "B", "ground_truth_path": ""}
        assert load_ground_truth(row, layout.run_dir) == "B"

    def test_load_errors_on_dangling_path(self, layout):
        row = {"sample_id": "s1", "ground_truth": "", "ground_truth_path": "ground_truth/gone.txt"}
        with pytest.raises(ManifestError, match="does not exist"):
            load_ground_truth(row, layout.run_dir)

    def test_boundary_length_stays_inline(self, layout):
        from pov.manifest import MAX_INLINE_GROUND_TRUTH

        text = "z" * MAX_INLINE_GROUND_TRUTH
        inline, rel = store_ground_truth(
            text, filename="s.txt",
            ground_truth_dir=layout.ground_truth_dir, run_dir=layout.run_dir,
        )
        assert rel == "" and inline == text
