"""End-to-end generation, then evaluation, then reporting.

These tests write real media, so they are marked `integration` and skipped when
ffmpeg is unavailable. Run just these with:

    pytest -m integration
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from pov.config import Config
from pov.eval import evaluate
from pov.eval.base import SCORE_PREFIX
from pov.eval.chess import parse_transcript
from pov.manifest import MODEL_OUTPUT_COLUMN, load_ground_truth, read_manifest
from pov.registry import build_generator
from pov.report import build_report_from_csv
from pov.video import EncodeSettings, probe_video, write_timeline
from tests.conftest import requires_ffmpeg

pytestmark = [requires_ffmpeg, pytest.mark.integration]


def run(config: dict):
    return build_generator(Config.from_mapping(config)).run()


def fill_predictions(manifest_path: Path, make_output) -> Path:
    """Copy a manifest, filling model_output with `make_output(row, run_dir)`."""
    rows = read_manifest(manifest_path)
    run_dir = manifest_path.parent
    for row in rows:
        row[MODEL_OUTPUT_COLUMN] = make_output(row, run_dir)
    target = manifest_path.parent / "preds.csv"
    with open(target, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target


# ── Chess ─────────────────────────────────────────────────────────────────────


class TestChessGeneration:
    def test_produces_expected_artifacts(self, chess_config, out_root):
        result = run(chess_config)
        assert result.ok
        assert result.n_rows == 2                     # 2 games x 1 duration
        assert result.manifest_path.exists()
        assert (result.run_dir / "config.resolved.yaml").exists()

        rows = read_manifest(result.manifest_path)
        for row in rows:
            media = result.run_dir / row["media_path"]
            assert media.exists() and media.stat().st_size > 0
            assert (result.run_dir / row["ground_truth_path"]).exists()

    def test_manifest_records_real_media_properties(self, chess_config):
        result = run(chess_config)
        for row in read_manifest(result.manifest_path):
            probed = probe_video(result.run_dir / row["media_path"])
            assert int(row["n_frames"]) == probed.n_frames
            assert int(row["width"]) == probed.width
            assert int(row["height"]) == probed.height
            assert float(row["duration_sec"]) == pytest.approx(probed.duration_sec, abs=0.05)

    def test_manifest_carries_config_and_provenance(self, chess_config):
        result = run(chess_config)
        row = read_manifest(result.manifest_path)[0]
        assert row["cfg_params.num_games"] == "2"
        assert row["cfg_video.fps"] == "30"
        assert row["config_hash"] and row["pov_version"] and row["generated_at"]
        assert row["seed"] and row["run_id"] == "test"
        assert row[MODEL_OUTPUT_COLUMN] == ""

    def test_ground_truth_matches_the_clip_length(self, chess_config):
        result = run(chess_config)
        for row in read_manifest(result.manifest_path):
            transcript = load_ground_truth(row, result.run_dir)
            assert len(parse_transcript(transcript)) == int(row["n_half_moves"])

    def test_frames_are_rendered_once_per_position(self, chess_config):
        # Three durations off one game must not triple the render count.
        chess_config["params"]["durations"] = [
            {"label": "2s", "seconds": 2},
            {"label": "3s", "seconds": 3},
            {"label": "4s", "seconds": 4},
        ]
        chess_config["params"]["num_games"] = 1
        result = run(chess_config)
        assert result.n_rows == 3
        longest = max(int(r["n_half_moves"]) for r in read_manifest(result.manifest_path))
        # One intro frame plus one frame per position of the longest clip.
        assert result.stats["frames_rendered"] <= longest + 1

    def test_pipe_reduction_is_reported(self, chess_config):
        result = run(chess_config)
        assert result.stats["frames_piped"] < result.stats["frames_encoded"]

    def test_same_seed_reproduces_the_run(self, chess_config, tmp_path):
        first = read_manifest(run(chess_config).manifest_path)

        chess_config["run"]["output_root"] = str(tmp_path / "second")
        second = read_manifest(run(chess_config).manifest_path)

        assert [r["ground_truth"] for r in first] == [r["ground_truth"] for r in second]
        assert [r["n_frames"] for r in first] == [r["n_frames"] for r in second]

    def test_different_seed_changes_the_games(self, chess_config, tmp_path):
        first = read_manifest(run(chess_config).manifest_path)
        chess_config["run"]["seed"] = 999
        chess_config["run"]["output_root"] = str(tmp_path / "second")
        second = read_manifest(run(chess_config).manifest_path)
        assert [r["ground_truth"] for r in first] != [r["ground_truth"] for r in second]

    def test_resume_skips_existing_media(self, chess_config):
        run(chess_config)
        chess_config["run"]["resume"] = True
        second = run(chess_config)
        assert second.n_skipped == second.n_rows
        assert second.stats["frames_encoded"] == 0

    def test_resume_still_reports_real_properties(self, chess_config):
        first = read_manifest(run(chess_config).manifest_path)
        chess_config["run"]["resume"] = True
        second = read_manifest(run(chess_config).manifest_path)
        assert [r["n_frames"] for r in first] == [r["n_frames"] for r in second]

    def test_overwrite_clears_the_run_directory(self, chess_config):
        result = run(chess_config)
        stray = result.run_dir / "media" / "stray.mp4"
        stray.write_bytes(b"junk")

        chess_config["run"]["resume"] = False
        chess_config["run"]["overwrite"] = True
        run(chess_config)
        assert not stray.exists()

    def test_clip_duration_tracks_the_label(self, chess_config):
        chess_config["params"]["durations"] = [
            {"label": "2s", "seconds": 2},
            {"label": "6s", "seconds": 6},
        ]
        chess_config["params"]["num_games"] = 1
        chess_config["params"]["max_half_moves"] = 200
        rows = {r["duration_label"]: r for r in read_manifest(run(chess_config).manifest_path)}
        assert float(rows["2s"]["duration_sec"]) < float(rows["6s"]["duration_sec"])

    def test_animated_motion_renders_slides(self, chess_config):
        chess_config["params"]["motion"] = "animated"
        chess_config["params"]["timing"] = {
            "intro_frames": 4, "slide_frames": 3, "pause_frames": 3, "outro_frames": 4
        }
        result = run(chess_config)
        assert result.ok
        assert all(r["motion"] == "animated" for r in read_manifest(result.manifest_path))

    def test_config_snapshot_round_trips(self, chess_config):
        result = run(chess_config)
        snapshot = yaml.safe_load((result.run_dir / "config.resolved.yaml").read_text())
        assert snapshot["experiment"] == "chess"
        assert snapshot["params"]["num_games"] == 2
        assert snapshot["_meta"]["config_hash"]


class TestChessEndToEnd:
    def test_perfect_and_empty_predictions(self, chess_config):
        result = run(chess_config)

        def perfect(row, run_dir):
            moves = parse_transcript(load_ground_truth(row, run_dir))
            return "\n".join(
                f"Move {m['move_num']}: {m['color']} Pawn {m['from']}{m['to']}" for m in moves
            )

        report = evaluate(fill_predictions(result.manifest_path, perfect), write=False)
        assert all(row[f"{SCORE_PREFIX}strict"] == 1.0 for row in report.rows)

        report = evaluate(
            fill_predictions(result.manifest_path, lambda row, d: "no idea"), write=False
        )
        assert all(row[f"{SCORE_PREFIX}loose"] == 0.0 for row in report.rows)

    def test_report_links_resolve(self, chess_config, tmp_path):
        result = run(chess_config)
        preds = fill_predictions(result.manifest_path, lambda row, d: "Move 1: White Pawn a2a3")
        scored = evaluate(preds, result.run_dir).scored_path

        html_path = build_report_from_csv(scored, result.run_dir / "report.html")
        import json
        import re

        payload = json.loads(
            re.search(r'id="pov-data">(.*?)</script>', html_path.read_text(), re.S).group(1)
        )
        assert payload["samples"]
        for sample in payload["samples"]:
            assert (html_path.parent / sample["mediaSrc"]).exists()


# ── Word-by-word MCQ ──────────────────────────────────────────────────────────


class TestWbwGeneration:
    def test_produces_image_and_video_conditions(self, wbw_config):
        result = run(wbw_config)
        assert result.ok
        rows = read_manifest(result.manifest_path)
        # 2 questions x (1 static + 1 mode x 1 speed)
        assert len(rows) == 4
        kinds = {row["media_type"] for row in rows}
        assert kinds == {"image", "video"}
        for row in rows:
            assert (result.run_dir / row["media_path"]).exists()

    def test_all_conditions_present(self, wbw_config):
        wbw_config["params"]["modes"] = ["vanishing", "cumulative"]
        wbw_config["params"]["speeds"] = {"slow": 1.0, "fast": 5.0}
        rows = read_manifest(run(wbw_config).manifest_path)
        conditions = {row["condition"] for row in rows}
        assert conditions == {
            "static_image",
            "vanishing_slow", "vanishing_fast",
            "cumulative_slow", "cumulative_fast",
        }

    def test_frames_are_shared_across_speeds(self, wbw_config):
        wbw_config["params"]["modes"] = ["cumulative"]
        wbw_config["params"]["speeds"] = {"a": 1.0, "b": 2.0, "c": 5.0}
        result = run(wbw_config)
        rows = read_manifest(result.manifest_path)
        total_words = sum(
            int(row["word_count"]) for row in rows if row["condition"] == "static_image"
        )
        # One render per word per question, plus one static image per question —
        # not one per (word x speed).
        assert result.stats["frames_rendered"] <= total_words + 2

    def test_slower_speed_makes_a_longer_video(self, wbw_config):
        wbw_config["params"]["speeds"] = {"slow": 1.0, "fast": 5.0}
        rows = [r for r in read_manifest(run(wbw_config).manifest_path)
                if r["media_type"] == "video"]
        by_key = {(r["question_id"], r["speed"]): float(r["duration_sec"]) for r in rows}
        for qid in {r["question_id"] for r in rows}:
            assert by_key[(qid, "slow")] > by_key[(qid, "fast")]

    def test_frame_count_matches_words_and_hold(self, wbw_config):
        wbw_config["params"]["modes"] = ["cumulative"]
        for row in read_manifest(run(wbw_config).manifest_path):
            if row["condition"].startswith("cumulative"):
                expected = int(row["word_count"]) * int(row["frames_per_word"])
                assert int(row["n_frames"]) == expected

    def test_vanishing_adds_the_blank_gap(self, wbw_config):
        wbw_config["params"]["modes"] = ["vanishing"]
        wbw_config["params"]["blank_gap_frames"] = 2
        for row in read_manifest(run(wbw_config).manifest_path):
            if row["condition"].startswith("vanishing"):
                words = int(row["word_count"])
                expected = words * (int(row["frames_per_word"]) + 2)
                assert int(row["n_frames"]) == expected

    def test_ground_truth_is_the_answer_letter(self, wbw_config):
        for row in read_manifest(run(wbw_config).manifest_path):
            assert row["ground_truth"] in ("A", "B", "C", "D")

    def test_question_metadata_is_recorded(self, wbw_config):
        row = read_manifest(run(wbw_config).manifest_path)[0]
        assert row["stem"] and row["option_a"] and row["question_id"]
        assert int(row["word_count"]) > 0

    def test_limit_applies(self, wbw_config):
        wbw_config["params"]["limit"] = 1
        result = run(wbw_config)
        assert result.stats["questions"] == 1

    def test_static_only_writes_no_videos(self, wbw_config):
        wbw_config["params"]["modes"] = []
        rows = read_manifest(run(wbw_config).manifest_path)
        assert {row["media_type"] for row in rows} == {"image"}

    def test_resume_skips_existing(self, wbw_config):
        run(wbw_config)
        wbw_config["run"]["resume"] = True
        second = run(wbw_config)
        assert second.n_skipped == second.n_rows

    def test_end_to_end_scoring(self, wbw_config):
        result = run(wbw_config)
        preds = fill_predictions(
            result.manifest_path, lambda row, d: f"ANSWER: {row['ground_truth']}"
        )
        report = evaluate(preds, write=False)
        assert report.overall()[f"{SCORE_PREFIX}correct"] == 1.0


# ── ASL ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def asl_corpus(tmp_path: Path) -> dict:
    """A synthetic How2Sign-shaped corpus with real (tiny) videos."""
    video_dir = tmp_path / "raw_videos"
    video_dir.mkdir()
    settings = EncodeSettings(fps=10, preset="ultrafast", crf=35, tune=None)

    rows = []
    # Durations chosen to land one clip in each of three buckets.
    for index, seconds in enumerate([1.0, 4.0, 7.0, 1.5]):
        name = f"clip_{index}-rgb_front"
        frame = np.full((32, 48, 3), index * 40 % 255, dtype=np.uint8)
        write_timeline(video_dir / f"{name}.mp4",
                       [(frame, int(seconds * 10))], settings)
        rows.append({
            "VIDEO_ID": f"vid{index}",
            "SENTENCE_NAME": name,
            "SENTENCE": f"This is sentence number {index}.",
            "START": "0.0",
            "END": f"{seconds}",
        })
    # A metadata row whose video is absent — must be skipped, not fatal.
    rows.append({"VIDEO_ID": "gone", "SENTENCE_NAME": "missing-rgb_front",
                 "SENTENCE": "Not downloaded.", "START": "0.0", "END": "2.0"})

    metadata = tmp_path / "how2sign_val.csv"
    with open(metadata, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    return {
        "experiment": "asl",
        "run": {"output_root": str(tmp_path / "data"), "run_id": "test",
                "seed": 3, "workers": 2, "resume": False},
        "video": {"fps": 10, "preset": "ultrafast", "crf": 35, "tune": None},
        "params": {
            "metadata_csv": str(metadata),
            "video_dir": str(video_dir),
            "buckets": {"edges": [0, 3, 5, 10]},
            "samples_per_bucket": 2,
            "scale_height": 16,
        },
    }


class TestAslGeneration:
    def test_samples_across_buckets(self, asl_corpus):
        result = run(asl_corpus)
        assert result.ok
        counts = result.stats["bucket_counts"]
        assert counts == {"<3s": 2, "<5s": 1, "<10s": 1}
        assert result.n_rows == 4

    def test_missing_source_video_is_skipped_not_fatal(self, asl_corpus):
        result = run(asl_corpus)
        assert result.stats["videos_missing"] == 1
        assert result.ok

    def test_clips_are_normalised(self, asl_corpus):
        result = run(asl_corpus)
        for row in read_manifest(result.manifest_path):
            probed = probe_video(result.run_dir / row["media_path"])
            assert probed.height == 16
            assert probed.width % 2 == 0

    def test_ground_truth_is_the_sentence(self, asl_corpus):
        for row in read_manifest(run(asl_corpus).manifest_path):
            assert row["ground_truth"].startswith("This is sentence")

    def test_bucket_and_source_recorded(self, asl_corpus):
        for row in read_manifest(run(asl_corpus).manifest_path):
            assert row["bucket"] in ("<3s", "<5s", "<10s")
            assert row["source_name"] and row["source_duration_sec"]

    def test_duration_cache_is_reused(self, asl_corpus):
        first = run(asl_corpus)
        assert first.stats["videos_probed"] == 4

        asl_corpus["run"]["resume"] = True
        second = run(asl_corpus)
        assert second.stats["videos_probed"] == 0

    def test_cache_lives_under_the_output_root(self, asl_corpus):
        run(asl_corpus)
        cache = Path(asl_corpus["run"]["output_root"]) / "asl" / ".pov_durations.json"
        assert cache.exists()
        # And never inside the source video directory.
        assert not list(Path(asl_corpus["params"]["video_dir"]).glob(".pov_*"))

    def test_same_seed_samples_the_same_clips(self, asl_corpus, tmp_path):
        first = [r["sample_id"] for r in read_manifest(run(asl_corpus).manifest_path)]
        asl_corpus["run"]["output_root"] = str(tmp_path / "again")
        second = [r["sample_id"] for r in read_manifest(run(asl_corpus).manifest_path)]
        assert first == second

    def test_total_samples_mode(self, asl_corpus):
        asl_corpus["params"].pop("samples_per_bucket")
        asl_corpus["params"]["total_samples"] = 3
        result = run(asl_corpus)
        assert sum(result.stats["bucket_counts"].values()) == 3

    def test_time_range_cutting(self, asl_corpus):
        asl_corpus["params"]["use_time_range"] = True
        asl_corpus["params"]["max_clip_seconds"] = 1.0
        for row in read_manifest(run(asl_corpus).manifest_path):
            assert float(row["duration_sec"]) <= 1.3  # keyframe slack

    def test_end_to_end_scoring(self, asl_corpus):
        result = run(asl_corpus)
        preds = fill_predictions(result.manifest_path, lambda row, d: row["ground_truth"])
        report = evaluate(preds, write=False)
        assert report.overall()[f"{SCORE_PREFIX}exact_match"] == 1.0
        assert report.overall()[f"{SCORE_PREFIX}wer"] == 0.0
