"""Evaluation driver: dispatch, scoring, aggregation, and output files."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pov.eval import EvalError, evaluate, get_scorer, summarise
from pov.eval.base import SCORE_PREFIX
from pov.manifest import read_manifest


def write_csv(path: Path, rows: list[dict]) -> Path:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return path


def mcq_rows(**overrides) -> list[dict]:
    base = [
        {"sample_id": "q1_static", "experiment": "wbw_mcq", "condition": "static_image",
         "ground_truth": "B", "ground_truth_path": "", "model_output": "ANSWER: B"},
        {"sample_id": "q1_van", "experiment": "wbw_mcq", "condition": "vanishing_slow",
         "ground_truth": "B", "ground_truth_path": "", "model_output": "ANSWER: A"},
        {"sample_id": "q2_static", "experiment": "wbw_mcq", "condition": "static_image",
         "ground_truth": "C", "ground_truth_path": "", "model_output": "ANSWER: C"},
    ]
    for row in base:
        row.update(overrides)
    return base


class TestGetScorer:
    @pytest.mark.parametrize("name", ["chess", "asl", "wbw_mcq"])
    def test_known_experiments(self, name):
        assert get_scorer(name).experiment == name

    def test_unknown_experiment(self):
        with pytest.raises(EvalError, match="no scorer for experiment"):
            get_scorer("nope")


class TestEvaluate:
    def test_scores_are_added_with_prefix(self, tmp_path):
        report = evaluate(write_csv(tmp_path / "p.csv", mcq_rows()), write=False)
        assert f"{SCORE_PREFIX}correct" in report.rows[0]
        assert report.rows[0][f"{SCORE_PREFIX}correct"] == 1.0

    def test_original_columns_are_preserved(self, tmp_path):
        report = evaluate(write_csv(tmp_path / "p.csv", mcq_rows()), write=False)
        assert report.rows[0]["sample_id"] == "q1_static"
        assert report.rows[0]["model_output"] == "ANSWER: B"

    def test_overall_means(self, tmp_path):
        report = evaluate(write_csv(tmp_path / "p.csv", mcq_rows()), write=False)
        assert report.overall()[f"{SCORE_PREFIX}correct"] == pytest.approx(2 / 3)

    def test_summary_groups_by_condition(self, tmp_path):
        report = evaluate(write_csv(tmp_path / "p.csv", mcq_rows()), write=False)
        by_condition = {row["condition"]: row for row in report.summary}
        assert by_condition["static_image"][f"{SCORE_PREFIX}correct"] == 1.0
        assert by_condition["vanishing_slow"][f"{SCORE_PREFIX}correct"] == 0.0
        assert by_condition["static_image"]["n"] == 2

    def test_model_column_joins_the_grouping(self, tmp_path):
        rows = mcq_rows()
        for index, row in enumerate(rows):
            row["model"] = "m1" if index < 2 else "m2"
        report = evaluate(write_csv(tmp_path / "p.csv", rows), write=False)
        assert "model" in report.group_columns
        assert {row["model"] for row in report.summary} == {"m1", "m2"}

    def test_explicit_group_by(self, tmp_path):
        report = evaluate(
            write_csv(tmp_path / "p.csv", mcq_rows()), write=False, group_by=["experiment"]
        )
        assert report.group_columns == ["experiment"]
        assert len(report.summary) == 1

    def test_unknown_group_by_column(self, tmp_path):
        with pytest.raises(EvalError, match="group-by column"):
            evaluate(write_csv(tmp_path / "p.csv", mcq_rows()), write=False,
                     group_by=["nope"])

    def test_missing_model_output_column(self, tmp_path):
        rows = mcq_rows()
        for row in rows:
            row.pop("model_output")
        with pytest.raises(EvalError, match="no 'model_output' column"):
            evaluate(write_csv(tmp_path / "p.csv", rows), write=False)

    def test_empty_model_output_scores_zero_and_is_counted(self, tmp_path):
        rows = mcq_rows()
        rows[0]["model_output"] = ""
        report = evaluate(write_csv(tmp_path / "p.csv", rows), write=False)
        assert report.n_missing_output == 1
        assert report.rows[0][f"{SCORE_PREFIX}correct"] == 0.0

    def test_empty_output_does_not_score_as_a_perfect_transcription(self, tmp_path):
        # Regression: empty predictions were zeroed across the board, so `wer`
        # (an error rate) read 0.0 — flattering a model that answered nothing
        # and dragging the reported mean down.
        rows = [
            {"sample_id": "a1", "experiment": "asl", "condition": "video_<3s",
             "ground_truth": "hello there friend", "ground_truth_path": "",
             "model_output": ""},
        ]
        report = evaluate(write_csv(tmp_path / "p.csv", rows), write=False)
        assert report.rows[0][f"{SCORE_PREFIX}wer"] == 1.0
        assert report.rows[0][f"{SCORE_PREFIX}token_f1"] == 0.0
        assert report.n_missing_output == 1

    def test_empty_output_keeps_ground_truth_counts(self, tmp_path):
        # Regression: moves_expected is a property of the transcript; zeroing it
        # corrupted the mean printed in summary.csv.
        rows = [
            {"sample_id": "g1", "experiment": "chess", "condition": "video_5s",
             "ground_truth": "1 White b2 b3; 1 Black c7 c5", "ground_truth_path": "",
             "n_half_moves": "2", "model_output": ""},
        ]
        report = evaluate(write_csv(tmp_path / "p.csv", rows), write=False)
        assert report.rows[0][f"{SCORE_PREFIX}moves_expected"] == 2
        assert report.rows[0][f"{SCORE_PREFIX}moves_predicted"] == 0
        assert report.rows[0][f"{SCORE_PREFIX}loose"] == 0.0

    def test_missing_experiment_value(self, tmp_path):
        rows = mcq_rows()
        rows[1]["experiment"] = ""
        with pytest.raises(EvalError, match="no 'experiment' value"):
            evaluate(write_csv(tmp_path / "p.csv", rows), write=False)

    def test_unknown_experiment_value(self, tmp_path):
        rows = mcq_rows()
        for row in rows:
            row["experiment"] = "mystery"
        with pytest.raises(EvalError, match="no scorer"):
            evaluate(write_csv(tmp_path / "p.csv", rows), write=False)

    def test_mixed_experiments_in_one_file(self, tmp_path):
        rows = mcq_rows() + [
            {"sample_id": "a1", "experiment": "asl", "condition": "video_<3s",
             "ground_truth": "hello there", "ground_truth_path": "",
             "model_output": "hello there"}
        ]
        report = evaluate(write_csv(tmp_path / "p.csv", rows), write=False)
        assert sorted(report.experiments) == ["asl", "wbw_mcq"]
        # A metric only one experiment produces must be blank elsewhere, not 0.
        asl_row = [r for r in report.rows if r["experiment"] == "asl"][0]
        mcq_row = [r for r in report.rows if r["experiment"] == "wbw_mcq"][0]
        assert asl_row[f"{SCORE_PREFIX}token_f1"] == 1.0
        assert mcq_row[f"{SCORE_PREFIX}token_f1"] == ""

    def test_reads_ground_truth_from_file(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "ground_truth").mkdir(parents=True)
        (run_dir / "ground_truth" / "g.txt").write_text(
            "1 White b2 b3; 1 Black c7 c5", encoding="utf-8"
        )
        rows = [{
            "sample_id": "g1", "experiment": "chess", "condition": "video_5s",
            "ground_truth": "truncated…", "ground_truth_path": "ground_truth/g.txt",
            "n_half_moves": "2",
            "model_output": "Move 1: White Pawn b2b3\nMove 1: Black Pawn c7c5",
        }]
        report = evaluate(write_csv(run_dir / "p.csv", rows), write=False)
        assert report.rows[0][f"{SCORE_PREFIX}strict"] == 1.0

    def test_dangling_ground_truth_path_errors(self, tmp_path):
        rows = [{
            "sample_id": "g1", "experiment": "chess", "condition": "video_5s",
            "ground_truth": "", "ground_truth_path": "ground_truth/missing.txt",
            "model_output": "x",
        }]
        with pytest.raises(EvalError, match="does not exist"):
            evaluate(write_csv(tmp_path / "p.csv", rows), write=False)

    def test_run_dir_override(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir / "ground_truth").mkdir(parents=True)
        (run_dir / "ground_truth" / "g.txt").write_text("1 White b2 b3", encoding="utf-8")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        rows = [{
            "sample_id": "g1", "experiment": "chess", "condition": "video_5s",
            "ground_truth": "", "ground_truth_path": "ground_truth/g.txt",
            "model_output": "Move 1: White Pawn b2b3",
        }]
        report = evaluate(
            write_csv(elsewhere / "p.csv", rows), write=False, run_dir=run_dir
        )
        assert report.rows[0][f"{SCORE_PREFIX}strict"] == 1.0

    def test_empty_file_rejected(self, tmp_path):
        path = tmp_path / "p.csv"
        path.write_text("sample_id,experiment,model_output\n")
        with pytest.raises(EvalError, match="no rows"):
            evaluate(path, write=False)


class TestOutputFiles:
    def test_writes_scored_and_summary(self, tmp_path):
        report = evaluate(write_csv(tmp_path / "p.csv", mcq_rows()), tmp_path / "out")
        assert report.scored_path.name == "scored.csv"
        assert report.summary_path.name == "summary.csv"
        assert report.scored_path.exists() and report.summary_path.exists()

    def test_defaults_next_to_the_input(self, tmp_path):
        report = evaluate(write_csv(tmp_path / "p.csv", mcq_rows()))
        assert report.scored_path.parent == tmp_path

    def test_scored_csv_round_trips(self, tmp_path):
        report = evaluate(write_csv(tmp_path / "p.csv", mcq_rows()), tmp_path / "out")
        rows = read_manifest(report.scored_path)
        assert len(rows) == 3
        assert rows[0][f"{SCORE_PREFIX}correct"] == "1.0"

    def test_summary_csv_has_group_and_metric_columns(self, tmp_path):
        report = evaluate(write_csv(tmp_path / "p.csv", mcq_rows()), tmp_path / "out")
        with open(report.summary_path, newline="") as f:
            header = next(csv.reader(f))
        assert "condition" in header and "n" in header
        assert f"{SCORE_PREFIX}correct" in header

    def test_no_temp_files_left(self, tmp_path):
        evaluate(write_csv(tmp_path / "p.csv", mcq_rows()), tmp_path / "out")
        assert list((tmp_path / "out").glob("*.tmp")) == []

    def test_text_summary_mentions_counts(self, tmp_path):
        report = evaluate(write_csv(tmp_path / "p.csv", mcq_rows()), write=False)
        text = report.text_summary()
        assert "Scored 3 row(s)" in text and "wbw_mcq" in text


class TestSummarise:
    def test_empty_input(self):
        assert summarise([], ["condition"], ["score_x"]) == []

    def test_means_and_counts(self):
        rows = [
            {"condition": "a", "score_x": 1.0},
            {"condition": "a", "score_x": 0.0},
            {"condition": "b", "score_x": 0.5},
        ]
        summary = {r["condition"]: r for r in summarise(rows, ["condition"], ["score_x"])}
        assert summary["a"]["score_x"] == 0.5 and summary["a"]["n"] == 2
        assert summary["b"]["score_x"] == 0.5

    def test_blank_values_are_excluded_from_the_mean(self):
        rows = [{"condition": "a", "score_x": 1.0}, {"condition": "a", "score_x": ""}]
        summary = summarise(rows, ["condition"], ["score_x"])
        assert summary[0]["score_x"] == 1.0 and summary[0]["n"] == 2

    def test_all_blank_metric_is_blank(self):
        rows = [{"condition": "a", "score_x": ""}]
        assert summarise(rows, ["condition"], ["score_x"])[0]["score_x"] == ""

    def test_groups_are_sorted(self):
        rows = [{"condition": "z", "score_x": 1}, {"condition": "a", "score_x": 1}]
        assert [r["condition"] for r in summarise(rows, ["condition"], ["score_x"])] == ["a", "z"]
