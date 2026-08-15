"""Command-line interface: parsing, exit codes, and error reporting."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pov.cli import main
from tests.conftest import write_yaml


class TestExperiments:
    def test_lists_all_three(self, capsys):
        assert main(["experiments"]) == 0
        listed = capsys.readouterr().out.split()
        assert listed == ["asl", "chess", "wbw_mcq"]


class TestValidate:
    def test_accepts_a_good_config(self, tmp_path, chess_config, capsys):
        path = write_yaml(tmp_path / "c.yaml", chess_config)
        assert main(["validate", "-c", str(path)]) == 0
        assert "OK" in capsys.readouterr().out

    def test_rejects_unknown_key(self, tmp_path, chess_config, capsys):
        chess_config["params"]["nonsense"] = 1
        path = write_yaml(tmp_path / "c.yaml", chess_config)
        assert main(["validate", "-c", str(path)]) == 2
        assert "unknown key" in capsys.readouterr().err

    def test_rejects_unknown_experiment(self, tmp_path, capsys):
        path = write_yaml(tmp_path / "c.yaml", {"experiment": "mystery"})
        assert main(["validate", "-c", str(path)]) == 2
        assert "unknown experiment" in capsys.readouterr().err

    def test_missing_file(self, tmp_path, capsys):
        assert main(["validate", "-c", str(tmp_path / "nope.yaml")]) == 2
        assert "not found" in capsys.readouterr().err

    def test_applies_overrides(self, tmp_path, chess_config, capsys):
        path = write_yaml(tmp_path / "c.yaml", chess_config)
        assert main(["validate", "-c", str(path), "--set", "run.workers=3"]) == 0

    def test_bad_override_syntax(self, tmp_path, chess_config, capsys):
        path = write_yaml(tmp_path / "c.yaml", chess_config)
        assert main(["validate", "-c", str(path), "--set", "workers"]) == 2
        assert "expected the form" in capsys.readouterr().err


class TestGenerateDryRun:
    def test_prints_plan_without_writing(self, tmp_path, chess_config, capsys, out_root):
        path = write_yaml(tmp_path / "c.yaml", chess_config)
        assert main(["generate", "-c", str(path), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "run dir" in out and "no media written" in out
        assert not (out_root / "chess" / "test" / "media").exists()

    def test_dry_run_reports_config_hash(self, tmp_path, chess_config, capsys):
        path = write_yaml(tmp_path / "c.yaml", chess_config)
        main(["generate", "-c", str(path), "--dry-run"])
        assert "config hash" in capsys.readouterr().out

    def test_override_changes_the_plan(self, tmp_path, chess_config, capsys):
        path = write_yaml(tmp_path / "c.yaml", chess_config)
        main(["generate", "-c", str(path), "--dry-run",
              "--set", "run.run_id=custom"])
        assert "custom" in capsys.readouterr().out

    def test_dry_run_fails_when_source_data_is_missing(self, tmp_path, capsys):
        # A dry run that reports a valid plan for a command which cannot
        # possibly execute is worse than useless.
        config = {
            "experiment": "wbw_mcq",
            "run": {"output_root": str(tmp_path / "out")},
            "params": {"questions_path": str(tmp_path / "absent.jsonl")},
        }
        path = write_yaml(tmp_path / "c.yaml", config)
        assert main(["generate", "-c", str(path), "--dry-run"]) == 2
        captured = capsys.readouterr()
        assert "source data is NOT ready" in captured.err
        assert "fetch_mmlu.py" in captured.err
        assert "config is valid, source data is present" not in captured.out

    def test_dry_run_passes_when_source_data_is_present(self, tmp_path, questions_file, capsys):
        config = {
            "experiment": "wbw_mcq",
            "run": {"output_root": str(tmp_path / "out")},
            "params": {"questions_path": str(questions_file)},
        }
        path = write_yaml(tmp_path / "c.yaml", config)
        assert main(["generate", "-c", str(path), "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "questions  : 2" in out
        assert "source data is present" in out

    def test_dry_run_needs_no_source_data_for_chess(self, tmp_path, chess_config):
        path = write_yaml(tmp_path / "c.yaml", chess_config)
        assert main(["generate", "-c", str(path), "--dry-run"]) == 0


class TestEvalCommand:
    def preds(self, tmp_path: Path) -> Path:
        rows = [
            {"sample_id": "q1", "experiment": "wbw_mcq", "condition": "static_image",
             "ground_truth": "B", "ground_truth_path": "", "model_output": "ANSWER: B"},
            {"sample_id": "q2", "experiment": "wbw_mcq", "condition": "static_image",
             "ground_truth": "C", "ground_truth_path": "", "model_output": "ANSWER: A"},
        ]
        path = tmp_path / "preds.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_scores_and_writes_outputs(self, tmp_path, capsys):
        path = self.preds(tmp_path)
        assert main(["eval", "-i", str(path)]) == 0
        out = capsys.readouterr().out
        assert "Scored 2 row(s)" in out
        assert (tmp_path / "scored.csv").exists()
        assert (tmp_path / "summary.csv").exists()

    def test_output_dir_option(self, tmp_path):
        path = self.preds(tmp_path)
        assert main(["eval", "-i", str(path), "-o", str(tmp_path / "res")]) == 0
        assert (tmp_path / "res" / "scored.csv").exists()

    def test_group_by_option(self, tmp_path, capsys):
        path = self.preds(tmp_path)
        assert main(["eval", "-i", str(path), "--group-by", "experiment"]) == 0
        assert "experiment" in capsys.readouterr().out

    def test_report_flag_writes_html(self, tmp_path):
        path = self.preds(tmp_path)
        html_path = tmp_path / "r.html"
        assert main(["eval", "-i", str(path), "--report", str(html_path)]) == 0
        assert html_path.exists()

    def test_missing_input(self, tmp_path, capsys):
        # A missing input file is bad input, not a pov bug: exit 2, and the
        # message must not leak the exception class name.
        assert main(["eval", "-i", str(tmp_path / "nope.csv")]) == 2
        err = capsys.readouterr().err
        assert "not found" in err.lower()
        assert "ManifestError" not in err
        assert "internal error" not in err

    def test_unreadable_questions_reports_cleanly(self, tmp_path, capsys):
        # QuestionError used to surface as a raw class name via the catch-all.
        bad = tmp_path / "q.jsonl"
        bad.write_text("not json\n")
        config = {
            "experiment": "wbw_mcq",
            "run": {"output_root": str(tmp_path / "out")},
            "params": {"questions_path": str(bad)},
        }
        path = write_yaml(tmp_path / "c.yaml", config)
        assert main(["generate", "-c", str(path)]) == 2
        err = capsys.readouterr().err
        assert "QuestionError" not in err
        assert err.startswith("error:")


class TestFailedRunCleanup:
    """A run that fails before producing anything must leave no directory."""

    def config(self, tmp_path: Path, questions: str) -> dict:
        # These tests are about run-directory lifecycle, not rendering, so keep
        # the media tiny — the defaults would encode 14 full-size videos,
        # including a 60-frames-per-word one, for no added coverage.
        return {
            "experiment": "wbw_mcq",
            "run": {"output_root": str(tmp_path / "data"), "run_id": "r1"},
            "video": {"fps": 10, "preset": "ultrafast", "crf": 40},
            "params": {
                "questions_path": questions,
                "static_image": True,
                "modes": [],
                "canvas": {"width": 160, "height": 120, "static_font_size": 8,
                           "static_padding": 8, "video_padding": 8},
            },
        }

    def test_missing_source_data_leaves_no_run_directory(self, tmp_path):
        # Regression: the run dir, media/, ground_truth/ and config.resolved.yaml
        # were all created before generation raised, leaving an empty run behind
        # that looked exactly like a real one.
        path = write_yaml(
            tmp_path / "c.yaml", self.config(tmp_path, str(tmp_path / "absent.jsonl"))
        )
        assert main(["generate", "-c", str(path)]) == 2
        assert not (tmp_path / "data" / "wbw_mcq" / "r1").exists()
        assert not (tmp_path / "data" / "wbw_mcq").exists()

    def test_unreadable_source_leaves_no_run_directory(self, tmp_path):
        bad = tmp_path / "q.jsonl"
        bad.write_text("{not json\n")
        path = write_yaml(tmp_path / "c.yaml", self.config(tmp_path, str(bad)))
        assert main(["generate", "-c", str(path)]) == 2
        assert not (tmp_path / "data" / "wbw_mcq" / "r1").exists()

    def test_a_successful_run_keeps_its_directory(self, tmp_path, questions_file):
        path = write_yaml(tmp_path / "c.yaml", self.config(tmp_path, str(questions_file)))
        assert main(["generate", "-c", str(path)]) == 0
        assert (tmp_path / "data" / "wbw_mcq" / "r1" / "manifest.csv").exists()

    def test_existing_output_is_never_discarded(self, tmp_path):
        # A pre-existing run directory belongs to an earlier run; a later
        # failure must not delete it.
        run_dir = tmp_path / "data" / "wbw_mcq" / "r1"
        (run_dir / "media").mkdir(parents=True)
        (run_dir / "media" / "keep.mp4").write_bytes(b"earlier run")
        path = write_yaml(
            tmp_path / "c.yaml", self.config(tmp_path, str(tmp_path / "absent.jsonl"))
        )
        assert main(["generate", "-c", str(path)]) == 2
        assert (run_dir / "media" / "keep.mp4").exists()


class TestReportCommand:
    def test_builds_from_scored_csv(self, tmp_path):
        rows = [{"sample_id": "s1", "experiment": "wbw_mcq", "condition": "static_image",
                 "media_path": "", "ground_truth": "B", "ground_truth_path": "",
                 "model_output": "ANSWER: B", "score_correct": "1.0"}]
        path = tmp_path / "scored.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        assert main(["report", "-i", str(path)]) == 0
        assert (tmp_path / "scored.html").exists()

    def test_explicit_output_and_title(self, tmp_path):
        rows = [{"sample_id": "s1", "experiment": "asl", "condition": "video",
                 "media_path": "", "ground_truth": "hi", "ground_truth_path": "",
                 "model_output": "hi", "score_token_f1": "1.0"}]
        path = tmp_path / "scored.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        target = tmp_path / "custom.html"
        assert main(["report", "-i", str(path), "-o", str(target),
                     "--title", "My Title"]) == 0
        assert "<title>My Title</title>" in target.read_text()


class TestParser:
    def test_requires_a_command(self):
        with pytest.raises(SystemExit):
            main([])

    def test_unknown_command(self):
        with pytest.raises(SystemExit):
            main(["frobnicate"])

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert "pov" in capsys.readouterr().out

    def test_generate_requires_config(self):
        with pytest.raises(SystemExit):
            main(["generate"])

    def test_eval_requires_input(self):
        with pytest.raises(SystemExit):
            main(["eval"])
