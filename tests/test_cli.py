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
        assert main(["eval", "-i", str(tmp_path / "nope.csv")]) == 1
        assert "not found" in capsys.readouterr().err.lower()


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
