"""Single-file HTML report."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pov.eval.base import SCORE_PREFIX
from pov.report import build_report, build_report_from_csv
from tests.test_eval_runner import write_csv


def extract_payload(html: str) -> dict:
    match = re.search(
        r'<script type="application/json" id="pov-data">(.*?)</script>', html, re.S
    )
    assert match, "embedded JSON payload not found"
    return json.loads(match.group(1))


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    (run / "media").mkdir(parents=True)
    (run / "media" / "s1.mp4").write_bytes(b"fake")
    (run / "media" / "s2.jpg").write_bytes(b"fake")
    return run


@pytest.fixture
def rows() -> list[dict]:
    return [
        {"sample_id": "s1", "experiment": "wbw_mcq", "condition": "vanishing_slow",
         "media_path": "media/s1.mp4", "ground_truth": "B",
         "model_output": "ANSWER: B", f"{SCORE_PREFIX}correct": 1.0, "word_count": 12},
        {"sample_id": "s2", "experiment": "wbw_mcq", "condition": "static_image",
         "media_path": "media/s2.jpg", "ground_truth": "C",
         "model_output": "ANSWER: A", f"{SCORE_PREFIX}correct": 0.0, "word_count": 9},
    ]


class TestBuildReport:
    def test_writes_a_single_html_file(self, tmp_path, run_dir, rows):
        path = build_report(rows, tmp_path / "r.html", run_dir=run_dir,
                            metrics=[f"{SCORE_PREFIX}correct"])
        assert path.exists()
        assert path.read_text().startswith("<!doctype html>")

    def test_creates_parent_directories(self, tmp_path, run_dir, rows):
        path = build_report(rows, tmp_path / "deep" / "nested" / "r.html", run_dir=run_dir)
        assert path.exists()

    def test_payload_contains_every_sample(self, tmp_path, run_dir, rows):
        path = build_report(rows, tmp_path / "r.html", run_dir=run_dir,
                            metrics=[f"{SCORE_PREFIX}correct"])
        payload = extract_payload(path.read_text())
        assert len(payload["samples"]) == 2
        assert payload["samples"][0]["id"] == "s1"

    def test_media_paths_are_relative_and_resolve(self, tmp_path, run_dir, rows):
        path = build_report(rows, tmp_path / "r.html", run_dir=run_dir)
        payload = extract_payload(path.read_text())
        for sample in payload["samples"]:
            assert not sample["mediaSrc"].startswith("/")
            assert (path.parent / sample["mediaSrc"]).exists()

    def test_media_kind_is_detected(self, tmp_path, run_dir, rows):
        path = build_report(rows, tmp_path / "r.html", run_dir=run_dir)
        kinds = {s["id"]: s["mediaKind"] for s in extract_payload(path.read_text())["samples"]}
        assert kinds == {"s1": "video", "s2": "image"}

    def test_report_beside_media_uses_short_paths(self, run_dir, rows):
        path = build_report(rows, run_dir / "r.html", run_dir=run_dir)
        payload = extract_payload(path.read_text())
        assert payload["samples"][0]["mediaSrc"] == "media/s1.mp4"

    def test_no_media_path_is_handled(self, tmp_path, run_dir):
        rows = [{"sample_id": "s1", "experiment": "asl", "condition": "video",
                 "media_path": "", "ground_truth": "x", "model_output": "y"}]
        path = build_report(rows, tmp_path / "r.html", run_dir=run_dir)
        assert extract_payload(path.read_text())["samples"][0]["mediaSrc"] == ""

    def test_scores_are_exposed_without_prefix(self, tmp_path, run_dir, rows):
        path = build_report(rows, tmp_path / "r.html", run_dir=run_dir,
                            metrics=[f"{SCORE_PREFIX}correct"])
        payload = extract_payload(path.read_text())
        assert payload["samples"][0]["scores"] == {"correct": 1.0}

    def test_summary_table_is_rendered(self, tmp_path, run_dir, rows):
        summary = [{"condition": "static_image", "n": 1, f"{SCORE_PREFIX}correct": 0.0},
                   {"condition": "vanishing_slow", "n": 1, f"{SCORE_PREFIX}correct": 1.0}]
        path = build_report(rows, tmp_path / "r.html", run_dir=run_dir, summary=summary,
                            metrics=[f"{SCORE_PREFIX}correct"], group_columns=["condition"])
        html = path.read_text()
        assert "vanishing_slow" in html and "1.0000" in html

    def test_empty_summary_renders_placeholder(self, tmp_path, run_dir, rows):
        html = build_report(rows, tmp_path / "r.html", run_dir=run_dir).read_text()
        assert "No summary rows" in html

    def test_max_samples_caps_the_listing(self, tmp_path, run_dir, rows):
        path = build_report(rows, tmp_path / "r.html", run_dir=run_dir, max_samples=1)
        assert len(extract_payload(path.read_text())["samples"]) == 1

    def test_title_is_used(self, tmp_path, run_dir, rows):
        html = build_report(rows, tmp_path / "r.html", run_dir=run_dir,
                            title="My Results").read_text()
        assert "<title>My Results</title>" in html

    def test_html_in_content_is_escaped(self, tmp_path, run_dir):
        rows = [{"sample_id": "s1", "experiment": "asl", "condition": "video",
                 "media_path": "", "ground_truth": "<script>alert(1)</script>",
                 "model_output": "</script><img onerror=x>"}]
        html = build_report(rows, tmp_path / "r.html", run_dir=run_dir).read_text()
        # The payload is JSON-encoded, so a raw closing tag must not appear inside it.
        payload = extract_payload(html)
        assert payload["samples"][0]["groundTruth"] == "<script>alert(1)</script>"

    def test_closing_script_tag_cannot_break_out(self, tmp_path, run_dir):
        # Regression: a model answering with "</script>" would otherwise end the
        # JSON block early and inject the rest of the string as live markup.
        payload_text = '</script><img src=x onerror=alert(1)>'
        rows = [{"sample_id": "s1", "experiment": "asl", "condition": "video",
                 "media_path": "", "ground_truth": payload_text,
                 "model_output": payload_text}]
        html_text = build_report(rows, tmp_path / "r.html", run_dir=run_dir).read_text()

        block = re.search(
            r'<script type="application/json" id="pov-data">(.*?)</script>',
            html_text, re.S,
        ).group(1)
        assert "</script>" not in block
        assert "<img" not in block
        # …and the original text is still recovered intact by JSON.parse.
        assert json.loads(block)["samples"][0]["modelOutput"] == payload_text

    def test_title_is_escaped(self, tmp_path, run_dir, rows):
        html = build_report(rows, tmp_path / "r.html", run_dir=run_dir,
                            title="A & B <tag>").read_text()
        assert "A &amp; B &lt;tag&gt;" in html

    def test_is_self_contained(self, tmp_path, run_dir, rows):
        html = build_report(rows, tmp_path / "r.html", run_dir=run_dir).read_text()
        assert "<style>" in html and "<script>" in html
        assert "http://" not in html and "https://" not in html
        assert "cdn" not in html.lower()

    def test_unicode_survives(self, tmp_path, run_dir):
        rows = [{"sample_id": "s1", "experiment": "asl", "condition": "video",
                 "media_path": "", "ground_truth": "café ♔ 日本語", "model_output": "x"}]
        path = build_report(rows, tmp_path / "r.html", run_dir=run_dir)
        assert extract_payload(path.read_text())["samples"][0]["groundTruth"] == "café ♔ 日本語"

    def test_config_columns_are_not_shown_in_details(self, tmp_path, run_dir):
        rows = [{"sample_id": "s1", "experiment": "asl", "condition": "video",
                 "media_path": "", "ground_truth": "x", "model_output": "y",
                 "cfg_params.num_games": "20", "bucket": "<3s"}]
        path = build_report(rows, tmp_path / "r.html", run_dir=run_dir)
        details = extract_payload(path.read_text())["samples"][0]["details"]
        assert "bucket" in details
        assert not any(key.startswith("cfg_") for key in details)


class TestBuildReportFromCsv:
    def test_end_to_end(self, tmp_path, run_dir):
        rows = [
            {"sample_id": "s1", "experiment": "wbw_mcq", "condition": "static_image",
             "media_path": "media/s2.jpg", "ground_truth": "B", "ground_truth_path": "",
             "model_output": "ANSWER: B", f"{SCORE_PREFIX}correct": 1.0},
            {"sample_id": "s2", "experiment": "wbw_mcq", "condition": "vanishing_fast",
             "media_path": "media/s1.mp4", "ground_truth": "C", "ground_truth_path": "",
             "model_output": "ANSWER: A", f"{SCORE_PREFIX}correct": 0.0},
        ]
        csv_path = write_csv(run_dir / "scored.csv", rows)
        path = build_report_from_csv(csv_path, tmp_path / "r.html")
        payload = extract_payload(path.read_text())
        assert len(payload["samples"]) == 2
        assert payload["metricLabels"] == ["correct"]

    def test_derives_summary(self, tmp_path, run_dir):
        rows = [
            {"sample_id": "s1", "experiment": "wbw_mcq", "condition": "a",
             "media_path": "", "ground_truth": "B", "ground_truth_path": "",
             "model_output": "ANSWER: B", f"{SCORE_PREFIX}correct": 1.0},
            {"sample_id": "s2", "experiment": "wbw_mcq", "condition": "a",
             "media_path": "", "ground_truth": "C", "ground_truth_path": "",
             "model_output": "ANSWER: A", f"{SCORE_PREFIX}correct": 0.0},
        ]
        path = build_report_from_csv(write_csv(run_dir / "scored.csv", rows),
                                     tmp_path / "r.html")
        assert len(extract_payload(path.read_text())["summary"]) == 1

    def test_empty_csv_rejected(self, tmp_path):
        path = tmp_path / "scored.csv"
        path.write_text("sample_id,experiment\n")
        with pytest.raises(ValueError, match="file is empty"):
            build_report_from_csv(path, tmp_path / "r.html")
