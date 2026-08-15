"""Shared fixtures.

Tests that write real media are marked `integration` and skipped when ffmpeg is
not installed, so the rest of the suite runs anywhere in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pov.video import ffmpeg_available, ffprobe_available

# Skip marker for anything that shells out to ffmpeg/ffprobe.
requires_ffmpeg = pytest.mark.skipif(
    not (ffmpeg_available() and ffprobe_available()),
    reason="ffmpeg/ffprobe not on PATH",
)


@pytest.fixture
def out_root(tmp_path: Path) -> Path:
    """An output root for a generation run."""
    root = tmp_path / "data"
    root.mkdir()
    return root


@pytest.fixture
def questions_file(tmp_path: Path) -> Path:
    """A small questions.jsonl covering both accepted record shapes."""
    path = tmp_path / "questions.jsonl"
    records = [
        {
            "id": "q_0001",
            "domain": "stem",
            "difficulty": "easy",
            "source": "MMLU",
            "stem": "What is the SI unit of electric current?",
            "options": {"A": "Volt", "B": "Ohm", "C": "Ampere", "D": "Watt"},
            "answer": "C",
        },
        {
            "question": "What is the capital of France?",
            "choices": ["Berlin", "Madrid", "Paris", "Rome"],
            "answer": 2,
            "subject": "geography",
        },
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return path


@pytest.fixture
def chess_config(out_root: Path) -> dict:
    """A minimal chess config mapping (fast to generate)."""
    return {
        "experiment": "chess",
        "run": {
            "output_root": str(out_root),
            "run_id": "test",
            "seed": 7,
            "workers": 2,
            "resume": False,
        },
        "video": {"fps": 30, "preset": "ultrafast", "crf": 30},
        "params": {
            "num_games": 2,
            "max_half_moves": 40,
            "motion": "static",
            "durations": [{"label": "3s", "seconds": 3}],
            "timing": {"intro_frames": 6, "hold_frames": 6, "outro_frames": 6},
            "theme": {"square": 16, "padding": 8, "panel": 16},
        },
    }


@pytest.fixture
def wbw_config(out_root: Path, questions_file: Path) -> dict:
    """A minimal word-by-word MCQ config."""
    return {
        "experiment": "wbw_mcq",
        "run": {
            "output_root": str(out_root),
            "run_id": "test",
            "workers": 2,
            "resume": False,
        },
        "video": {"fps": 10, "preset": "ultrafast", "crf": 35},
        "params": {
            "questions_path": str(questions_file),
            "speeds": {"fast": 5.0},
            "modes": ["cumulative"],
            "canvas": {"width": 320, "height": 240, "video_font_size": 14,
                       "video_padding": 20, "static_font_size": 10,
                       "static_padding": 16},
        },
    }


def write_yaml(path: Path, data: dict) -> Path:
    """Dump a mapping to YAML at `path`."""
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path
