"""Word-by-word MCQ: question parsing, rendering, and generator params."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pov.config import Config, ConfigError
from pov.experiments.wbw_mcq.generate import WbwMcqGenerator, frames_per_word
from pov.experiments.wbw_mcq.question import (
    Question,
    QuestionError,
    load_questions,
    normalise_answer,
)
from pov.experiments.wbw_mcq.render import Canvas, FrameRenderer, wrap_text

OPTIONS = {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"}


def make_question(**overrides) -> Question:
    fields = dict(id="q1", stem="Pick one?", options=dict(OPTIONS), answer="B")
    fields.update(overrides)
    return Question(**fields)


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return path


# ── Answer normalisation ──────────────────────────────────────────────────────


class TestNormaliseAnswer:
    @pytest.mark.parametrize(
        "value,expected",
        [("A", "A"), ("b", "B"), (" c ", "C"), (0, "A"), (3, "D"), ("2", "C")],
    )
    def test_accepts_letters_and_indices(self, value, expected):
        assert normalise_answer(value) == expected

    def test_one_based_string_fallback(self):
        # "4" cannot be a 0-based index, so it is read as 1-based.
        assert normalise_answer("4") == "D"

    def test_integer_four_is_out_of_range(self):
        # A real int is always 0-based; 4 has no option.
        with pytest.raises(QuestionError, match="out of range"):
            normalise_answer(4)

    @pytest.mark.parametrize("bad", ["E", "", "hello", 9, True])
    def test_rejects_nonsense(self, bad):
        with pytest.raises(QuestionError):
            normalise_answer(bad)


# ── Question ──────────────────────────────────────────────────────────────────


class TestQuestion:
    def test_word_sequence_includes_labels(self):
        question = make_question(stem="Which one is best?")
        assert question.word_sequence == [
            "Which", "one", "is", "best?",
            "(A)", "alpha", "(B)", "beta", "(C)", "gamma", "(D)", "delta",
        ]

    def test_punctuation_stays_attached(self):
        question = make_question(stem="What is mitosis, exactly?")
        assert "mitosis," in question.word_sequence
        assert "exactly?" in question.word_sequence

    def test_word_count_matches_sequence(self):
        question = make_question()
        assert question.word_count == len(question.word_sequence)

    def test_multiword_options_are_split(self):
        question = make_question(options={**OPTIONS, "A": "two words"})
        assert "two" in question.word_sequence and "words" in question.word_sequence

    def test_explicit_word_sequence_is_respected(self):
        question = make_question(word_sequence=["only", "these"])
        assert question.word_sequence == ["only", "these"]

    def test_rejects_empty_stem(self):
        with pytest.raises(QuestionError, match="stem is empty"):
            make_question(stem="   ")

    def test_rejects_missing_option(self):
        with pytest.raises(QuestionError, match="missing option"):
            make_question(options={"A": "a", "B": "b", "C": "c", "D": ""})

    def test_rejects_answer_without_option(self):
        with pytest.raises(QuestionError):
            make_question(answer="E")

    def test_format_as_text(self):
        text = make_question().format_as_text()
        assert text.startswith("Pick one?")
        assert "(C) gamma" in text

    def test_to_dict_round_trip(self):
        original = make_question()
        restored = Question.from_dict(original.to_dict())
        assert restored.word_sequence == original.word_sequence
        assert restored.answer == original.answer


class TestQuestionFromDict:
    def test_options_dict_shape(self):
        question = Question.from_dict(
            {"id": "x", "stem": "S?", "options": OPTIONS, "answer": "A"}
        )
        assert question.answer == "A"

    def test_huggingface_choices_shape(self):
        question = Question.from_dict(
            {"question": "Capital of France?",
             "choices": ["Berlin", "Madrid", "Paris", "Rome"],
             "answer": 2}
        )
        assert question.options["C"] == "Paris" and question.answer == "C"

    def test_flat_letter_keys(self):
        question = Question.from_dict(
            {"stem": "S?", "A": "a", "B": "b", "C": "c", "D": "d", "answer": "D"}
        )
        assert question.options["D"] == "d"

    def test_options_as_list(self):
        question = Question.from_dict(
            {"stem": "S?", "options": ["a", "b", "c", "d"], "answer": 1}
        )
        assert question.options["B"] == "b"

    def test_rejects_wrong_number_of_choices(self):
        with pytest.raises(QuestionError, match="expected 4 choices"):
            Question.from_dict({"stem": "S?", "choices": ["a", "b"], "answer": 0})

    def test_rejects_missing_answer(self):
        with pytest.raises(QuestionError, match="no 'answer'"):
            Question.from_dict({"stem": "S?", "options": OPTIONS})

    def test_rejects_missing_stem(self):
        with pytest.raises(QuestionError, match="no 'stem' or 'question'"):
            Question.from_dict({"options": OPTIONS, "answer": "A"})

    def test_rejects_missing_options(self):
        with pytest.raises(QuestionError, match="no options"):
            Question.from_dict({"stem": "S?", "answer": "A"})


class TestLoadQuestions:
    def test_loads_jsonl(self, tmp_path):
        path = write_jsonl(tmp_path / "q.jsonl", [
            {"id": "a", "stem": "S1?", "options": OPTIONS, "answer": "A"},
            {"id": "b", "stem": "S2?", "options": OPTIONS, "answer": "B"},
        ])
        assert [q.id for q in load_questions(path)] == ["a", "b"]

    def test_blank_lines_ignored(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text(
            json.dumps({"id": "a", "stem": "S?", "options": OPTIONS, "answer": "A"})
            + "\n\n\n"
        )
        assert len(load_questions(path)) == 1

    def test_auto_ids_never_collide_with_explicit_ids(self, tmp_path):
        # Regression: an auto id of q_0000 must not clash with an explicit one.
        path = write_jsonl(tmp_path / "q.jsonl", [
            {"stem": "auto?", "options": OPTIONS, "answer": "A"},
            {"id": "q_0000", "stem": "explicit?", "options": OPTIONS, "answer": "B"},
        ])
        ids = [q.id for q in load_questions(path)]
        assert len(set(ids)) == 2 and "q_0000" in ids

    def test_duplicate_explicit_ids_rejected(self, tmp_path):
        path = write_jsonl(tmp_path / "q.jsonl", [
            {"id": "same", "stem": "S1?", "options": OPTIONS, "answer": "A"},
            {"id": "same", "stem": "S2?", "options": OPTIONS, "answer": "B"},
        ])
        with pytest.raises(QuestionError, match="duplicate question id"):
            load_questions(path)

    def test_bad_json_reports_line_number(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text('{"id": "a"}\nnot json\n')
        with pytest.raises(QuestionError, match=":2:"):
            load_questions(path)

    def test_invalid_record_reports_line_number(self, tmp_path):
        path = write_jsonl(tmp_path / "q.jsonl", [
            {"id": "a", "stem": "S?", "options": OPTIONS, "answer": "A"},
            {"id": "b", "stem": "", "options": OPTIONS, "answer": "A"},
        ])
        with pytest.raises(QuestionError, match=":2:"):
            load_questions(path)

    def test_missing_file(self, tmp_path):
        with pytest.raises(QuestionError, match="not found"):
            load_questions(tmp_path / "nope.jsonl")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text("")
        with pytest.raises(QuestionError, match="no questions"):
            load_questions(path)

    def test_mmlu_csv(self, tmp_path):
        path = tmp_path / "astronomy_test.csv"
        path.write_text("What orbits the Sun?,Moon,Earth,Mars bar,Pluto,B\n")
        questions = load_questions(path)
        assert questions[0].answer == "B"
        assert questions[0].options["A"] == "Moon"
        assert questions[0].source == "astronomy"

    def test_mmlu_csv_skips_header(self, tmp_path):
        path = tmp_path / "x.csv"
        path.write_text("question,A,B,C,D,answer\nQ?,a,b,c,d,C\n")
        questions = load_questions(path)
        assert len(questions) == 1 and questions[0].answer == "C"

    def test_mmlu_csv_wrong_column_count(self, tmp_path):
        path = tmp_path / "x.csv"
        path.write_text("Q?,a,b\n")
        with pytest.raises(QuestionError, match="expected 6 columns"):
            load_questions(path)

    def test_unknown_format(self, tmp_path):
        path = write_jsonl(tmp_path / "q.jsonl", [
            {"id": "a", "stem": "S?", "options": OPTIONS, "answer": "A"}
        ])
        with pytest.raises(QuestionError, match="unknown questions format"):
            load_questions(path, fmt="parquet")


# ── Rendering ─────────────────────────────────────────────────────────────────


class TestCanvas:
    def test_defaults_are_valid(self):
        assert Canvas().validate().size == (1200, 800)

    @pytest.mark.parametrize("width,height", [(1201, 800), (1200, 801)])
    def test_rejects_odd_dimensions(self, width, height):
        with pytest.raises(ValueError, match="even dimensions"):
            Canvas(width=width, height=height).validate()

    def test_rejects_tiny_canvas(self):
        with pytest.raises(ValueError, match="too small"):
            Canvas(width=32, height=32).validate()

    def test_rejects_padding_wider_than_canvas(self):
        with pytest.raises(ValueError, match="video_padding"):
            Canvas(width=100, height=100, video_padding=60).validate()

    def test_line_heights(self):
        canvas = Canvas(video_font_size=36, video_line_gap=8)
        assert canvas.video_line_height == 44


class TestWrapText:
    def setup_method(self):
        self.image = Image.new("RGB", (400, 100))
        self.draw = ImageDraw.Draw(self.image)
        from pov.experiments.wbw_mcq.render import _font

        self.font = _font(14)

    def test_short_text_is_one_line(self):
        assert wrap_text("hello world", self.font, self.draw, 400) == ["hello world"]

    def test_empty_text_yields_one_empty_line(self):
        assert wrap_text("", self.font, self.draw, 400) == [""]

    def test_long_text_wraps(self):
        lines = wrap_text("word " * 60, self.font, self.draw, 200)
        assert len(lines) > 1

    def test_unbreakable_word_is_kept_not_dropped(self):
        # A single token wider than the line must still appear.
        lines = wrap_text("A" * 200, self.font, self.draw, 50)
        assert lines == ["A" * 200]

    def test_no_words_are_lost(self):
        text = " ".join(f"w{i}" for i in range(40))
        lines = wrap_text(text, self.font, self.draw, 120)
        assert " ".join(lines).split() == text.split()


class TestFrameRenderer:
    def test_blank_frame_size_and_colour(self):
        renderer = FrameRenderer(Canvas(width=200, height=100))
        blank = renderer.render_blank()
        assert blank.shape == (100, 200, 3)
        assert np.all(blank == 255)

    def test_blank_frame_is_cached(self):
        renderer = FrameRenderer(Canvas(width=200, height=100))
        assert renderer.render_blank() is renderer.render_blank()

    def test_word_frame_has_dark_pixels(self):
        renderer = FrameRenderer(Canvas(width=320, height=200, video_font_size=20,
                                        video_padding=10))
        frame = renderer.render_words(["hello"])
        assert frame.shape == (200, 320, 3)
        assert frame.min() < 128  # text was drawn

    def test_more_words_means_more_ink(self):
        renderer = FrameRenderer(Canvas(width=320, height=200, video_font_size=16,
                                        video_padding=10))
        one = (renderer.render_words(["hello"]) < 128).sum()
        many = (renderer.render_words(["hello", "there", "world"]) < 128).sum()
        assert many > one

    def test_full_question_image_size(self):
        renderer = FrameRenderer(Canvas(width=320, height=240, static_font_size=10,
                                        static_padding=12))
        image = renderer.render_full_question(make_question())
        assert isinstance(image, Image.Image) and image.size == (320, 240)

    def test_frames_are_writable_copies(self):
        renderer = FrameRenderer(Canvas(width=64, height=64, video_font_size=10,
                                        video_padding=4, static_padding=4))
        frame = renderer.render_words(["a"])
        frame[0, 0] = 7  # must not raise


# ── Generator params ──────────────────────────────────────────────────────────


class TestFramesPerWord:
    @pytest.mark.parametrize(
        "fps,wps,expected", [(30, 0.5, 60), (30, 2.0, 15), (30, 5.0, 6), (10, 5.0, 2)]
    )
    def test_frames_per_word(self, fps, wps, expected):
        assert frames_per_word(fps, wps) == expected

    def test_minimum_is_enforced(self):
        assert frames_per_word(30, 1000.0) == 1
        assert frames_per_word(30, 1000.0, minimum=3) == 3

    def test_rejects_non_positive_speed(self):
        with pytest.raises(ConfigError, match="> 0"):
            frames_per_word(30, 0)


class TestGeneratorParams:
    def build(self, params: dict, questions_file: Path):
        payload = {"experiment": "wbw_mcq",
                   "params": {"questions_path": str(questions_file), **params}}
        return WbwMcqGenerator(Config.from_mapping(payload))

    def test_defaults(self, questions_file):
        params = self.build({}, questions_file).params
        assert params.speeds == {"slow": 0.5, "normal": 2.0, "fast": 5.0}
        assert params.modes == ("vanishing", "cumulative")
        assert params.static_image is True

    def test_questions_path_is_required(self):
        with pytest.raises(ConfigError, match="questions_path"):
            WbwMcqGenerator(Config.from_mapping({"experiment": "wbw_mcq"}))

    def test_rejects_unknown_mode(self, questions_file):
        with pytest.raises(ConfigError, match="not one of"):
            self.build({"modes": ["sideways"]}, questions_file)

    def test_duplicate_modes_are_collapsed(self, questions_file):
        params = self.build({"modes": ["vanishing", "vanishing"]}, questions_file).params
        assert params.modes == ("vanishing",)

    def test_rejects_nothing_to_generate(self, questions_file):
        with pytest.raises(ConfigError, match="nothing to generate"):
            self.build({"modes": [], "static_image": False}, questions_file)

    def test_static_only_is_allowed(self, questions_file):
        params = self.build({"modes": [], "static_image": True}, questions_file).params
        assert params.modes == ()

    def test_rejects_non_positive_speed(self, questions_file):
        with pytest.raises(ConfigError, match="must be >="):
            self.build({"speeds": {"zero": 0}}, questions_file)

    def test_rejects_unsafe_speed_name(self, questions_file):
        with pytest.raises(ConfigError, match="alphanumeric"):
            self.build({"speeds": {"a/b": 1.0}}, questions_file)

    def test_rejects_bad_limit(self, questions_file):
        with pytest.raises(ConfigError, match="limit"):
            self.build({"limit": 0}, questions_file)

    def test_limit_may_be_null(self, questions_file):
        assert self.build({"limit": None}, questions_file).params.limit is None

    def test_rejects_unknown_param(self, questions_file):
        with pytest.raises(ConfigError, match="unknown key"):
            self.build({"speed": 2.0}, questions_file)

    def test_rejects_odd_canvas(self, questions_file):
        with pytest.raises(ConfigError, match="even dimensions"):
            self.build({"canvas": {"width": 101, "height": 100}}, questions_file)
