"""Scoring: text metrics, chess transcription, MCQ answers, ASL translation."""

from __future__ import annotations

import pytest

from pov.eval.asl import AslScorer, parse_judge
from pov.eval.chess import (
    ChessScorer,
    hybrid_score,
    lcs_alignment,
    loose_score,
    parse_model_output,
    parse_transcript,
    strict_score,
)
from pov.eval.mcq import McqScorer, is_refusal, parse_answer
from pov.eval.text import (
    bleu,
    char_similarity,
    edit_distance,
    exact_match,
    normalise,
    token_f1,
    tokenize,
    word_error_rate,
)

TRANSCRIPT = """Game #0 — Move Transcription
==============================================

Move    Turn      Source    Destination
----    ----      ------    -----------
1       White     b2        b3
1       Black     c7        c5
2       White     g1        f3
2       Black     b8        c6

Result: Move limit reached (4 half-moves) — Draw.
Total half-moves: 4
"""


# ── Text metrics ──────────────────────────────────────────────────────────────


class TestNormalise:
    def test_lowercases_and_strips_punctuation(self):
        assert normalise("Hello, World!") == "hello world"

    def test_collapses_whitespace(self):
        assert normalise("one   two\n\tthree") == "one two three"

    def test_strips_articles_by_default(self):
        # ASL omits articles, so scoring must not penalise their absence.
        assert normalise("the cat and a dog") == "cat and dog"

    def test_article_only_text_normalises_to_empty(self):
        assert normalise("a the an") == ""

    def test_can_keep_articles(self):
        assert normalise("the cat", strip_articles=False) == "the cat"

    def test_strips_accents(self):
        assert normalise("café naïve") == "cafe naive"

    def test_empty(self):
        assert normalise("") == "" and tokenize("") == []


class TestExactMatch:
    def test_identical(self):
        assert exact_match("Hello there.", "hello there") == 1.0

    def test_different(self):
        assert exact_match("hello", "goodbye") == 0.0

    def test_both_empty(self):
        assert exact_match("", "") == 1.0


class TestTokenF1:
    def test_identical_is_one(self):
        assert token_f1("the quick brown fox", "The quick brown fox!") == 1.0

    def test_disjoint_is_zero(self):
        assert token_f1("alpha beta", "gamma delta") == 0.0

    def test_partial_overlap(self):
        score = token_f1("a b c d", "a b x y")
        assert 0.0 < score < 1.0

    def test_empty_hypothesis_is_zero(self):
        assert token_f1("something", "") == 0.0

    def test_both_empty_is_one(self):
        assert token_f1("", "") == 1.0

    def test_is_symmetric(self):
        assert token_f1("a b c", "b c d") == pytest.approx(token_f1("b c d", "a b c"))

    def test_repeated_tokens_use_multiset_intersection(self):
        # One "cat" in the hypothesis cannot match two in the reference twice.
        assert token_f1("cat cat dog", "cat dog") == pytest.approx(0.8)


class TestBleu:
    def test_identical_is_one(self):
        assert bleu("the quick brown fox jumps", "the quick brown fox jumps") == pytest.approx(1.0)

    def test_no_unigram_overlap_is_zero(self):
        assert bleu("alpha beta gamma", "delta epsilon zeta") == 0.0

    def test_partial_match_is_between(self):
        score = bleu("the cat sat on the mat", "the cat sat on a hat")
        assert 0.0 < score < 1.0

    def test_empty_is_zero(self):
        assert bleu("something", "") == 0.0
        assert bleu("", "something") == 0.0

    def test_short_hypothesis_is_penalised(self):
        full = bleu("the cat sat on the mat", "the cat sat on the mat")
        short = bleu("the cat sat on the mat", "the cat")
        assert short < full

    def test_smoothing_keeps_short_matches_nonzero(self):
        # Unsmoothed BLEU-4 would be 0 here; smoothing must preserve signal.
        assert bleu("hello world", "hello world") > 0


class TestEditDistance:
    @pytest.mark.parametrize(
        "a,b,expected",
        [("", "", 0), ("abc", "abc", 0), ("abc", "abd", 1),
         ("abc", "ab", 1), ("", "abc", 3), ("kitten", "sitting", 3)],
    )
    def test_distance(self, a, b, expected):
        assert edit_distance(a, b) == expected

    def test_is_symmetric(self):
        assert edit_distance("flaw", "lawn") == edit_distance("lawn", "flaw")


class TestCharSimilarity:
    def test_identical_is_one(self):
        assert char_similarity("hello world", "Hello, world!") == 1.0

    def test_completely_different_is_low(self):
        assert char_similarity("aaaa", "zzzzzzzz") < 0.3

    def test_never_negative(self):
        assert char_similarity("a", "b" * 100) >= 0.0

    def test_both_empty(self):
        assert char_similarity("", "") == 1.0


class TestWordErrorRate:
    def test_perfect_is_zero(self):
        assert word_error_rate("a b c", "a b c") == 0.0

    def test_one_substitution(self):
        assert word_error_rate("one two three", "one xxx three") == pytest.approx(1 / 3)

    def test_empty_hypothesis_is_one(self):
        assert word_error_rate("one two three", "") == 1.0

    def test_may_exceed_one_with_insertions(self):
        assert word_error_rate("one", "www xxx yyy zzz") > 1.0

    def test_empty_reference(self):
        assert word_error_rate("", "") == 0.0
        assert word_error_rate("", "word") == 1.0

    def test_articles_are_not_penalised(self):
        # "the dog" and "dog" are the same utterance for this benchmark.
        assert word_error_rate("the dog ran", "dog ran") == 0.0


# ── Chess scoring ─────────────────────────────────────────────────────────────


class TestParseTranscript:
    def test_parses_table_format(self):
        moves = parse_transcript(TRANSCRIPT)
        assert len(moves) == 4
        assert moves[0] == {"move_num": 1, "color": "White", "from": "b2", "to": "b3"}

    def test_skips_headers_and_footers(self):
        assert all(m["from"][0] in "abcdefgh" for m in parse_transcript(TRANSCRIPT))

    def test_parses_compact_format(self):
        moves = parse_transcript("1 White b2 b3; 1 Black c7 c5")
        assert len(moves) == 2 and moves[1]["to"] == "c5"

    def test_orders_white_before_black(self):
        moves = parse_transcript("1 Black c7 c5; 1 White b2 b3")
        assert moves[0]["color"] == "White"

    def test_n_moves_limit(self):
        assert len(parse_transcript(TRANSCRIPT, n_moves=1)) == 2

    def test_empty_input(self):
        assert parse_transcript("") == []


class TestParseModelOutput:
    def test_move_n_format(self):
        moves = parse_model_output("Move 1: White Pawn b2b3\nMove 1: Black Pawn c7c5")
        assert len(moves) == 2 and moves[0]["from"] == "b2"

    def test_numbered_list_format(self):
        moves = parse_model_output("1. White Pawn b2 b3\n2. Black Knight b8 c6")
        assert len(moves) == 2 and moves[1]["to"] == "c6"

    def test_bare_format_gets_sequential_numbers(self):
        moves = parse_model_output("White b2b3\nBlack c7c5")
        assert [m["move_num"] for m in moves] == [1, 2]

    def test_arrow_separator(self):
        moves = parse_model_output("Move 1: White Pawn b2 -> b3")
        assert moves[0] == {"move_num": 1, "color": "White", "from": "b2", "to": "b3"}

    def test_case_insensitive(self):
        assert parse_model_output("MOVE 1: WHITE PAWN B2B3")[0]["from"] == "b2"

    def test_ignores_prose(self):
        assert parse_model_output("I cannot read this video.") == []

    def test_empty(self):
        assert parse_model_output("") == []

    def test_specific_pattern_wins_over_fallback(self):
        # A numbered list must not be re-parsed by the looser bare pattern.
        moves = parse_model_output("Move 1: White Pawn b2b3")
        assert len(moves) == 1


class TestChessScores:
    def truth(self):
        return parse_transcript(TRANSCRIPT)

    def test_perfect_scores_one(self):
        truth = self.truth()
        assert strict_score(truth, truth) == 1.0
        assert loose_score(truth, truth) == 1.0
        assert hybrid_score(truth, truth) == 1.0

    def test_empty_prediction_scores_zero(self):
        truth = self.truth()
        assert strict_score(truth, []) == 0.0
        assert loose_score(truth, []) == 0.0
        assert hybrid_score(truth, []) == 0.0

    def test_empty_truth_scores_zero(self):
        assert strict_score([], self.truth()) == 0.0

    def test_strict_stops_at_first_mistake(self):
        truth = self.truth()
        predicted = list(truth)
        predicted[1] = {**predicted[1], "to": "h6"}
        assert strict_score(truth, predicted) == pytest.approx(0.25)

    def test_loose_allows_gaps(self):
        truth = self.truth()
        predicted = [truth[0], truth[3]]
        assert loose_score(truth, predicted) == pytest.approx(0.5)

    def test_hybrid_ignores_isolated_hits(self):
        truth = self.truth()
        predicted = [truth[0], truth[3]]      # two isolated correct moves
        assert hybrid_score(truth, predicted) == 0.0

    def test_hybrid_counts_consecutive_pairs(self):
        truth = self.truth()
        predicted = [truth[0], truth[1]]
        assert hybrid_score(truth, predicted) == pytest.approx(0.5)

    def test_ordering_invariant_holds(self):
        truth = self.truth()
        predicted = [truth[0], truth[1], truth[3]]
        assert strict_score(truth, predicted) <= hybrid_score(truth, predicted)
        assert hybrid_score(truth, predicted) <= loose_score(truth, predicted)

    def test_piece_name_is_ignored_in_matching(self):
        truth = self.truth()
        text = "\n".join(
            f"Move {m['move_num']}: {m['color']} Queen {m['from']}{m['to']}" for m in truth
        )
        assert strict_score(truth, parse_model_output(text)) == 1.0

    def test_wrong_colour_does_not_match(self):
        truth = self.truth()
        predicted = [{**truth[0], "color": "Black"}]
        assert strict_score(truth, predicted) == 0.0

    def test_lcs_alignment_is_ordered(self):
        truth = self.truth()
        alignment = lcs_alignment(truth, [truth[3], truth[0]])
        assert len(alignment) == 1

    def test_extra_predictions_do_not_help_strict(self):
        truth = self.truth()
        predicted = [{"move_num": 0, "color": "White", "from": "a1", "to": "a2"}] + truth
        assert strict_score(truth, predicted) == 0.0
        assert loose_score(truth, predicted) == 1.0


class TestChessScorer:
    def test_scores_a_row(self):
        scorer = ChessScorer()
        text = "\n".join(
            f"Move {m['move_num']}: {m['color']} Pawn {m['from']}{m['to']}"
            for m in parse_transcript(TRANSCRIPT)
        )
        scores = scorer.score({"model_output": text, "n_half_moves": 4}, TRANSCRIPT)
        assert scores["strict"] == 1.0
        assert scores["moves_expected"] == 4 and scores["moves_predicted"] == 4

    def test_truth_is_capped_by_clip_length(self):
        scorer = ChessScorer()
        scores = scorer.score({"model_output": "", "n_half_moves": 2}, TRANSCRIPT)
        assert scores["moves_expected"] == 2

    def test_missing_n_half_moves_uses_whole_transcript(self):
        scores = ChessScorer().score({"model_output": ""}, TRANSCRIPT)
        assert scores["moves_expected"] == 4

    def test_empty_scores(self):
        assert ChessScorer().empty_scores()["strict"] == 0.0


# ── MCQ scoring ───────────────────────────────────────────────────────────────


class TestParseAnswer:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("ANSWER: B", "B"),
            ("answer: c", "C"),
            ("ANSWER:D", "D"),
            ("ANSWER - A", "A"),
            ("REASONING: blah\nANSWER: C", "C"),
            ("B) Cell division", "B"),
            ("(C) Paris", "C"),
            ("The correct choice is (D).", "D"),
            ("I think B is right", "B"),
        ],
    )
    def test_extraction(self, text, expected):
        assert parse_answer(text) == expected

    def test_structured_tag_beats_earlier_letters(self):
        assert parse_answer("Option C looks plausible but ANSWER: A") == "A"

    @pytest.mark.parametrize("text", ["", "no letters here", "option E", "42"])
    def test_returns_none_when_absent(self, text):
        assert parse_answer(text) is None

    def test_lowercase_article_is_not_an_answer(self):
        assert parse_answer("this is a plausible response") is None


class TestRefusal:
    @pytest.mark.parametrize(
        "text",
        ["I'm sorry, I cannot help with that.", "As an AI, I am unable to help.",
         "I apologize, but the video is unreadable."],
    )
    def test_detects_refusals(self, text):
        assert is_refusal(text) is True

    def test_refusal_with_an_answer_is_not_a_refusal(self):
        assert is_refusal("I'm sorry, but the answer is ANSWER: B") is False

    def test_normal_response_is_not_a_refusal(self):
        assert is_refusal("ANSWER: A") is False

    def test_empty_is_not_a_refusal(self):
        assert is_refusal("") is False


class TestMcqScorer:
    def test_correct_answer(self):
        scores = McqScorer().score({"model_output": "ANSWER: B"}, "B")
        assert scores == {"correct": 1.0, "answered": 1.0, "refusal": 0.0,
                          "predicted_answer": "B"}

    def test_wrong_answer(self):
        scores = McqScorer().score({"model_output": "ANSWER: A"}, "B")
        assert scores["correct"] == 0.0 and scores["answered"] == 1.0

    def test_refusal_is_flagged_and_unanswered(self):
        scores = McqScorer().score({"model_output": "I'm sorry, I cannot."}, "B")
        assert scores["refusal"] == 1.0 and scores["answered"] == 0.0

    def test_ground_truth_case_is_ignored(self):
        assert McqScorer().score({"model_output": "ANSWER: b"}, "b")["correct"] == 1.0


# ── ASL scoring ───────────────────────────────────────────────────────────────


class TestParseJudge:
    @pytest.mark.parametrize(
        "value,expected",
        [(True, 1.0), (False, 0.0), (1, 1.0), ("yes", 1.0), ("NO", 0.0),
         ("true", 1.0), ("0", 0.0), ("0.75", 0.75), ("correct", 1.0)],
    )
    def test_parses_truthy_forms(self, value, expected):
        assert parse_judge(value) == expected

    @pytest.mark.parametrize("value", [None, "", "maybe", "n/a "])
    def test_unparseable_is_none(self, value):
        assert parse_judge(value) is None


class TestAslScorer:
    def test_perfect_translation(self):
        scores = AslScorer().score({"model_output": "Open your eyes."}, "Open your eyes")
        assert scores["exact_match"] == 1.0
        assert scores["token_f1"] == 1.0
        assert scores["wer"] == 0.0

    def test_wrong_translation(self):
        scores = AslScorer().score({"model_output": "Completely unrelated"}, "Open your eyes")
        assert scores["exact_match"] == 0.0
        assert scores["token_f1"] == 0.0

    def test_partial_translation(self):
        scores = AslScorer().score({"model_output": "Open your"}, "Open your eyes now")
        assert 0.0 < scores["token_f1"] < 1.0

    def test_empty_output(self):
        scores = AslScorer().score({"model_output": ""}, "Open your eyes")
        assert scores["token_f1"] == 0.0 and scores["bleu"] == 0.0

    def test_judge_columns_are_passed_through(self):
        scores = AslScorer().score(
            {"model_output": "x", "judge_strict": "yes", "judge_loose": "0"}, "y"
        )
        assert scores["judge_strict"] == 1.0 and scores["judge_loose"] == 0.0

    def test_judge_columns_absent_when_not_supplied(self):
        scores = AslScorer().score({"model_output": "x"}, "y")
        assert "judge_strict" not in scores

    def test_unparseable_judge_is_omitted(self):
        scores = AslScorer().score({"model_output": "x", "judge_strict": "maybe"}, "y")
        assert "judge_strict" not in scores
