"""Task prompts: registry, rendering, and agreement with the rest of pov.

The prompt is part of the benchmark, so these tests check more than "a string
comes back": that every experiment has one, that every condition the generators
emit maps to a message, that the judge's output contract is what the ASL scorer
actually parses, and that the shipped chess prompt describes the frames the
renderer really produces.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pov import prompts
from pov.errors import PovError
from pov.prompts import PromptError

#: The original study, used for fidelity checks when it is present.
REFERENCE = Path(
    "/Users/aditya/Programming/sethu_research/pov-stuff/icml_workshop"
)
requires_reference = pytest.mark.skipif(
    not REFERENCE.is_dir(), reason="original icml_workshop repo not available"
)


# ── Registry ──────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_every_experiment_has_a_task_prompt(self):
        # The point of the module: no experiment may ship without an
        # instruction telling the model what to do.
        from pov.registry import EXPERIMENTS

        for experiment in EXPERIMENTS:
            assert prompts.get(experiment, "task").strip()

    def test_experiments_match_the_generator_registry(self):
        from pov.registry import EXPERIMENTS

        assert set(prompts.experiments()) == set(EXPERIMENTS)

    def test_available_pairs_are_sorted_and_unique(self):
        pairs = prompts.available()
        assert pairs == sorted(pairs)
        assert len(pairs) == len(set(pairs))

    def test_every_registered_prompt_loads_and_is_non_trivial(self):
        for experiment, kind in prompts.available():
            text = prompts.get(experiment, kind)
            assert isinstance(text, str)
            assert len(text.strip()) > 20, f"{experiment}/{kind} looks empty"

    def test_kinds_lists_only_that_experiment(self):
        assert prompts.kinds("asl") == ["judge", "task"]
        assert "judge" not in prompts.kinds("chess")

    def test_default_kind_is_task(self):
        assert prompts.DEFAULT_KIND == "task"
        assert prompts.get("asl") == prompts.get("asl", "task")

    def test_unknown_experiment_lists_the_alternatives(self):
        with pytest.raises(PromptError, match="Available: "):
            prompts.get("mystery")

    def test_unknown_kind_lists_the_alternatives(self):
        with pytest.raises(PromptError, match="Available kinds: "):
            prompts.get("asl", "nonsense")

    def test_prompt_error_is_a_pov_error(self):
        # So the CLI reports it as bad input, not an internal crash.
        assert issubclass(PromptError, PovError)
        with pytest.raises(PovError):
            prompts.get("mystery")

    def test_every_data_file_is_registered(self):
        # A stray .txt nobody can reach is a porting mistake.
        data_dir = Path(prompts.__file__).parent / "data"
        on_disk = {p.name for p in data_dir.glob("*.txt")}
        registered = set(prompts._REGISTRY.values())
        assert on_disk == registered


# ── Placeholders and rendering ────────────────────────────────────────────────


class TestPlaceholders:
    def test_task_prompts_need_no_fields(self):
        for experiment in prompts.experiments():
            assert prompts.placeholders(experiment, "task") == []

    def test_judge_needs_the_pair_being_scored(self):
        assert prompts.placeholders("asl", "judge") == ["ground_truth", "model_output"]

    def test_mcq_text_message_needs_the_question(self):
        assert prompts.placeholders("wbw_mcq", "user_text") == [
            "A", "B", "C", "D", "stem"
        ]

    def test_doubled_braces_are_not_placeholders(self):
        # The judge's output contract contains a literal {"strict": ...}.
        assert "explanation" not in prompts.placeholders("asl", "judge")


class TestRender:
    def test_fills_the_fields(self):
        text = prompts.render(
            "asl", "judge", ground_truth="a cat sat", model_output="the cat sat"
        )
        assert "a cat sat" in text and "the cat sat" in text
        assert "{ground_truth}" not in text

    def test_literal_braces_survive(self):
        text = prompts.render("asl", "judge", ground_truth="x", model_output="y")
        assert '{"strict"' in text
        assert "{{" not in text

    def test_missing_field_is_refused(self):
        # Half-rendering a judge prompt would score the wrong thing silently.
        with pytest.raises(PromptError, match="missing"):
            prompts.render("asl", "judge", ground_truth="x")

    def test_unexpected_field_is_refused(self):
        with pytest.raises(PromptError, match="unexpected"):
            prompts.render("asl", "judge", ground_truth="x", model_output="y", extra="z")

    def test_render_on_a_prompt_with_no_fields(self):
        assert prompts.render("chess") == prompts.get("chess")

    def test_mcq_text_message_renders(self):
        text = prompts.render(
            "wbw_mcq", "user_text",
            stem="What is 2+2?", A="3", B="4", C="5", D="6",
        )
        assert "What is 2+2?" in text and "(B) 4" in text

    def test_values_with_braces_do_not_break_rendering(self):
        text = prompts.render(
            "asl", "judge", ground_truth="a {weird} value", model_output="y"
        )
        assert "a {weird} value" in text


# ── Condition mapping ─────────────────────────────────────────────────────────


class TestForCondition:
    @pytest.mark.parametrize(
        "condition,expected_kind",
        [
            ("text", "user_text"),
            ("static_image", "user_static_image"),
            ("vanishing_slow", "user_video"),
            ("vanishing_fast", "user_video"),
            ("cumulative_normal", "user_video"),
        ],
    )
    def test_mcq_conditions_map_to_the_right_message(self, condition, expected_kind):
        assert prompts.for_condition("wbw_mcq", condition) == prompts.get(
            "wbw_mcq", expected_kind
        )

    @pytest.mark.parametrize("condition", ["video_5s", "video_10min", "video_<3s", "video"])
    def test_chess_and_asl_use_one_prompt_for_every_condition(self, condition):
        assert prompts.for_condition("chess", condition) == prompts.get("chess")
        assert prompts.for_condition("asl", condition) == prompts.get("asl")

    def test_unknown_mcq_condition_is_refused(self):
        with pytest.raises(PromptError, match="no user message"):
            prompts.for_condition("wbw_mcq", "interpretive_dance")

    def test_arbitrary_speed_names_still_map(self):
        # Speeds are user-named in the config, so the prefix must drive this.
        assert prompts.for_condition("wbw_mcq", "vanishing_glacial")
        assert prompts.for_condition("wbw_mcq", "cumulative_turbo")


# ── Agreement with the rest of pov ────────────────────────────────────────────


class TestAgreementWithScorers:
    def test_judge_contract_matches_what_the_asl_scorer_parses(self):
        from pov.eval.asl import JUDGE_COLUMNS, parse_judge

        text = prompts.get("asl", "judge")
        # The prompt demands a JSON object with these keys…
        assert '{{"strict"' in text and '"loose"' in text
        # …and the scorer reads exactly those, as judge_strict / judge_loose.
        assert JUDGE_COLUMNS == ("judge_strict", "judge_loose")
        for value in (True, False):
            assert parse_judge(value) in (0.0, 1.0)

    def test_a_judge_reply_in_the_demanded_shape_parses(self):
        from pov.eval.asl import parse_judge

        reply = json.loads('{"strict": true, "loose": true, "explanation": "ok"}')
        assert parse_judge(reply["strict"]) == 1.0
        assert parse_judge(reply["loose"]) == 1.0

    def test_mcq_prompt_demands_the_format_the_parser_reads(self):
        from pov.eval.mcq import parse_answer

        text = prompts.get("wbw_mcq", "task")
        assert "ANSWER:" in text
        # A reply in the demanded shape must parse.
        assert parse_answer("ANSWER: C\nREASONING: because") == "C"

    def test_chess_prompt_output_shape_is_what_the_parser_reads(self):
        from pov.eval.chess import parse_model_output

        text = prompts.get("chess", "task")
        assert "Move <N>: <Color> <Piece> <from_square><to_square>" in text
        moves = parse_model_output("Move 1: White Pawn e2e4\nMove 1: Black Knight g8f6")
        assert len(moves) == 2
        assert moves[0] == {"move_num": 1, "color": "White", "from": "e2", "to": "e4"}


class TestChessPromptDescribesTheRealFrames:
    """The shipped chess prompt must match what the renderer actually draws."""

    def test_does_not_claim_the_caption_holds_the_move(self):
        # Regression: the original prompt told the model to read
        # "Move N — Color: <algebraic notation>" off the frame. That caption is
        # gone, so instructing the model to read it would be a lie.
        text = prompts.get("chess", "task")
        assert "algebraic notation" not in text
        assert "read EVERY move label" not in text
        assert "for validation" not in text

    def test_says_the_caption_is_only_the_move_number(self):
        text = prompts.get("chess", "task").lower()
        assert "move number" in text

    def test_caption_example_matches_what_the_renderer_writes(self):
        from pov.config import Config
        from pov.experiments.chess.engine import play_random_game
        from pov.experiments.chess.generate import ChessGenerator, _FrameCache

        config = Config.from_mapping({
            "experiment": "chess",
            "params": {"durations": [{"label": "5s", "seconds": 5}]},
        })
        generator = ChessGenerator(config)
        record = play_random_game(seed=1, max_half_moves=20)
        cache = _FrameCache(None, record, generator.params, show_labels=True)
        label, _ = cache._labels(record.moves[0])

        # The prompt promises a caption shaped like "Move 8".
        assert re.fullmatch(r"Move \d+", label), label
        assert 'caption at the bottom giving only the move number' in prompts.get(
            "chess", "task"
        )

    def test_highlight_description_matches_the_theme(self):
        # The original said "yellow = destination"; ours is yellow = origin.
        from pov.experiments.chess.render import BoardTheme

        theme = BoardTheme()
        # from-square is the lighter of the two highlights.
        assert sum(theme.highlight_from) > sum(theme.highlight_to)
        text = prompts.get("chess", "task")
        assert "came FROM" in text and "moved TO" in text

    def test_legacy_prompt_is_kept_but_not_the_default(self):
        legacy = prompts.get("chess", "task_legacy")
        assert "algebraic notation" in legacy  # the original wording
        assert legacy != prompts.get("chess", "task")


# ── Fidelity to the original study ────────────────────────────────────────────


@requires_reference
class TestPortedVerbatim:
    """Prompts carried over unchanged must be byte-identical to the source."""

    def source_constants(self, relative: str) -> dict:
        namespace: dict = {}
        exec((REFERENCE / relative).read_text(), namespace)
        return namespace

    def test_asl_task_is_verbatim(self):
        original = self.source_constants("asl_text_data/prompts.py")["PROMPT_ASL"]
        assert prompts.get("asl", "task") == original

    def test_asl_judge_is_verbatim(self):
        original = self.source_constants("asl_text_data/prompts.py")["EVAL_PROMPT"]
        assert prompts.get("asl", "judge") == original

    def test_chess_legacy_is_verbatim(self):
        original = self.source_constants("chess_game_creator/prompts.py")["PROMPT_1"]
        assert prompts.get("chess", "task_legacy") == original

    def test_mcq_prompts_are_verbatim(self):
        import ast

        tree = ast.parse(
            (REFERENCE / "wbw_mcq/controllers/evaluator.py").read_text()
        )
        found = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                try:
                    found[node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:
                    pass

        assert prompts.get("wbw_mcq", "task") == found["SYSTEM_PROMPT"]
        assert prompts.get("wbw_mcq", "user_text") == found["_USER_MSG"]["text"]
        assert prompts.get("wbw_mcq", "user_static_image") == found["_USER_MSG"]["static_image"]
        assert prompts.get("wbw_mcq", "user_video") == found["_USER_MSG"]["_video"]
