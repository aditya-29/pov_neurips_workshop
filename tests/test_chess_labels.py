"""The on-screen panel must not give away the move.

A label reading "Move 8 - Black: Qd8" lets a model score by reading the caption
instead of the board, which is the opposite of what this benchmark measures.
These tests pin the panel to a bare move number.
"""

from __future__ import annotations

import pytest

from pov.config import Config
from pov.experiments.chess.engine import play_random_game
from pov.experiments.chess.generate import ChessGenerator, _FrameCache


@pytest.fixture
def cache(out_root):
    config = Config.from_mapping({
        "experiment": "chess",
        "run": {"output_root": str(out_root)},
        "params": {"durations": [{"label": "5s", "seconds": 5}]},
    })
    generator = ChessGenerator(config)
    record = play_random_game(seed=7, max_half_moves=60)
    # The renderer is unused by label construction.
    return _FrameCache(None, record, generator.params, show_labels=True), record


class TestLabels:
    def test_label_is_only_the_move_number(self, cache):
        frame_cache, record = cache
        label, sublabel = frame_cache._labels(record.moves[0])
        assert label == f"Move {record.moves[0].move_no}"
        assert sublabel == ""

    def test_label_never_contains_the_answer(self, cache):
        frame_cache, record = cache
        for move in record.moves:
            label, sublabel = frame_cache._labels(move)
            text = f"{label} {sublabel}"
            # No squares, no piece name, no colour, no algebraic notation.
            assert move.src not in text
            assert move.dst not in text
            assert move.piece not in text
            assert move.color not in text
            assert move.notation not in text

    def test_no_square_name_appears_at_all(self, cache):
        frame_cache, record = cache
        squares = {f"{file}{rank}" for file in "abcdefgh" for rank in "12345678"}
        for move in record.moves:
            label, sublabel = frame_cache._labels(move)
            text = f"{label} {sublabel}".lower()
            assert not any(square in text for square in squares)

    def test_move_number_is_still_present(self, cache):
        frame_cache, record = cache
        for move in record.moves:
            label, _ = frame_cache._labels(move)
            assert str(move.move_no) in label

    def test_show_labels_false_gives_an_empty_panel(self, out_root):
        config = Config.from_mapping({
            "experiment": "chess",
            "run": {"output_root": str(out_root)},
            "params": {
                "durations": [{"label": "5s", "seconds": 5}],
                "show_labels": False,
            },
        })
        generator = ChessGenerator(config)
        record = play_random_game(seed=7, max_half_moves=10)
        frame_cache = _FrameCache(None, record, generator.params, show_labels=False)
        assert frame_cache._labels(record.moves[0]) == ("", "")

    def test_checks_do_not_annotate_the_label(self, cache):
        # `notation` carries a '+' on check; none of it reaches the panel.
        frame_cache, record = cache
        checking = [m for m in record.moves if m.is_check]
        for move in checking:
            label, sublabel = frame_cache._labels(move)
            assert "+" not in label and "+" not in sublabel
