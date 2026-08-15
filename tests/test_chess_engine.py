"""Chess engine correctness.

The perft tests are the backbone: they compare the move generator's node counts
against the published values for the initial position, which catches essentially
any legality bug (missed pins, bad castling, wrong pawn rules).
"""

from __future__ import annotations

import pytest

from pov.experiments.chess.engine import (
    BLACK,
    WHITE,
    ChessGame,
    Move,
    color_of,
    opponent,
    parse_square,
    play_random_game,
    square_name,
)


def empty_board() -> list[list[str | None]]:
    return [[None] * 8 for _ in range(8)]


def place(pieces: dict[str, str]) -> list[list[str | None]]:
    """Build a board from {'e1': 'K', 'e8': 'k'}."""
    board = empty_board()
    for square, piece in pieces.items():
        row, col = parse_square(square)
        board[row][col] = piece
    return board


def perft(game: ChessGame, depth: int) -> int:
    if depth == 0:
        return 1
    total = 0
    for move in game.legal_moves():
        child = game.copy()
        child.apply_move(move)
        total += perft(child, depth - 1)
    return total


def move_between(game: ChessGame, src: str, dst: str) -> Move:
    """The legal move from `src` to `dst`, or fail the test."""
    from_row, from_col = parse_square(src)
    to_row, to_col = parse_square(dst)
    for move in game.legal_moves():
        if (move.from_row, move.from_col, move.to_row, move.to_col) == (
            from_row, from_col, to_row, to_col
        ):
            return move
    raise AssertionError(f"{src}->{dst} is not legal here")


# ── Squares ───────────────────────────────────────────────────────────────────


class TestSquares:
    @pytest.mark.parametrize(
        "row,col,name",
        [(0, 0, "a8"), (7, 0, "a1"), (7, 7, "h1"), (0, 7, "h8"), (4, 4, "e4")],
    )
    def test_square_name_and_parse_round_trip(self, row, col, name):
        assert square_name(row, col) == name
        assert parse_square(name) == (row, col)

    @pytest.mark.parametrize("bad", ["i1", "a9", "a0", "", "e", "e11"])
    def test_parse_rejects_invalid(self, bad):
        with pytest.raises(ValueError):
            parse_square(bad)

    def test_square_name_rejects_off_board(self):
        with pytest.raises(ValueError):
            square_name(8, 0)

    def test_colour_helpers(self):
        assert color_of("K") == WHITE and color_of("k") == BLACK
        assert color_of(None) is None
        assert opponent(WHITE) == BLACK and opponent(BLACK) == WHITE


# ── Initial position and perft ────────────────────────────────────────────────


class TestInitialPosition:
    def test_board_setup(self):
        game = ChessGame()
        assert "".join(game.board[0]) == "rnbqkbnr"
        assert "".join(game.board[1]) == "pppppppp"
        assert all(square is None for row in game.board[2:6] for square in row)
        assert "".join(game.board[6]) == "PPPPPPPP"
        assert "".join(game.board[7]) == "RNBQKBNR"

    def test_white_moves_first(self):
        assert ChessGame().current_player == WHITE

    def test_neither_side_is_in_check(self):
        game = ChessGame()
        assert not game.in_check(WHITE)
        assert not game.in_check(BLACK)

    @pytest.mark.parametrize("depth,expected", [(1, 20), (2, 400), (3, 8902)])
    def test_perft(self, depth, expected):
        assert perft(ChessGame(), depth) == expected

    @pytest.mark.slow
    def test_perft_4(self):
        assert perft(ChessGame(), 4) == 197281


# ── Check, checkmate, stalemate ───────────────────────────────────────────────


class TestCheckDetection:
    def test_fools_mate_is_checkmate(self):
        game = ChessGame()
        for src, dst in [("f2", "f3"), ("e7", "e5"), ("g2", "g4"), ("d8", "h4")]:
            game.apply_move(move_between(game, src, dst))
        assert game.in_check(WHITE)
        assert game.legal_moves(WHITE) == []

    def test_stalemate_has_no_moves_and_no_check(self):
        # Black king a8, white queen c7, white king a6 — classic stalemate.
        game = ChessGame(board=place({"a8": "k", "c7": "Q", "a6": "K"}),
                         current_player=BLACK)
        assert not game.in_check(BLACK)
        assert game.legal_moves(BLACK) == []

    def test_king_cannot_move_into_check(self):
        game = ChessGame(board=place({"e1": "K", "e8": "k", "d8": "r"}),
                         current_player=WHITE)
        destinations = {move.dst for move in game.legal_moves(WHITE)}
        assert "d1" not in destinations  # d-file is covered by the rook
        assert "e2" in destinations

    def test_pinned_piece_cannot_abandon_the_king(self):
        # White knight on e2 is pinned along the e-file by the black rook.
        game = ChessGame(board=place({"e1": "K", "e2": "N", "e8": "r", "a1": "R"}),
                         current_player=WHITE)
        moved_knight = [m for m in game.legal_moves(WHITE) if m.src == "e2"]
        assert moved_knight == []

    def test_must_address_check(self):
        game = ChessGame(board=place({"e1": "K", "e8": "r", "a1": "R", "h1": "B"}),
                         current_player=WHITE)
        # Every legal reply must leave the king safe.
        for move in game.legal_moves(WHITE):
            child = game.copy()
            child.apply_move(move)
            assert not child.in_check(WHITE)

    def test_is_attacked_by_each_piece_type(self):
        cases = [
            ({"d4": "R"}, "d8", True), ({"d4": "R"}, "e5", False),
            ({"d4": "B"}, "g7", True), ({"d4": "B"}, "d5", False),
            ({"d4": "Q"}, "h8", True), ({"d4": "N"}, "e6", True),
            ({"d4": "K"}, "d5", True), ({"d4": "K"}, "d6", False),
            ({"d4": "P"}, "e5", True), ({"d4": "P"}, "e3", False),
        ]
        for pieces, target, expected in cases:
            game = ChessGame(board=place(pieces))
            row, col = parse_square(target)
            assert game.is_attacked(row, col, WHITE) is expected, (pieces, target)

    def test_black_pawn_attacks_downward(self):
        game = ChessGame(board=place({"d5": "p"}))
        row, col = parse_square("e4")
        assert game.is_attacked(row, col, BLACK) is True
        row, col = parse_square("e6")
        assert game.is_attacked(row, col, BLACK) is False

    def test_slider_is_blocked(self):
        game = ChessGame(board=place({"a1": "R", "a4": "P"}))
        row, col = parse_square("a8")
        assert game.is_attacked(row, col, WHITE) is False


# ── Castling ──────────────────────────────────────────────────────────────────


class TestCastling:
    def base(self) -> ChessGame:
        return ChessGame(board=place({"e1": "K", "a1": "R", "h1": "R", "e8": "k"}),
                         current_player=WHITE)

    def test_both_sides_available(self):
        flags = {move.flag for move in self.base().legal_moves(WHITE)}
        assert "castle_ks" in flags and "castle_qs" in flags

    def test_kingside_castle_moves_both_pieces(self):
        game = self.base()
        game.apply_move(move_between(game, "e1", "g1"))
        assert game.board[7][6] == "K" and game.board[7][5] == "R"
        assert game.board[7][4] is None and game.board[7][7] is None

    def test_queenside_castle_moves_both_pieces(self):
        game = self.base()
        game.apply_move(move_between(game, "e1", "c1"))
        assert game.board[7][2] == "K" and game.board[7][3] == "R"
        assert game.board[7][0] is None

    def test_blocked_by_own_piece(self):
        game = ChessGame(board=place({"e1": "K", "h1": "R", "f1": "B", "e8": "k"}),
                         current_player=WHITE)
        assert not any(m.flag == "castle_ks" for m in game.legal_moves(WHITE))

    def test_cannot_castle_out_of_check(self):
        game = ChessGame(board=place({"e1": "K", "h1": "R", "e8": "r"}),
                         current_player=WHITE)
        assert not any(m.flag for m in game.legal_moves(WHITE) if m.flag)

    def test_cannot_castle_through_attacked_square(self):
        game = ChessGame(board=place({"e1": "K", "h1": "R", "f8": "r", "e8": "k"}),
                         current_player=WHITE)
        assert not any(m.flag == "castle_ks" for m in game.legal_moves(WHITE))

    def test_cannot_castle_into_attacked_square(self):
        game = ChessGame(board=place({"e1": "K", "h1": "R", "g8": "r", "e8": "k"}),
                         current_player=WHITE)
        assert not any(m.flag == "castle_ks" for m in game.legal_moves(WHITE))

    def test_queenside_allowed_when_only_b1_is_attacked(self):
        # The king never visits b1, so an attack there does not prevent O-O-O.
        game = ChessGame(board=place({"e1": "K", "a1": "R", "b8": "r", "e8": "k"}),
                         current_player=WHITE)
        assert any(m.flag == "castle_qs" for m in game.legal_moves(WHITE))

    def test_king_move_forfeits_both_rights(self):
        game = self.base()
        game.apply_move(move_between(game, "e1", "e2"))
        assert game.castling_rights[WHITE] == {"ks": False, "qs": False}

    def test_rook_move_forfeits_that_side(self):
        game = self.base()
        game.apply_move(move_between(game, "h1", "h2"))
        assert game.castling_rights[WHITE]["ks"] is False
        assert game.castling_rights[WHITE]["qs"] is True

    def test_capturing_a_home_rook_forfeits_that_right(self):
        # Regression: the original engine only cleared rights when a rook MOVED,
        # so a captured rook left the right set.
        game = ChessGame(
            board=place({"e1": "K", "h1": "R", "e8": "k", "h8": "r", "a8": "r"}),
            current_player=BLACK,
        )
        assert game.castling_rights[WHITE]["ks"] is True
        game.apply_move(move_between(game, "h8", "h1"))
        assert game.castling_rights[WHITE]["ks"] is False

    def test_castling_notation(self):
        game = self.base()
        assert game.to_notation(move_between(game, "e1", "g1")) == "O-O"
        assert game.to_notation(move_between(game, "e1", "c1")) == "O-O-O"

    def test_black_castles_on_its_own_rank(self):
        game = ChessGame(board=place({"e8": "k", "h8": "r", "e1": "K"}),
                         current_player=BLACK)
        game.apply_move(move_between(game, "e8", "g8"))
        assert game.board[0][6] == "k" and game.board[0][5] == "r"


# ── Pawns ─────────────────────────────────────────────────────────────────────


class TestPawns:
    def test_single_and_double_push(self):
        game = ChessGame()
        destinations = {m.dst for m in game.legal_moves(WHITE) if m.src == "e2"}
        assert destinations == {"e3", "e4"}

    def test_double_push_blocked(self):
        game = ChessGame(board=place({"e2": "P", "e3": "n", "e1": "K", "e8": "k"}),
                         current_player=WHITE)
        assert {m.dst for m in game.legal_moves(WHITE) if m.src == "e2"} == set()

    def test_pawn_cannot_capture_forward(self):
        game = ChessGame(board=place({"e4": "P", "e5": "p", "e1": "K", "e8": "k"}),
                         current_player=WHITE)
        assert {m.dst for m in game.legal_moves(WHITE) if m.src == "e4"} == set()

    def test_diagonal_capture(self):
        game = ChessGame(board=place({"e4": "P", "d5": "p", "e1": "K", "e8": "k"}),
                         current_player=WHITE)
        assert "d5" in {m.dst for m in game.legal_moves(WHITE) if m.src == "e4"}

    def test_en_passant_target_is_set_then_cleared(self):
        game = ChessGame()
        game.apply_move(move_between(game, "e2", "e4"))
        assert game.en_passant_target == parse_square("e3")
        game.apply_move(move_between(game, "e7", "e6"))
        assert game.en_passant_target is None

    def test_en_passant_capture_removes_the_right_pawn(self):
        game = ChessGame(
            board=place({"e5": "P", "d7": "p", "e1": "K", "e8": "k"}),
            current_player=BLACK,
        )
        game.apply_move(move_between(game, "d7", "d5"))  # double push past e5
        capture = [m for m in game.legal_moves(WHITE) if m.flag == "ep"]
        assert len(capture) == 1
        game.apply_move(capture[0])
        assert game.board[parse_square("d6")[0]][parse_square("d6")[1]] == "P"
        row, col = parse_square("d5")
        assert game.board[row][col] is None  # the captured pawn is gone

    def test_en_passant_notation(self):
        game = ChessGame(
            board=place({"e5": "P", "d7": "p", "e1": "K", "e8": "k"}),
            current_player=BLACK,
        )
        game.apply_move(move_between(game, "d7", "d5"))
        capture = [m for m in game.legal_moves(WHITE) if m.flag == "ep"][0]
        assert "e.p." in game.to_notation(capture)

    def test_promotion_to_queen(self):
        game = ChessGame(board=place({"a7": "P", "e1": "K", "e8": "k"}),
                         current_player=WHITE)
        move = move_between(game, "a7", "a8")
        assert game.to_notation(move).endswith("=Q")
        game.apply_move(move)
        assert game.board[0][0] == "Q"

    def test_black_promotion(self):
        game = ChessGame(board=place({"h2": "p", "e1": "K", "e8": "k"}),
                         current_player=BLACK)
        game.apply_move(move_between(game, "h2", "h1"))
        assert game.board[7][7] == "q"


# ── Notation and state ────────────────────────────────────────────────────────


class TestNotationAndState:
    def test_pawn_push_notation_is_bare_square(self):
        game = ChessGame()
        assert game.to_notation(move_between(game, "e2", "e4")) == "e4"

    def test_piece_and_capture_notation(self):
        game = ChessGame(board=place({"b1": "N", "c3": "p", "e1": "K", "e8": "k"}),
                         current_player=WHITE)
        assert game.to_notation(move_between(game, "b1", "c3")) == "Nxc3"

    def test_pawn_capture_notation_includes_file(self):
        game = ChessGame(board=place({"e4": "P", "d5": "p", "e1": "K", "e8": "k"}),
                         current_player=WHITE)
        assert game.to_notation(move_between(game, "e4", "d5")) == "exd5"

    def test_turn_alternates_and_move_number_increments(self):
        game = ChessGame()
        assert game.full_move == 1
        game.apply_move(move_between(game, "e2", "e4"))
        assert game.current_player == BLACK and game.full_move == 1
        game.apply_move(move_between(game, "e7", "e5"))
        assert game.current_player == WHITE and game.full_move == 2

    def test_apply_move_rejects_wrong_turn(self):
        game = ChessGame()
        black_move = Move(*parse_square("e7"), *parse_square("e5"))
        with pytest.raises(ValueError, match="turn"):
            game.apply_move(black_move)

    def test_apply_move_rejects_empty_source(self):
        game = ChessGame()
        with pytest.raises(ValueError, match="no piece"):
            game.apply_move(Move(*parse_square("e4"), *parse_square("e5")))

    def test_apply_move_rejects_self_capture(self):
        game = ChessGame()
        with pytest.raises(ValueError, match="friendly"):
            game.apply_move(Move(*parse_square("a1"), *parse_square("a2")))

    def test_copy_is_independent(self):
        game = ChessGame()
        clone = game.copy()
        clone.apply_move(move_between(clone, "e2", "e4"))
        assert game.board[6][4] == "P"
        assert game.current_player == WHITE
        assert game.castling_rights is not clone.castling_rights

    def test_board_snapshot_is_immutable(self):
        snapshot = ChessGame().board_snapshot()
        assert isinstance(snapshot, tuple) and isinstance(snapshot[0], tuple)

    def test_board_to_str_has_all_ranks(self):
        text = ChessGame().board_to_str()
        for rank in "12345678":
            assert rank in text
        assert "a  b  c  d  e  f  g  h" in text


# ── Random games ──────────────────────────────────────────────────────────────


class TestPlayRandomGame:
    def test_same_seed_gives_identical_games(self):
        a = play_random_game(seed=99, max_half_moves=60)
        b = play_random_game(seed=99, max_half_moves=60)
        assert [m.notation for m in a.moves] == [m.notation for m in b.moves]
        assert a.result == b.result

    def test_different_seeds_diverge(self):
        a = play_random_game(seed=1, max_half_moves=60)
        b = play_random_game(seed=2, max_half_moves=60)
        assert [m.src for m in a.moves] != [m.src for m in b.moves]

    def test_respects_move_limit(self):
        record = play_random_game(seed=5, max_half_moves=10)
        assert len(record.moves) <= 10

    def test_records_a_result(self):
        record = play_random_game(seed=5, max_half_moves=10)
        assert record.result

    def test_rejects_zero_length(self):
        with pytest.raises(ValueError, match="max_half_moves"):
            play_random_game(seed=1, max_half_moves=0)

    def test_every_move_is_legal_and_consistent(self):
        record = play_random_game(seed=3, max_half_moves=80)
        replay = ChessGame()
        for move_record in record.moves:
            legal = {(m.src, m.dst) for m in replay.legal_moves()}
            assert (move_record.src, move_record.dst) in legal
            replay.apply_move(move_between(replay, move_record.src, move_record.dst))
            assert replay.board_snapshot() == move_record.board_after

    def test_colours_alternate(self):
        record = play_random_game(seed=11, max_half_moves=40)
        colours = [m.color for m in record.moves]
        assert colours == ["White" if i % 2 == 0 else "Black" for i in range(len(colours))]

    def test_ply_numbering_is_sequential(self):
        record = play_random_game(seed=12, max_half_moves=30)
        assert [m.ply for m in record.moves] == list(range(1, len(record.moves) + 1))

    def test_check_flag_matches_position(self):
        record = play_random_game(seed=21, max_half_moves=120)
        for move_record in record.moves:
            if move_record.is_check:
                assert move_record.notation.endswith("+")

    def test_transcript_format(self):
        record = play_random_game(seed=4, max_half_moves=6, game_index=3)
        text = record.transcript()
        assert "Game #3" in text
        assert "Move    Turn      Source    Destination" in text
        assert f"Total half-moves: {len(record.moves)}" in text
        for move_record in record.moves:
            assert move_record.src in text and move_record.dst in text

    def test_compact_moves_round_trips_through_the_parser(self):
        from pov.eval.chess import parse_transcript

        record = play_random_game(seed=8, max_half_moves=12)
        parsed = parse_transcript(record.compact_moves())
        assert len(parsed) == len(record.moves)
        assert parsed[0]["from"] == record.moves[0].src

    def test_transcript_round_trips_through_the_parser(self):
        from pov.eval.chess import parse_transcript

        record = play_random_game(seed=8, max_half_moves=12)
        parsed = parse_transcript(record.transcript())
        assert [(p["from"], p["to"]) for p in parsed] == [
            (m.src, m.dst) for m in record.moves
        ]

    def test_truncated_prefix(self):
        record = play_random_game(seed=6, max_half_moves=40)
        clipped = record.truncated(5)
        assert len(clipped.moves) == 5
        assert clipped.moves == record.moves[:5]
        assert "Clip ends at move 5" in clipped.result

    def test_truncate_beyond_length_returns_same_record(self):
        record = play_random_game(seed=6, max_half_moves=10)
        assert record.truncated(999) is record
