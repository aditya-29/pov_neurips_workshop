"""Self-contained chess engine for synthetic game generation.

Ported from the original `chess_game_creator/chess.py`, with three changes:

* **No file I/O and no printing.** :func:`play_random_game` returns a
  :class:`GameRecord`; callers decide what to write. This is what makes the
  engine unit-testable.
* **Deterministic.** The RNG is injected, so a seed reproduces a game exactly.
* **Castling rights are cleared when a rook is captured on its home square.**
  The original only cleared them when a rook *moved*; if a home rook were
  captured and a different rook later landed on that square, castling would
  have been wrongly permitted.

Board representation: ``board[row][col]``, row 0 = rank 8, col 0 = file a.
White pieces are uppercase, black lowercase, empty squares are ``None``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator, NamedTuple, Sequence

FILES = "abcdefgh"

UNICODE = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}

PIECE_NAMES = {
    "P": "Pawn", "N": "Knight", "B": "Bishop",
    "R": "Rook", "Q": "Queen", "K": "King",
}

WHITE = "white"
BLACK = "black"

ROOK_DIRS = ((0, 1), (0, -1), (1, 0), (-1, 0))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
KNIGHT_DELTAS = ((-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1))

# Move flags
FLAG_NONE = None
FLAG_EN_PASSANT = "ep"
FLAG_CASTLE_KS = "castle_ks"
FLAG_CASTLE_QS = "castle_qs"


class Move(NamedTuple):
    """A move from (from_row, from_col) to (to_row, to_col)."""

    from_row: int
    from_col: int
    to_row: int
    to_col: int
    flag: str | None = None

    @property
    def src(self) -> str:
        return square_name(self.from_row, self.from_col)

    @property
    def dst(self) -> str:
        return square_name(self.to_row, self.to_col)

    def __str__(self) -> str:
        suffix = f"({self.flag})" if self.flag else ""
        return f"{self.src}->{self.dst}{suffix}"


def square_name(row: int, col: int) -> str:
    """(0, 0) → 'a8'."""
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError(f"square ({row}, {col}) is off the board")
    return f"{FILES[col]}{8 - row}"


def parse_square(name: str) -> tuple[int, int]:
    """'a8' → (0, 0)."""
    text = name.strip().lower()
    if len(text) != 2 or text[0] not in FILES or not text[1].isdigit():
        raise ValueError(f"invalid square {name!r}")
    rank = int(text[1])
    if not 1 <= rank <= 8:
        raise ValueError(f"invalid square {name!r}")
    return 8 - rank, FILES.index(text[0])


def color_of(piece: str | None) -> str | None:
    if piece is None:
        return None
    return WHITE if piece.isupper() else BLACK


def opponent(color: str) -> str:
    return BLACK if color == WHITE else WHITE


# ──────────────────────────────────────────────────────────────────────────────
# Game state
# ──────────────────────────────────────────────────────────────────────────────


class ChessGame:
    """Mutable chess position with full legal-move generation."""

    def __init__(self, board: list[list[str | None]] | None = None,
                 current_player: str = WHITE):
        self.board: list[list[str | None]] = board if board is not None else self._initial_board()
        self.current_player = current_player
        self.en_passant_target: tuple[int, int] | None = None
        self.castling_rights: dict[str, dict[str, bool]] = {
            WHITE: {"ks": True, "qs": True},
            BLACK: {"ks": True, "qs": True},
        }
        self.move_history: list[Move] = []
        self.full_move = 1

    @staticmethod
    def _initial_board() -> list[list[str | None]]:
        board: list[list[str | None]] = [[None] * 8 for _ in range(8)]
        board[0] = list("rnbqkbnr")
        board[1] = list("pppppppp")
        board[6] = list("PPPPPPPP")
        board[7] = list("RNBQKBNR")
        return board

    # -- inspection --------------------------------------------------------

    def piece_at(self, row: int, col: int) -> str | None:
        return self.board[row][col]

    def is_enemy(self, piece: str | None, color: str) -> bool:
        other = color_of(piece)
        return other is not None and other != color

    def find_king(self, color: str) -> tuple[int, int] | None:
        king = "K" if color == WHITE else "k"
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == king:
                    return row, col
        return None

    def copy(self) -> "ChessGame":
        clone = ChessGame(board=[row[:] for row in self.board], current_player=self.current_player)
        clone.en_passant_target = self.en_passant_target
        clone.castling_rights = {
            WHITE: dict(self.castling_rights[WHITE]),
            BLACK: dict(self.castling_rights[BLACK]),
        }
        clone.move_history = list(self.move_history)
        clone.full_move = self.full_move
        return clone

    # -- attack detection --------------------------------------------------

    def is_attacked(self, row: int, col: int, by: str) -> bool:
        """True if (row, col) is attacked by any piece of colour `by`."""
        board = self.board
        pawn = "P" if by == WHITE else "p"
        rook_queen = ("R", "Q") if by == WHITE else ("r", "q")
        bishop_queen = ("B", "Q") if by == WHITE else ("b", "q")
        knight = "N" if by == WHITE else "n"
        king = "K" if by == WHITE else "k"

        # Pawns: white pawns attack toward lower row indices.
        pawn_dir = 1 if by == WHITE else -1
        pawn_row = row + pawn_dir
        if 0 <= pawn_row < 8:
            for dcol in (-1, 1):
                pawn_col = col + dcol
                if 0 <= pawn_col < 8 and board[pawn_row][pawn_col] == pawn:
                    return True

        for drow, dcol in KNIGHT_DELTAS:
            r, c = row + drow, col + dcol
            if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == knight:
                return True

        for dirs, targets in ((ROOK_DIRS, rook_queen), (BISHOP_DIRS, bishop_queen)):
            for drow, dcol in dirs:
                r, c = row + drow, col + dcol
                while 0 <= r < 8 and 0 <= c < 8:
                    piece = board[r][c]
                    if piece is not None:
                        if piece in targets:
                            return True
                        break
                    r += drow
                    c += dcol

        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                if drow == 0 and dcol == 0:
                    continue
                r, c = row + drow, col + dcol
                if 0 <= r < 8 and 0 <= c < 8 and board[r][c] == king:
                    return True

        return False

    def in_check(self, color: str) -> bool:
        square = self.find_king(color)
        if square is None:
            return False
        return self.is_attacked(square[0], square[1], opponent(color))

    # -- pseudo-legal move generation -------------------------------------

    def _pawn_moves(self, row: int, col: int, color: str) -> Iterator[Move]:
        direction = -1 if color == WHITE else 1
        start_row = 6 if color == WHITE else 1
        next_row = row + direction

        if 0 <= next_row < 8 and self.board[next_row][col] is None:
            yield Move(row, col, next_row, col)
            double_row = row + 2 * direction
            if row == start_row and 0 <= double_row < 8 and self.board[double_row][col] is None:
                yield Move(row, col, double_row, col)

        if 0 <= next_row < 8:
            for dcol in (-1, 1):
                next_col = col + dcol
                if not 0 <= next_col < 8:
                    continue
                target = self.board[next_row][next_col]
                if target is not None and self.is_enemy(target, color):
                    yield Move(row, col, next_row, next_col)
                elif target is None and self.en_passant_target == (next_row, next_col):
                    yield Move(row, col, next_row, next_col, FLAG_EN_PASSANT)

    def _slider_moves(self, row: int, col: int, color: str,
                      directions: Sequence[tuple[int, int]]) -> Iterator[Move]:
        for drow, dcol in directions:
            r, c = row + drow, col + dcol
            while 0 <= r < 8 and 0 <= c < 8:
                target = self.board[r][c]
                if target is None:
                    yield Move(row, col, r, c)
                else:
                    if self.is_enemy(target, color):
                        yield Move(row, col, r, c)
                    break
                r += drow
                c += dcol

    def _knight_moves(self, row: int, col: int, color: str) -> Iterator[Move]:
        for drow, dcol in KNIGHT_DELTAS:
            r, c = row + drow, col + dcol
            if 0 <= r < 8 and 0 <= c < 8:
                target = self.board[r][c]
                if target is None or self.is_enemy(target, color):
                    yield Move(row, col, r, c)

    def _king_moves(self, row: int, col: int, color: str) -> Iterator[Move]:
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                if drow == 0 and dcol == 0:
                    continue
                r, c = row + drow, col + dcol
                if 0 <= r < 8 and 0 <= c < 8:
                    target = self.board[r][c]
                    if target is None or self.is_enemy(target, color):
                        yield Move(row, col, r, c)

        # Castling: emptiness and rook presence here; attacked squares in _is_legal.
        rights = self.castling_rights[color]
        back = 7 if color == WHITE else 0
        rook = "R" if color == WHITE else "r"
        if row == back and col == 4:
            if (rights["ks"]
                    and self.board[back][5] is None
                    and self.board[back][6] is None
                    and self.board[back][7] == rook):
                yield Move(row, col, back, 6, FLAG_CASTLE_KS)
            if (rights["qs"]
                    and self.board[back][3] is None
                    and self.board[back][2] is None
                    and self.board[back][1] is None
                    and self.board[back][0] == rook):
                yield Move(row, col, back, 2, FLAG_CASTLE_QS)

    def pseudo_legal_moves(self, color: str) -> list[Move]:
        moves: list[Move] = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if piece is None or color_of(piece) != color:
                    continue
                kind = piece.upper()
                if kind == "P":
                    moves.extend(self._pawn_moves(row, col, color))
                elif kind == "N":
                    moves.extend(self._knight_moves(row, col, color))
                elif kind == "R":
                    moves.extend(self._slider_moves(row, col, color, ROOK_DIRS))
                elif kind == "B":
                    moves.extend(self._slider_moves(row, col, color, BISHOP_DIRS))
                elif kind == "Q":
                    moves.extend(self._slider_moves(row, col, color, ROOK_DIRS + BISHOP_DIRS))
                elif kind == "K":
                    moves.extend(self._king_moves(row, col, color))
        return moves

    # -- legality ----------------------------------------------------------

    def _board_after(self, board: list[list[str | None]], move: Move,
                     color: str) -> list[list[str | None]]:
        new_board = [row[:] for row in board]
        piece = new_board[move.from_row][move.from_col]
        new_board[move.to_row][move.to_col] = piece
        new_board[move.from_row][move.from_col] = None

        if move.flag == FLAG_EN_PASSANT:
            # The captured pawn sits on the mover's originating rank.
            new_board[move.from_row][move.to_col] = None
        elif move.flag in (FLAG_CASTLE_KS, FLAG_CASTLE_QS):
            back = 7 if color == WHITE else 0
            rook = "R" if color == WHITE else "r"
            if move.flag == FLAG_CASTLE_KS:
                new_board[back][5] = rook
                new_board[back][7] = None
            else:
                new_board[back][3] = rook
                new_board[back][0] = None

        # Promotion is always to a queen.
        if piece == "P" and move.to_row == 0:
            new_board[move.to_row][move.to_col] = "Q"
        elif piece == "p" and move.to_row == 7:
            new_board[move.to_row][move.to_col] = "q"

        return new_board

    def _is_legal(self, move: Move, color: str) -> bool:
        back = 7 if color == WHITE else 0
        enemy = opponent(color)

        if move.flag == FLAG_CASTLE_KS:
            if any(self.is_attacked(back, col, enemy) for col in (4, 5, 6)):
                return False
        elif move.flag == FLAG_CASTLE_QS:
            if any(self.is_attacked(back, col, enemy) for col in (4, 3, 2)):
                return False

        original = self.board
        self.board = self._board_after(original, move, color)
        try:
            return not self.in_check(color)
        finally:
            self.board = original

    def legal_moves(self, color: str | None = None) -> list[Move]:
        color = color or self.current_player
        return [m for m in self.pseudo_legal_moves(color) if self._is_legal(m, color)]

    # -- notation ----------------------------------------------------------

    def to_notation(self, move: Move) -> str:
        """Algebraic notation for `move` **in the current position**.

        Must be called before :meth:`apply_move`.
        """
        piece = self.board[move.from_row][move.from_col]
        if piece is None:
            raise ValueError(f"no piece on {move.src}")

        if move.flag == FLAG_CASTLE_KS:
            return "O-O"
        if move.flag == FLAG_CASTLE_QS:
            return "O-O-O"

        kind = piece.upper()
        destination = move.dst
        is_capture = self.board[move.to_row][move.to_col] is not None or move.flag == FLAG_EN_PASSANT

        if kind == "P":
            notation = f"{FILES[move.from_col]}x{destination}" if is_capture else destination
        else:
            notation = f"{kind}{'x' if is_capture else ''}{destination}"

        if move.flag == FLAG_EN_PASSANT:
            notation += " e.p."

        promoted = self._board_after(self.board, move, color_of(piece))[move.to_row][move.to_col]
        if kind == "P" and promoted is not None and promoted.upper() == "Q":
            notation += "=Q"

        return notation

    # -- mutation ----------------------------------------------------------

    def apply_move(self, move: Move) -> None:
        """Apply a legal move, updating turn, castling rights and en passant."""
        piece = self.board[move.from_row][move.from_col]
        if piece is None:
            raise ValueError(f"no piece on {move.src}")
        color = color_of(piece)
        if color != self.current_player:
            raise ValueError(
                f"it is {self.current_player}'s turn but {move.src} holds a {color} piece"
            )
        destination = self.board[move.to_row][move.to_col]
        if destination is not None and not self.is_enemy(destination, color):
            raise ValueError(f"{move.dst} holds a friendly piece")

        captured = destination

        # En passant target is only live for the single reply.
        self.en_passant_target = None
        if piece.upper() == "P" and abs(move.to_row - move.from_row) == 2:
            self.en_passant_target = ((move.from_row + move.to_row) // 2, move.from_col)

        self.board = self._board_after(self.board, move, color)

        if self.in_check(color):
            raise ValueError(f"illegal move {move}: leaves {color} king in check")

        self._update_castling_rights(piece, move, captured)

        self.move_history.append(move)
        self.current_player = opponent(color)
        if color == BLACK:
            self.full_move += 1

    def _update_castling_rights(self, piece: str, move: Move, captured: str | None) -> None:
        # Moving the king or a rook forfeits rights.
        if piece == "K":
            self.castling_rights[WHITE] = {"ks": False, "qs": False}
        elif piece == "k":
            self.castling_rights[BLACK] = {"ks": False, "qs": False}
        elif piece == "R":
            if (move.from_row, move.from_col) == (7, 7):
                self.castling_rights[WHITE]["ks"] = False
            elif (move.from_row, move.from_col) == (7, 0):
                self.castling_rights[WHITE]["qs"] = False
        elif piece == "r":
            if (move.from_row, move.from_col) == (0, 7):
                self.castling_rights[BLACK]["ks"] = False
            elif (move.from_row, move.from_col) == (0, 0):
                self.castling_rights[BLACK]["qs"] = False

        # Capturing a rook on its home square forfeits that side's right.
        if captured is not None and captured.upper() == "R":
            target = (move.to_row, move.to_col)
            if target == (7, 7):
                self.castling_rights[WHITE]["ks"] = False
            elif target == (7, 0):
                self.castling_rights[WHITE]["qs"] = False
            elif target == (0, 7):
                self.castling_rights[BLACK]["ks"] = False
            elif target == (0, 0):
                self.castling_rights[BLACK]["qs"] = False

    # -- rendering ---------------------------------------------------------

    def board_lines(self, use_unicode: bool = False) -> list[str]:
        lines = ["    a  b  c  d  e  f  g  h", "  ╔" + "══╤" * 7 + "══╗"]
        for row in range(8):
            rank = 8 - row
            cells = []
            for col in range(8):
                piece = self.board[row][col]
                if piece:
                    cells.append(f"{UNICODE[piece] if use_unicode else piece} ")
                else:
                    cells.append(". " if (row + col) % 2 == 0 else "  ")
            lines.append(f"{rank} ║{'│'.join(cells)}║ {rank}")
            if row < 7:
                lines.append("  ╠" + "══╪" * 7 + "══╣")
        lines.append("  ╚" + "══╧" * 7 + "══╝")
        lines.append("    a  b  c  d  e  f  g  h")
        return lines

    def board_to_str(self, use_unicode: bool = False) -> str:
        return "\n".join(self.board_lines(use_unicode))

    def board_snapshot(self) -> tuple[tuple[str | None, ...], ...]:
        """Immutable copy of the board, safe to keep in a move record."""
        return tuple(tuple(row) for row in self.board)


# ──────────────────────────────────────────────────────────────────────────────
# Game records
# ──────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MoveRecord:
    """One half-move plus the position it produced."""

    ply: int                 # 1-based half-move index
    move_no: int             # full-move number as printed in the transcript
    color: str               # 'White' | 'Black'
    src: str                 # e.g. 'b2'
    dst: str                 # e.g. 'b3'
    piece: str               # 'Pawn', 'Knight', …
    notation: str            # algebraic, with '+' when it gives check
    is_capture: bool
    is_check: bool
    flag: str | None
    board_after: tuple[tuple[str | None, ...], ...]

    @property
    def from_square(self) -> tuple[int, int]:
        return parse_square(self.src)

    @property
    def to_square(self) -> tuple[int, int]:
        return parse_square(self.dst)


@dataclass
class GameRecord:
    """A finished synthetic game."""

    game_index: int
    seed: int
    moves: list[MoveRecord] = field(default_factory=list)
    result: str = ""

    def __len__(self) -> int:
        return len(self.moves)

    @property
    def n_half_moves(self) -> int:
        return len(self.moves)

    def truncated(self, n_half_moves: int) -> "GameRecord":
        """A prefix of this game, with a result note describing the cut."""
        if n_half_moves >= len(self.moves):
            return self
        return GameRecord(
            game_index=self.game_index,
            seed=self.seed,
            moves=self.moves[:n_half_moves],
            result=f"Clip ends at move {n_half_moves}",
        )

    def transcript(self) -> str:
        """Ground-truth transcript, byte-compatible with the original format."""
        lines = [
            f"Game #{self.game_index} — Move Transcription",
            "=" * 46,
            "",
            f"{'Move':<6}  {'Turn':<8}  {'Source':<8}  {'Destination'}",
            f"{'----':<6}  {'----':<8}  {'------':<8}  {'-----------'}",
        ]
        for record in self.moves:
            lines.append(
                f"{record.move_no:<6}  {record.color:<8}  {record.src:<8}  {record.dst}"
            )
        lines.append("")
        lines.append(f"Result: {self.result}")
        lines.append(f"Total half-moves: {len(self.moves)}")
        return "\n".join(lines) + "\n"

    def compact_moves(self) -> str:
        """One-line ground truth for the manifest cell."""
        return "; ".join(
            f"{r.move_no} {r.color} {r.src} {r.dst}" for r in self.moves
        )


def play_random_game(
    seed: int,
    max_half_moves: int = 200,
    game_index: int = 0,
    rng: random.Random | None = None,
) -> GameRecord:
    """Play one game of uniformly random legal moves.

    The same `seed` always produces the same game.
    """
    if max_half_moves < 1:
        raise ValueError(f"max_half_moves must be >= 1, got {max_half_moves}")

    rng = rng or random.Random(seed)
    game = ChessGame()
    record = GameRecord(game_index=game_index, seed=seed)

    while len(record.moves) < max_half_moves:
        color = game.current_player
        moves = game.legal_moves(color)

        if not moves:
            if game.in_check(color):
                winner = "Black" if color == WHITE else "White"
                record.result = f"CHECKMATE — {winner} wins!"
            else:
                record.result = "STALEMATE — Draw!"
            return record

        move = rng.choice(moves)
        notation = game.to_notation(move)
        move_no = game.full_move
        piece = game.board[move.from_row][move.from_col]
        piece_name = PIECE_NAMES[piece.upper()]
        is_capture = (
            game.board[move.to_row][move.to_col] is not None
            or move.flag == FLAG_EN_PASSANT
        )

        game.apply_move(move)

        gives_check = game.in_check(game.current_player)
        record.moves.append(
            MoveRecord(
                ply=len(record.moves) + 1,
                move_no=move_no,
                color="White" if color == WHITE else "Black",
                src=move.src,
                dst=move.dst,
                piece=piece_name,
                notation=notation + ("+" if gives_check else ""),
                is_capture=is_capture,
                is_check=gives_check,
                flag=move.flag,
                board_after=game.board_snapshot(),
            )
        )

    record.result = f"Move limit reached ({max_half_moves} half-moves) — Draw."
    return record
