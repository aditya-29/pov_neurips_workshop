"""Chess board frame rendering.

Speed notes
-----------
* The empty board (squares, border, file/rank labels) is rasterised **once**
  and reused as a numpy buffer; each frame starts as a copy of it.
* Piece art is rasterised once per piece type into an RGBA tile and
  alpha-composited with numpy slicing — no per-frame text or SVG work.
* Only positions that actually differ produce a frame. Holding a position for
  a second costs one render, because :func:`pov.video.write_timeline` expands
  the hold inside ffmpeg.

Piece art falls back gracefully: python-chess SVGs (if `cairosvg` is
installed) → a Unicode chess glyph from a system font → a lettered disc.
Every path produces the same tile size, so the renderer does not care.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pov.experiments.chess.engine import FILES

Board = Sequence[Sequence[str | None]]

# Fonts that carry the Unicode chess glyphs (U+2654–U+265F).
_GLYPH_FONTS = (
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
)

# Fonts for the label panel and board coordinates.
_TEXT_FONTS = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
)

# U+2654\u20132659 are the *outline* (white) pieces, U+265A\u2013265F the *solid* (black)
# ones. Using the correct codepoint per colour is what makes the two sides
# distinguishable \u2014 colouring a single glyph shape differently is not enough
# once the board is downscaled for a model's vision encoder.
_GLYPHS_WHITE = {
    "K": "\u2654", "Q": "\u2655", "R": "\u2656",
    "B": "\u2657", "N": "\u2658", "P": "\u2659",
}
_GLYPHS_BLACK = {
    "K": "\u265a", "Q": "\u265b", "R": "\u265c",
    "B": "\u265d", "N": "\u265e", "P": "\u265f",
}


@dataclass(frozen=True)
class BoardTheme:
    """Colours and geometry. Frame dimensions are derived and always even."""

    square: int = 48
    padding: int = 16
    panel: int = 64

    light: tuple[int, int, int] = (240, 217, 181)
    dark: tuple[int, int, int] = (181, 136, 99)
    highlight_from: tuple[int, int, int] = (247, 247, 105)
    highlight_to: tuple[int, int, int] = (172, 195, 51)
    background: tuple[int, int, int] = (22, 22, 22)
    panel_bg: tuple[int, int, int] = (30, 30, 30)
    label: tuple[int, int, int] = (150, 150, 150)
    info: tuple[int, int, int] = (195, 195, 195)
    check: tuple[int, int, int] = (220, 70, 70)

    white_fill: tuple[int, int, int] = (238, 238, 218)
    white_text: tuple[int, int, int] = (38, 28, 8)
    black_fill: tuple[int, int, int] = (52, 52, 48)
    black_text: tuple[int, int, int] = (222, 222, 212)
    outline: tuple[int, int, int] = (5, 5, 5)

    @property
    def board_px(self) -> int:
        return self.square * 8

    @property
    def width(self) -> int:
        return self.padding * 2 + self.board_px

    @property
    def height(self) -> int:
        return self.padding * 2 + self.board_px + self.panel

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    def validate(self) -> "BoardTheme":
        if self.square < 8:
            raise ValueError(f"theme.square must be >= 8, got {self.square}")
        if self.width % 2 or self.height % 2:
            raise ValueError(
                f"theme produces an odd frame size {self.width}x{self.height}; "
                "adjust square/padding/panel so both are even"
            )
        return self


def _load_font(paths: Sequence[str], size: int) -> ImageFont.FreeTypeFont:
    for path in paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 9.2
        return ImageFont.load_default()


@lru_cache(maxsize=8)
def _text_font(size: int) -> ImageFont.FreeTypeFont:
    return _load_font(_TEXT_FONTS, size)


def _glyph_font(size: int) -> ImageFont.FreeTypeFont | None:
    """A font that actually contains the chess glyphs, or None."""
    for path in _GLYPH_FONTS:
        if not os.path.exists(path):
            continue
        try:
            font = ImageFont.truetype(path, size)
        except OSError:
            continue
        try:
            # Both colour ranges must be present, not just the white one.
            masks = [font.getmask("\u2654"), font.getmask("\u265a")]
        except Exception:
            continue
        if all(mask.getbbox() is not None for mask in masks):
            return font
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Piece art
# ──────────────────────────────────────────────────────────────────────────────


class PieceArt:
    """RGBA tiles for the twelve pieces, rasterised once."""

    def __init__(self, theme: BoardTheme, use_svg: bool = True):
        self.theme = theme
        self.use_svg = use_svg
        self._tiles: dict[str, np.ndarray] = {}
        self.source: str = "unloaded"

    def load(self) -> "PieceArt":
        if self._tiles:
            return self
        if self.use_svg:
            tiles = self._load_svg()
            if tiles:
                self._tiles, self.source = tiles, "svg"
                return self
        glyph = _glyph_font(int(self.theme.square * 0.86))
        if glyph is not None:
            self._tiles, self.source = self._load_glyph(glyph), "glyph"
            return self
        self._tiles, self.source = self._load_disc(), "disc"
        return self

    def tile(self, piece: str) -> np.ndarray:
        if not self._tiles:
            self.load()
        try:
            return self._tiles[piece]
        except KeyError:
            raise KeyError(f"unknown piece symbol {piece!r}") from None

    # -- rasterisers -------------------------------------------------------

    def _load_svg(self) -> dict[str, np.ndarray] | None:
        try:
            import cairosvg  # noqa: F401
            import chess
            import chess.svg
        except Exception:
            return None

        size = self.theme.square
        tiles: dict[str, np.ndarray] = {}
        try:
            for symbol in "KQRBNPkqrbnp":
                svg = chess.svg.piece(chess.Piece.from_symbol(symbol), size=size)
                png = cairosvg.svg2png(
                    bytestring=svg.encode("utf-8"),
                    output_width=size,
                    output_height=size,
                )
                image = Image.open(__import__("io").BytesIO(png)).convert("RGBA")
                tiles[symbol] = np.asarray(image)
        except Exception:
            return None
        return tiles

    def _load_glyph(self, font: ImageFont.FreeTypeFont) -> dict[str, np.ndarray]:
        theme, size = self.theme, self.theme.square
        tiles: dict[str, np.ndarray] = {}
        for symbol in "KQRBNPkqrbnp":
            kind = symbol.upper()
            is_white = symbol.isupper()
            image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            glyph = (_GLYPHS_WHITE if is_white else _GLYPHS_BLACK)[kind]
            box = draw.textbbox((0, 0), glyph, font=font)
            x = (size - (box[2] - box[0])) / 2 - box[0]
            y = (size - (box[3] - box[1])) / 2 - box[1]
            fill = theme.white_fill if is_white else theme.black_fill
            # Outline in the opposite tone so either colour stays legible on
            # either square colour.
            halo = theme.outline if is_white else theme.white_fill
            for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                           (0, 1), (1, -1), (1, 0), (1, 1)):
                draw.text((x + dx, y + dy), glyph, font=font, fill=(*halo, 255))
            draw.text((x, y), glyph, font=font, fill=(*fill, 255))
            tiles[symbol] = np.asarray(image)
        return tiles

    def _load_disc(self) -> dict[str, np.ndarray]:
        theme, size = self.theme, self.theme.square
        font = _text_font(int(size * 0.5))
        margin = max(2, size // 10)
        tiles: dict[str, np.ndarray] = {}
        for symbol in "KQRBNPkqrbnp":
            is_white = symbol.isupper()
            fill = theme.white_fill if is_white else theme.black_fill
            text_color = theme.white_text if is_white else theme.black_text
            image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            draw.ellipse(
                (margin, margin, size - margin - 1, size - margin - 1),
                fill=(*fill, 255),
                outline=(*theme.outline, 255),
                width=2,
            )
            letter = symbol.upper()
            box = draw.textbbox((0, 0), letter, font=font)
            x = (size - (box[2] - box[0])) / 2 - box[0]
            y = (size - (box[3] - box[1])) / 2 - box[1]
            draw.text((x, y), letter, font=font, fill=(*text_color, 255))
            tiles[symbol] = np.asarray(image)
        return tiles


def composite(frame: np.ndarray, tile: np.ndarray, x: float, y: float) -> None:
    """Alpha-composite an RGBA tile onto an RGB frame in place, with clipping."""
    ix, iy = int(round(x)), int(round(y))
    fh, fw = frame.shape[:2]
    th, tw = tile.shape[:2]

    sx0, sy0 = max(0, -ix), max(0, -iy)
    sx1, sy1 = min(tw, fw - ix), min(th, fh - iy)
    if sx0 >= sx1 or sy0 >= sy1:
        return

    dx0, dy0 = ix + sx0, iy + sy0
    src = tile[sy0:sy1, sx0:sx1]
    dst = frame[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)]

    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    blended = src[:, :, :3].astype(np.float32) * alpha + dst.astype(np.float32) * (1.0 - alpha)
    dst[:] = blended.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Board renderer
# ──────────────────────────────────────────────────────────────────────────────


class BoardRenderer:
    """Renders board positions to RGB numpy frames."""

    def __init__(self, theme: BoardTheme | None = None, use_svg: bool = True):
        self.theme = (theme or BoardTheme()).validate()
        self.art = PieceArt(self.theme, use_svg=use_svg)
        self._background: np.ndarray | None = None

    @property
    def size(self) -> tuple[int, int]:
        return self.theme.size

    def prepare(self) -> "BoardRenderer":
        """Rasterise piece art and the board background up front.

        Worth calling once before a parallel render: the caches are then
        read-only and safe to share across threads.
        """
        self.art.load()
        self._ensure_background()
        return self

    # -- background --------------------------------------------------------

    def _ensure_background(self) -> np.ndarray:
        if self._background is None:
            self._background = self._build_background()
        return self._background

    def _build_background(self) -> np.ndarray:
        theme = self.theme
        image = Image.new("RGB", theme.size, theme.background)
        draw = ImageDraw.Draw(image)

        for row in range(8):
            for col in range(8):
                x0 = theme.padding + col * theme.square
                y0 = theme.padding + row * theme.square
                color = theme.light if (row + col) % 2 == 0 else theme.dark
                draw.rectangle(
                    (x0, y0, x0 + theme.square - 1, y0 + theme.square - 1), fill=color
                )

        # Panel strip under the board.
        draw.rectangle(
            (0, theme.padding * 2 + theme.board_px, theme.width, theme.height),
            fill=theme.panel_bg,
        )

        font = _text_font(max(8, theme.padding - 4))
        for col in range(8):
            x = theme.padding + col * theme.square + theme.square // 2
            draw.text((x, theme.padding // 2), FILES[col], font=font,
                      fill=theme.label, anchor="mm")
        for row in range(8):
            y = theme.padding + row * theme.square + theme.square // 2
            draw.text((theme.padding // 2, y), str(8 - row), font=font,
                      fill=theme.label, anchor="mm")

        return np.asarray(image).copy()

    # -- frames ------------------------------------------------------------

    def render(
        self,
        board: Board,
        *,
        highlight_from: tuple[int, int] | None = None,
        highlight_to: tuple[int, int] | None = None,
        label: str = "",
        sublabel: str = "",
        label_color: tuple[int, int, int] | None = None,
        skip: tuple[int, int] | None = None,
        floating: tuple[str, float, float] | None = None,
    ) -> np.ndarray:
        """Render one position.

        `skip` omits a square's piece (it is being animated), and `floating`
        draws ``(piece, x_px, y_px)`` on top for slide animations.
        """
        theme = self.theme
        frame = self._ensure_background().copy()

        for square, color in ((highlight_from, theme.highlight_from),
                              (highlight_to, theme.highlight_to)):
            if square is not None:
                self._tint_square(frame, square, color)

        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece is None or (skip is not None and (row, col) == skip):
                    continue
                composite(
                    frame,
                    self.art.tile(piece),
                    theme.padding + col * theme.square,
                    theme.padding + row * theme.square,
                )

        if floating is not None:
            piece, fx, fy = floating
            composite(frame, self.art.tile(piece), fx, fy)

        if label or sublabel:
            frame = self._draw_panel(frame, label, sublabel, label_color)

        return frame

    def _tint_square(self, frame: np.ndarray, square: tuple[int, int],
                     color: tuple[int, int, int]) -> None:
        row, col = square
        if not (0 <= row < 8 and 0 <= col < 8):
            return
        theme = self.theme
        x0 = theme.padding + col * theme.square
        y0 = theme.padding + row * theme.square
        region = frame[y0:y0 + theme.square, x0:x0 + theme.square]
        blended = region.astype(np.float32) * 0.35 + np.array(color, np.float32) * 0.65
        region[:] = blended.astype(np.uint8)

    def _draw_panel(self, frame: np.ndarray, label: str, sublabel: str,
                    label_color: tuple[int, int, int] | None) -> np.ndarray:
        theme = self.theme
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        top = theme.padding * 2 + theme.board_px

        main_font = _text_font(max(9, theme.panel // 4))
        sub_font = _text_font(max(8, theme.panel // 5))

        if label:
            draw.text(
                (theme.width // 2, top + theme.panel // 3),
                label,
                font=main_font,
                fill=label_color or theme.info,
                anchor="mm",
            )
        if sublabel:
            draw.text(
                (theme.width // 2, top + (2 * theme.panel) // 3),
                sublabel,
                font=sub_font,
                fill=theme.label,
                anchor="mm",
            )
        return np.asarray(image).copy()

    def square_origin(self, row: int, col: int) -> tuple[int, int]:
        """Top-left pixel of a square, used for slide interpolation."""
        theme = self.theme
        return theme.padding + col * theme.square, theme.padding + row * theme.square
