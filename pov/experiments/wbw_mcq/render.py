"""Text frame rendering for the word-by-word MCQ experiment.

Every frame is produced once as a numpy array and then reused: the *same*
word frames drive the slow, normal and fast videos — only the hold length
differs — so a question costs one render pass regardless of how many speeds
are configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_SEARCH = (
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


@dataclass(frozen=True)
class Canvas:
    """Canvas geometry and typography for both conditions."""

    width: int = 1200
    height: int = 800
    background: tuple[int, int, int] = (255, 255, 255)
    foreground: tuple[int, int, int] = (0, 0, 0)

    video_font_size: int = 36
    video_padding: int = 80
    video_line_gap: int = 8

    static_font_size: int = 18
    static_padding: int = 60
    static_line_gap: int = 14

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def video_line_height(self) -> int:
        return self.video_font_size + self.video_line_gap

    @property
    def static_line_height(self) -> int:
        return self.static_font_size + self.static_line_gap

    def validate(self) -> "Canvas":
        if self.width % 2 or self.height % 2:
            raise ValueError(
                f"canvas {self.width}x{self.height} must have even dimensions "
                "so it can be encoded as yuv420p"
            )
        if self.width < 64 or self.height < 64:
            raise ValueError(f"canvas {self.width}x{self.height} is too small")
        if self.video_padding * 2 >= self.width:
            raise ValueError("canvas.video_padding leaves no room for text")
        if self.static_padding * 2 >= self.width:
            raise ValueError("canvas.static_padding leaves no room for text")
        return self


@lru_cache(maxsize=16)
def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_SEARCH:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 9.2
        return ImageFont.load_default()


class FrameRenderer:
    """Renders word frames, blank frames, and full-question images."""

    def __init__(self, canvas: Canvas | None = None):
        self.canvas = (canvas or Canvas()).validate()
        self._blank: np.ndarray | None = None

    # -- frames ------------------------------------------------------------

    def render_blank(self) -> np.ndarray:
        """The gap frame between words in vanishing mode (rendered once)."""
        if self._blank is None:
            image = Image.new("RGB", self.canvas.size, self.canvas.background)
            self._blank = np.asarray(image).copy()
        return self._blank

    def render_words(self, words: Sequence[str]) -> np.ndarray:
        """Render `words` centred and wrapped.

        One word for vanishing mode; everything so far for cumulative mode.
        """
        canvas = self.canvas
        image = Image.new("RGB", canvas.size, canvas.background)
        draw = ImageDraw.Draw(image)
        font = _font(canvas.video_font_size)

        lines = wrap_text(" ".join(words), font, draw, canvas.width - 2 * canvas.video_padding)
        total_height = len(lines) * canvas.video_line_height
        y = (canvas.height - total_height) // 2

        for line in lines:
            width = draw.textlength(line, font=font)
            draw.text(((canvas.width - width) // 2, y), line, font=font, fill=canvas.foreground)
            y += canvas.video_line_height

        return np.asarray(image).copy()

    def render_full_question(self, question) -> Image.Image:
        """The static image condition: stem and all options at once."""
        canvas = self.canvas
        image = Image.new("RGB", canvas.size, canvas.background)
        draw = ImageDraw.Draw(image)
        font = _font(canvas.static_font_size)
        max_width = canvas.width - 2 * canvas.static_padding

        lines = list(wrap_text(question.stem, font, draw, max_width))
        lines.append("")
        for letter in ("A", "B", "C", "D"):
            option = f"({letter}) {question.options.get(letter, '')}"
            lines.extend(wrap_text(option, font, draw, max_width))

        total_height = len(lines) * canvas.static_line_height
        y = max(canvas.static_padding, (canvas.height - total_height) // 2)

        for line in lines:
            if line:
                draw.text((canvas.static_padding, y), line, font=font, fill=canvas.foreground)
            y += canvas.static_line_height

        return image


def wrap_text(text: str, font, draw, max_width: int) -> list[str]:
    """Greedy word wrap to `max_width` pixels.

    A single word wider than the line is emitted on its own rather than being
    dropped or looping forever.
    """
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word])
        if not current or draw.textlength(candidate, font=font) <= max_width:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines
