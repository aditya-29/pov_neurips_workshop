"""Progress reporting for long generation runs.

Generation can legitimately run for a long time — a 200-question word-by-word
set at 0.5 words/sec is over a million encoded frames — so a run must say what
it is doing rather than sitting silent.

Falls back to periodic one-line updates when `tqdm` is unavailable or the
output is not a terminal (a redirected log should not collect thousands of
carriage returns).
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - tqdm is a declared dependency
    _tqdm = None


class Progress:
    """Thread-safe progress counter.

    Workers run in a thread pool and finish out of order, so `update` is called
    concurrently; the lock keeps the count and the fallback output coherent.

        with Progress(total=1400, desc="wbw_mcq", unit="file") as bar:
            bar.update()
    """

    #: Seconds between lines in the non-TTY fallback.
    FALLBACK_INTERVAL = 10.0

    def __init__(
        self,
        total: int | None = None,
        desc: str = "",
        unit: str = "item",
        disable: bool | None = None,
        stream: Any = None,
    ):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.count = 0
        self._lock = threading.Lock()
        self._stream = stream if stream is not None else sys.stderr
        self._started = time.monotonic()
        self._last_report = self._started

        # `disable=True` means silence. Otherwise a live bar needs both tqdm and
        # a terminal; failing either, fall back to sparse lines rather than
        # going silent — a long run must never look like a hang.
        self.disabled = bool(disable)
        self._bar = None
        if not self.disabled and _tqdm is not None and _is_tty(self._stream):
            self._bar = _tqdm(
                total=total, desc=desc, unit=unit, dynamic_ncols=True,
                file=self._stream, leave=False,
            )
        self._fallback = not self.disabled and self._bar is None

    # -- counting ----------------------------------------------------------

    def update(self, n: int = 1) -> None:
        with self._lock:
            self.count += n
            if self._bar is not None:
                self._bar.update(n)
            elif self._fallback:
                self._maybe_report()

    def set_description(self, text: str) -> None:
        self.desc = text
        if self._bar is not None:
            self._bar.set_description(text)

    def _maybe_report(self) -> None:
        now = time.monotonic()
        if now - self._last_report < self.FALLBACK_INTERVAL:
            return
        self._last_report = now
        elapsed = now - self._started
        if self.total:
            share = self.count / self.total
            eta = (elapsed / share - elapsed) if share > 0 else 0.0
            line = (
                f"{self.desc}: {self.count}/{self.total} {self.unit}s "
                f"({share:.0%}) — {elapsed:.0f}s elapsed, ~{eta:.0f}s left"
            )
        else:
            line = f"{self.desc}: {self.count} {self.unit}s — {elapsed:.0f}s elapsed"
        print(line, file=self._stream, flush=True)

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _is_tty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False
