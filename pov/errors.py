"""Common base for every error pov raises on purpose.

The CLI prints :class:`PovError` as a plain message and exits 2. Anything that
is *not* a `PovError` is a bug in pov, and is reported with its type name and a
non-zero exit so it is obvious which is which.

Each subclass keeps its historical base (`ValueError` / `RuntimeError` /
`KeyError`) so existing `except` clauses and tests are unaffected.
"""

from __future__ import annotations


class PovError(Exception):
    """A problem with the user's input, config, or data — not a pov bug."""
