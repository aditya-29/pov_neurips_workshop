"""Experiment registry.

Imports are lazy so that `pov eval` — and the test suite for one experiment —
never pull in another experiment's optional dependencies.
"""

from __future__ import annotations

import importlib
from typing import Any

from pov.errors import PovError

#: experiment name → "module:ClassName"
EXPERIMENTS: dict[str, str] = {
    "chess": "pov.experiments.chess.generate:ChessGenerator",
    "asl": "pov.experiments.asl.generate:AslGenerator",
    "wbw_mcq": "pov.experiments.wbw_mcq.generate:WbwMcqGenerator",
}


class UnknownExperiment(PovError, KeyError):
    """Raised when a config names an experiment that does not exist."""

    def __init__(self, name: str):
        self.name = name
        super().__init__(
            f"unknown experiment {name!r}. Available: {sorted(EXPERIMENTS)}"
        )

    def __str__(self) -> str:  # KeyError repr-quotes its message otherwise
        return self.args[0]


def get_generator_class(name: str) -> type:
    """Resolve an experiment name to its Generator subclass."""
    try:
        target = EXPERIMENTS[name]
    except KeyError:
        raise UnknownExperiment(name) from None
    module_name, _, class_name = target.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_generator(config: Any) -> Any:
    """Instantiate the generator for a :class:`pov.config.Config`."""
    return get_generator_class(config.experiment)(config)
