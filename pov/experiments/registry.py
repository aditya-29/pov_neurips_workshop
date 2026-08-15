"""Re-export of :mod:`pov.registry` for `from pov.experiments import ...`."""

from pov.registry import (
    EXPERIMENTS,
    UnknownExperiment,
    build_generator,
    get_generator_class,
)

__all__ = [
    "EXPERIMENTS",
    "UnknownExperiment",
    "build_generator",
    "get_generator_class",
]
