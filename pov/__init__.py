"""POV — video-vs-image benchmark generation and evaluation.

Two independent halves, joined by a CSV:

    pov.experiments  →  generation (writes media + manifest.csv)
    pov.eval         →  scoring (reads manifest.csv + a model_output column)

Nothing in `pov.eval` imports `pov.experiments`, and generation never calls a
model. See README.md for the end-to-end workflow.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
