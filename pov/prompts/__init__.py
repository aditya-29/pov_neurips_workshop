"""Task prompts — the instruction half of each benchmark.

The prompt is part of the benchmark, not part of your harness: two people
running "the ASL task" with different instructions are not running the same
benchmark. So the prompts ship here, verbatim from the original study, rather
than living in whatever script happens to call a model.

`pov` still never calls a model. These are strings; you decide what to do with
them.

    from pov import prompts

    prompts.get("asl")                    # the ASL translation task
    prompts.get("chess")                  # chess move transcription
    prompts.get("wbw_mcq")                # MCQ system prompt
    prompts.for_condition("wbw_mcq", "vanishing_slow")   # matching user message

    # The ASL judge is a template — render it with the pair to score:
    prompts.render("asl", "judge", ground_truth=..., model_output=...)

Its output contract (`{"strict": bool, "loose": bool, "explanation": str}`)
feeds the `judge_strict` / `judge_loose` fields that :mod:`pov.eval.asl`
already aggregates.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from pov.errors import PovError

try:  # Python 3.9+
    from importlib.resources import files as _resource_files
except ImportError:  # pragma: no cover - floor is 3.10
    _resource_files = None

#: (experiment, kind) → filename under prompts/data/
_REGISTRY: dict[tuple[str, str], str] = {
    ("asl", "task"): "asl_task.txt",
    ("asl", "judge"): "asl_judge.txt",
    # Translation-only variant. The default asl/task asks for
    # GLOSS/TRANSLATION/CONFIDENCE, but AslScorer compares model_output
    # verbatim to a plain sentence: in a 170-row run 93 replies never
    # reached a TRANSLATION section, so ~5.9k characters of gloss analysis
    # were scored as the hypothesis and median WER hit 31.97. This asks
    # for the sentence alone, so the reply *is* the hypothesis.
    ("asl", "task_direct"): "asl_task_direct.txt",
    ("chess", "task"): "chess_task.txt",
    # The original study's wording, kept for provenance. It describes a caption
    # that read "Move N — Color: <notation>"; the renderer now prints only the
    # move number, so this prompt tells the model to read a label that no longer
    # exists. Use it to reproduce the old runs, not for new ones.
    ("chess", "task_legacy"): "chess_task_legacy.txt",
    ("wbw_mcq", "task"): "wbw_mcq_task.txt",
    ("wbw_mcq", "user_text"): "wbw_mcq_user_text.txt",
    ("wbw_mcq", "user_static_image"): "wbw_mcq_user_static_image.txt",
    ("wbw_mcq", "user_video"): "wbw_mcq_user_video.txt",
}

#: The kind returned when none is asked for.
DEFAULT_KIND = "task"

_PLACEHOLDER = re.compile(r"(?<!\{)\{([a-zA-Z_][a-zA-Z_0-9]*)\}(?!\})")


class PromptError(PovError, KeyError):
    """Raised for an unknown prompt or a bad render."""

    def __str__(self) -> str:  # KeyError repr-quotes its message otherwise
        return self.args[0]


def available() -> list[tuple[str, str]]:
    """Every registered ``(experiment, kind)`` pair, sorted."""
    return sorted(_REGISTRY)


def experiments() -> list[str]:
    """Experiments that have at least one prompt."""
    return sorted({experiment for experiment, _ in _REGISTRY})


def kinds(experiment: str) -> list[str]:
    """Prompt kinds available for one experiment."""
    return sorted(kind for name, kind in _REGISTRY if name == experiment)


@lru_cache(maxsize=None)
def get(experiment: str, kind: str = DEFAULT_KIND) -> str:
    """The prompt text for ``(experiment, kind)``."""
    try:
        filename = _REGISTRY[(experiment, kind)]
    except KeyError:
        if experiment not in experiments():
            raise PromptError(
                f"no prompts for experiment {experiment!r}. "
                f"Available: {experiments()}"
            ) from None
        raise PromptError(
            f"no {kind!r} prompt for {experiment!r}. "
            f"Available kinds: {kinds(experiment)}"
        ) from None
    return _read(filename)


def placeholders(experiment: str, kind: str = DEFAULT_KIND) -> list[str]:
    """Field names :func:`render` requires, in sorted order.

    A doubled brace (``{{"strict": ...}}`` in the judge's output contract) is
    a literal brace, not a placeholder, and is not reported here.
    """
    return sorted(set(_PLACEHOLDER.findall(get(experiment, kind))))


def render(experiment: str, kind: str = DEFAULT_KIND, **fields: Any) -> str:
    """Fill a prompt's placeholders.

    Raises rather than silently emitting a half-filled prompt: a judge prompt
    missing its `model_output` would score whatever the model last said.
    """
    text = get(experiment, kind)
    required = set(placeholders(experiment, kind))
    missing = sorted(required - set(fields))
    if missing:
        raise PromptError(
            f"{experiment}/{kind} prompt needs {sorted(required)}; missing {missing}"
        )
    unexpected = sorted(set(fields) - required)
    if unexpected:
        raise PromptError(
            f"{experiment}/{kind} prompt takes {sorted(required)}; "
            f"got unexpected {unexpected}"
        )
    try:
        return text.format(**fields)
    except (KeyError, IndexError, ValueError) as exc:  # pragma: no cover
        raise PromptError(f"{experiment}/{kind} prompt failed to render: {exc}") from exc


def for_condition(experiment: str, condition: str) -> str:
    """The user message matching a manifest `condition`.

    Conditions come straight from the manifest, so this must cover every value
    the generators emit: `static_image`, `vanishing_*`, `cumulative_*` for
    word-by-word MCQ, and `video*` for chess and ASL.
    """
    if experiment != "wbw_mcq":
        # Chess and ASL use one instruction for every condition.
        return get(experiment, DEFAULT_KIND)

    if condition == "text":
        return get("wbw_mcq", "user_text")
    if condition == "static_image":
        return get("wbw_mcq", "user_static_image")
    if condition.startswith(("vanishing", "cumulative")):
        return get("wbw_mcq", "user_video")
    raise PromptError(
        f"no user message for wbw_mcq condition {condition!r} "
        "(expected text, static_image, vanishing_*, or cumulative_*)"
    )


@lru_cache(maxsize=None)
def _read(filename: str) -> str:
    if _resource_files is not None:
        return (_resource_files(__name__) / "data" / filename).read_text(encoding="utf-8")
    from pathlib import Path  # pragma: no cover - fallback for odd installs

    return (Path(__file__).parent / "data" / filename).read_text(encoding="utf-8")


__all__ = [
    "DEFAULT_KIND",
    "PromptError",
    "available",
    "experiments",
    "for_condition",
    "get",
    "kinds",
    "placeholders",
    "render",
]
