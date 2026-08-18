"""The README must stay true to the code.

A document that quietly drifts is worse than no document: an agent following it
will confidently do the wrong thing. These tests derive the facts from the code
and fail if the README stops matching, so documenting a field, metric, prompt, or
CLI flag is not optional when one is added.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parent.parent / "README.md"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


class TestManifestSchema:
    def test_every_core_field_is_documented(self, readme):
        from pov.manifest import CORE_FIELDS

        missing = [f for f in CORE_FIELDS if f"`{f}`" not in readme]
        assert not missing, f"undocumented core fields: {missing}"

    def test_the_stated_core_field_count_is_right(self, readme):
        from pov.manifest import CORE_FIELDS

        assert f"all {len(CORE_FIELDS)}" in readme

    def test_model_output_field_is_documented(self, readme):
        from pov.manifest import MODEL_OUTPUT_COLUMN

        assert f"`{MODEL_OUTPUT_COLUMN}`" in readme


class TestMetrics:
    def test_every_metric_of_every_scorer_is_documented(self, readme):
        from pov.eval import SCORERS

        missing = []
        for name, cls in SCORERS.items():
            for metric in cls().metrics:
                if f"`{metric}`" not in readme:
                    missing.append(f"{name}.{metric}")
        assert not missing, f"undocumented metrics: {missing}"

    def test_every_experiment_has_a_metrics_row(self, readme):
        from pov.eval import SCORERS

        for name in SCORERS:
            assert f"| `{name}` |" in readme, f"{name} missing from the metrics table"


class TestPrompts:
    def test_every_prompt_is_documented(self, readme):
        from pov import prompts

        missing = [
            f"{exp}/{kind}"
            for exp, kind in prompts.available()
            if f"`{exp}/{kind}`" not in readme
        ]
        assert not missing, f"undocumented prompts: {missing}"


class TestCli:
    def test_every_subcommand_is_documented(self, readme):
        from pov.cli import build_parser

        parser = build_parser()
        subparsers = [
            action for action in parser._actions
            if hasattr(action, "choices") and isinstance(action.choices, dict)
        ][0]
        missing = [
            name for name in subparsers.choices if f"pov {name}" not in readme
        ]
        assert not missing, f"undocumented subcommands: {missing}"

    def test_every_flag_of_every_subcommand_is_documented(self, readme):
        from pov.cli import build_parser

        parser = build_parser()
        subparsers = [
            action for action in parser._actions
            if hasattr(action, "choices") and isinstance(action.choices, dict)
        ][0]

        missing = []
        for name, sub in subparsers.choices.items():
            for action in sub._actions:
                options = [f for f in action.option_strings if f not in ("-h", "--help")]
                if not options:
                    continue
                # Either spelling counts: the README uses `-c` and `-i`, which
                # document the same flag as `--config` and `--input`.
                if not any(option in readme for option in options):
                    missing.append(f"pov {name} {options[-1]}")
        assert not missing, f"undocumented flags: {missing}"

    def test_readme_invents_no_subcommands(self, readme):
        """Anything invoked as `pov <word>` in a code block must really exist."""
        from pov.cli import build_parser

        parser = build_parser()
        subparsers = [
            action for action in parser._actions
            if hasattr(action, "choices") and isinstance(action.choices, dict)
        ][0]
        real = set(subparsers.choices)

        # Only look inside fenced code blocks, and only at lines that actually
        # start with `pov`. Prose mentions ("shipped with pov so everyone…") and
        # the sentence listing commands that deliberately do NOT exist are not
        # invocations.
        invoked: set[str] = set()
        for block in re.findall(r"```[a-z]*\n(.*?)```", readme, re.S):
            for line in block.splitlines():
                match = re.match(r"\s*pov\s+([a-z_]+)", line)
                if match:
                    invoked.add(match.group(1))

        invented = invoked - real
        assert not invented, f"README invokes non-existent commands: {invented}"

    def test_readme_says_which_commands_do_not_exist(self, readme):
        # A hallucinating reader is likeliest to reach for these.
        assert "no `pov run`" in readme
        assert "no `pov predict`" in readme


class TestConfigKeys:
    def test_every_run_and_video_key_is_documented(self, readme):
        import dataclasses

        from pov.config import RunConfig, VideoConfig

        missing = []
        for cls in (RunConfig, VideoConfig):
            for field in dataclasses.fields(cls):
                if f"`{field.name}`" not in readme:
                    missing.append(f"{cls.__name__}.{field.name}")
        assert not missing, f"undocumented config keys: {missing}"

    def test_every_experiment_param_is_documented(self, readme):
        import dataclasses

        from pov.experiments.asl.generate import AslParams
        from pov.experiments.chess.generate import ChessParams
        from pov.experiments.wbw_mcq.generate import WbwParams

        # Internal-only fields with no YAML key of their own.
        internal = {"auto_duration_cache", "timing", "theme", "canvas", "buckets"}
        missing = []
        for cls in (ChessParams, WbwParams, AslParams):
            for field in dataclasses.fields(cls):
                if field.name in internal:
                    continue
                if f"`{field.name}`" not in readme:
                    missing.append(f"{cls.__name__}.{field.name}")
        assert not missing, f"undocumented params: {missing}"


class TestExperimentCoverage:
    def test_every_experiment_is_named(self, readme):
        from pov.registry import EXPERIMENTS

        for name in EXPERIMENTS:
            assert f"`{name}`" in readme

    def test_source_data_table_covers_every_experiment(self, readme):
        from pov.registry import EXPERIMENTS

        section = readme.split("# 1. Get the source data")[1].split("# 2.")[0]
        for name in EXPERIMENTS:
            assert f"`{name}`" in section, f"{name} missing from the source-data table"


class TestScriptsExist:
    @pytest.mark.parametrize(
        "path", ["scripts/fetch_mmlu.py", "scripts/download_how2sign.sh"]
    )
    def test_referenced_scripts_are_present(self, readme, path):
        assert path in readme
        assert (README.parent / path).exists(), f"README references missing {path}"

    def test_referenced_example_file_exists(self, readme):
        assert "examples/questions.jsonl" in readme
        assert (README.parent / "examples/questions.jsonl").exists()

    def test_referenced_configs_exist(self, readme):
        for name in ("chess", "asl", "wbw_mcq"):
            assert f"configs/{name}.yaml" in readme
            assert (README.parent / f"configs/{name}.yaml").exists()


class TestStatedCounts:
    def test_test_count_is_current(self, readme):
        """The README quotes a test count; keep it honest."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=README.parent, capture_output=True, text=True,
        )
        match = re.search(r"(\d+) tests? collected", result.stdout)
        if not match:  # pragma: no cover - collection shape changed
            pytest.skip("could not determine the collected test count")
        actual = int(match.group(1))

        quoted = [int(n) for n in re.findall(r"(\d+) tests", readme)]
        assert quoted, "README no longer states a test count"
        assert all(abs(q - actual) <= 5 for q in quoted), (
            f"README says {quoted} tests, suite actually collects {actual}"
        )
