"""Command-line interface.

    pov generate -c configs/chess.yaml
    pov eval -i data/chess/<run>/manifest.jsonl
    pov report -i data/chess/<run>/scored.jsonl -o report.html
    pov validate -c configs/chess.yaml
    pov experiments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pov import __version__
from pov.config import Config, ConfigError, deep_merge, parse_override
from pov.errors import PovError
from pov.registry import EXPERIMENTS, UnknownExperiment, build_generator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pov",
        description="Generate and evaluate video-vs-image benchmark data.",
    )
    parser.add_argument("--version", action="version", version=f"pov {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- generate ----------------------------------------------------------
    generate = subparsers.add_parser(
        "generate", help="generate media and a manifest from a YAML config"
    )
    generate.add_argument("-c", "--config", required=True, help="path to a YAML config")
    generate.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
        help="override a config value, e.g. --set run.workers=4 (repeatable)",
    )
    generate.add_argument(
        "--dry-run", action="store_true",
        help="validate the config and source data, and print the resolved plan, "
             "without writing media",
    )
    generate.add_argument(
        "--quiet", "-q", action="store_true",
        help="suppress the progress bar",
    )
    generate.set_defaults(func=_cmd_generate)

    # -- eval --------------------------------------------------------------
    evaluate_cmd = subparsers.add_parser(
        "eval", help="score a manifest that has a model_output column"
    )
    evaluate_cmd.add_argument(
        "-i", "--input", required=True, help="manifest (.jsonl or .csv) with a filled-in model_output column"
    )
    evaluate_cmd.add_argument(
        "-o", "--output-dir", default=None,
        help="where to write scored/summary output (default: next to the input)",
    )
    evaluate_cmd.add_argument(
        "--run-dir", default=None,
        help="run directory that ground_truth_path values are relative to "
             "(default: the input file's directory)",
    )
    evaluate_cmd.add_argument(
        "--group-by", default=None,
        help="comma-separated columns to group the summary by "
             "(default: experiment,condition[,model])",
    )
    evaluate_cmd.add_argument(
        "--report", default=None, metavar="HTML",
        help="also write a single-file HTML report to this path",
    )
    evaluate_cmd.set_defaults(func=_cmd_eval)

    # -- report ------------------------------------------------------------
    report = subparsers.add_parser(
        "report", help="build a single-file HTML report from scored results"
    )
    report.add_argument("-i", "--input", required=True,
                    help="scored rows from `pov eval` (.jsonl or .csv)")
    report.add_argument("-o", "--output", default=None, help="output .html path")
    report.add_argument(
        "--run-dir", default=None,
        help="run directory holding media/ (default: the input file's directory)",
    )
    report.add_argument("--title", default=None, help="page title")
    report.add_argument(
        "--max-samples", type=int, default=None,
        help="cap how many samples are listed in the page",
    )
    report.set_defaults(func=_cmd_report)

    # -- validate ----------------------------------------------------------
    validate = subparsers.add_parser("validate", help="check a config without generating")
    validate.add_argument("-c", "--config", required=True)
    validate.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    validate.set_defaults(func=_cmd_validate)

    # -- experiments -------------------------------------------------------
    listing = subparsers.add_parser("experiments", help="list available experiments")
    listing.set_defaults(func=_cmd_experiments)

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────────────────────


def _load_config(args: argparse.Namespace) -> Config:
    overrides: dict = {}
    for item in args.overrides:
        overrides = deep_merge(overrides, parse_override(item))
    return Config.load(args.config, overrides=overrides or None)


def _cmd_generate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    generator = build_generator(config)
    generator.show_progress = not getattr(args, "quiet", False)

    if args.dry_run:
        layout = generator.build_layout()
        print(f"experiment : {config.experiment}")
        print(f"run id     : {generator.run_id}")
        print(f"run dir    : {layout.run_dir}")
        print(f"manifest   : {layout.manifest_path}")
        print(f"config hash: {generator.config_hash}")
        print(f"encoder    : {config.video.codec} crf={config.video.crf} "
              f"preset={config.video.preset} fps={config.video.fps}")

        problems = generator.check_inputs()
        if problems:
            # Flush first: without it the stderr block jumps ahead of the
            # buffered plan above and the output reads out of order.
            sys.stdout.flush()
            print("\nsource data is NOT ready — this run would fail:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 2

        for line in generator.describe_inputs():
            print(line)
        print("config is valid, source data is present; "
              "no media written (--dry-run)")
        return 0

    result = generator.run()
    print(result.summary())
    return 0 if result.ok else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    from pov.eval import evaluate

    group_by = (
        [column.strip() for column in args.group_by.split(",") if column.strip()]
        if args.group_by
        else None
    )
    report = evaluate(
        args.input,
        args.output_dir,
        run_dir=args.run_dir,
        group_by=group_by,
    )
    print(report.text_summary())
    if report.scored_path:
        print(f"\nwrote {report.scored_path}")
    if report.summary_path:
        print(f"wrote {report.summary_path}")

    if args.report:
        from pov.report import build_report

        path = build_report(
            report.rows,
            args.report,
            run_dir=args.run_dir or Path(args.input).parent,
            summary=report.summary,
            metrics=report.metrics,
            group_columns=report.group_columns,
        )
        print(f"wrote {path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from pov.report import build_report_from_file

    output = args.output or str(Path(args.input).with_suffix(".html"))
    path = build_report_from_file(
        args.input,
        output,
        run_dir=args.run_dir,
        title=args.title,
        max_samples=args.max_samples,
    )
    print(f"wrote {path}")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    config = _load_config(args)
    build_generator(config)  # parses params, raising on anything invalid
    print(f"{args.config}: OK ({config.experiment}, config hash {config.config_hash()})")
    return 0


def _cmd_experiments(args: argparse.Namespace) -> int:
    for name in sorted(EXPERIMENTS):
        print(name)
    return 0


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, UnknownExperiment) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except PovError as exc:
        # Every deliberate pov error: a problem with the user's input or data.
        # Printed as a plain message — the exception class name is an internal
        # detail and only confuses the reader.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        # Not a PovError, so this is a bug in pov rather than bad input. Name
        # the type, and say so, so it gets reported instead of puzzled over.
        print(f"internal error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("This is a bug in pov, not a problem with your input.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
