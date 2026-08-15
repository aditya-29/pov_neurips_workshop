"""Single-file HTML reporting for evaluation results."""

from pov.report.html import (
    build_report,
    build_report_from_csv,
    build_report_from_file,
)

__all__ = ["build_report", "build_report_from_csv", "build_report_from_file"]
