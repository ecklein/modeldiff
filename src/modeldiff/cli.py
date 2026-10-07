"""Command-line interface for precomputed-prediction comparisons."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from modeldiff import ModelDiffConfig, compare_predictions


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(prog="modeldiff", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare_parser = subparsers.add_parser("compare", help="compare predictions in a CSV file")
    compare_parser.add_argument("--data", required=True, type=Path)
    compare_parser.add_argument("--target", required=True)
    compare_parser.add_argument("--baseline-pred", required=True)
    compare_parser.add_argument("--candidate-pred", required=True)
    compare_parser.add_argument(
        "--task", required=True, choices=["binary_classification", "regression"]
    )
    compare_parser.add_argument("--metric", required=True)
    compare_parser.add_argument("--exclude", action="append", default=[])
    compare_parser.add_argument("--html", type=Path)
    compare_parser.add_argument("--json", type=Path)
    compare_parser.add_argument("--random-state", type=int, default=42)
    compare_parser.add_argument("--fail-on-significant-regression", action="store_true")
    compare_parser.add_argument("--max-overall-regression", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ModelDiff CLI and return a CI-friendly exit status."""
    args = build_parser().parse_args(argv)
    data = pd.read_csv(args.data)
    required = {args.target, args.baseline_pred, args.candidate_pred}
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"missing columns: {', '.join(sorted(missing))}")
    excluded = required | set(args.exclude)
    features = data.drop(columns=list(excluded), errors="ignore")
    if features.shape[1] == 0:
        raise SystemExit("no feature columns remain after exclusions")
    config = ModelDiffConfig(random_state=args.random_state)
    report = compare_predictions(
        data[args.baseline_pred],
        data[args.candidate_pred],
        features,
        data[args.target],
        args.task,
        args.metric,
        config=config,
    )
    print(report.summary())
    if args.html:
        report.to_html(args.html)
    if args.json:
        report.to_json(args.json)
    failed = args.fail_on_significant_regression and bool(report.regressions(significant_only=True))
    if (
        args.max_overall_regression is not None
        and report.overall["absolute_delta"] > args.max_overall_regression
    ):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
