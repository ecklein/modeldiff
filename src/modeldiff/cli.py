"""Command-line interface for precomputed-prediction comparisons."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from modeldiff import CheckSuite, ModelDiffConfig, compare_predictions, evaluate_checks


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed < float("inf"):
        raise argparse.ArgumentTypeError("must be a finite, nonnegative number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(prog="modeldiff-ml", description=__doc__)
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
    compare_parser.add_argument(
        "--validation-strategy",
        choices=["random", "group", "time", "explicit"],
        default="random",
        help="how to partition discovery and validation rows (default: random)",
    )
    compare_parser.add_argument("--group-column", help="group identifier for group splitting")
    compare_parser.add_argument("--time-column", help="ordering column for time splitting")
    compare_parser.add_argument(
        "--split-column",
        help="column containing 'discovery'/'validation' labels for explicit splitting",
    )
    compare_parser.add_argument("--fail-on-significant-regression", action="store_true")
    compare_parser.add_argument("--max-overall-regression", type=float)
    compare_parser.add_argument(
        "--export-checks",
        type=Path,
        help="export significant regression regions as a reusable check suite",
    )
    compare_parser.add_argument(
        "--check-max-loss-increase",
        type=_nonnegative_float,
        default=0.0,
        help="allowed mean loss increase for exported checks (default: 0)",
    )
    compare_parser.add_argument(
        "--check-min-samples",
        type=_positive_int,
        default=30,
        help="minimum fresh-data support for exported checks (default: 30)",
    )

    check_parser = subparsers.add_parser(
        "check",
        help="evaluate a saved check suite (exit 0 pass, 1 fail/incomplete/empty, 2 error)",
        description=(
            "Evaluate fixed saved checks. Exit 0: all pass; exit 1: failure, insufficient "
            "data, or empty suite; exit 2: invalid suite, inputs, or usage."
        ),
    )
    check_parser.add_argument("--checks", required=True, type=Path)
    check_parser.add_argument("--data", required=True, type=Path)
    check_parser.add_argument("--target", required=True)
    check_parser.add_argument("--baseline-pred", required=True)
    check_parser.add_argument("--candidate-pred", required=True)
    check_parser.add_argument("--exclude", action="append", default=[])
    check_parser.add_argument("--html", type=Path)
    check_parser.add_argument("--json", type=Path)
    return parser


def _run_compare(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    data = pd.read_csv(args.data)
    required = {args.target, args.baseline_pred, args.candidate_pred}
    strategy_columns = {
        "group": ("group_column", args.group_column),
        "time": ("time_column", args.time_column),
        "explicit": ("split_column", args.split_column),
    }
    split_options: dict[str, Any] = {"validation_strategy": args.validation_strategy}
    control_columns: set[str] = set()
    split_column: str | None = None
    if args.validation_strategy in strategy_columns:
        option_name, column = strategy_columns[args.validation_strategy]
        if column is None:
            parser.error(
                f"--{option_name.replace('_', '-')} is required for "
                f"--validation-strategy {args.validation_strategy}"
            )
        required.add(column)
        control_columns.add(column)
        split_column = column
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"missing columns: {', '.join(sorted(missing))}")
    if split_column is not None:
        if args.validation_strategy == "group":
            split_options["groups"] = data[split_column]
        elif args.validation_strategy == "time":
            split_options["time_values"] = data[split_column]
        else:
            labels = data[split_column].astype(str).str.lower()
            unexpected = set(labels) - {"discovery", "validation"}
            if unexpected:
                parser.error(
                    "--split-column values must be 'discovery' or 'validation'; found: "
                    + ", ".join(sorted(unexpected))
                )
            split_options["discovery_indices"] = labels.eq("discovery")
            split_options["validation_indices"] = labels.eq("validation")
    excluded = {args.target, args.baseline_pred, args.candidate_pred} | control_columns | set(
        args.exclude
    )
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
        **split_options,
    )
    print(report.summary())
    if args.html:
        report.to_html(args.html)
    if args.json:
        report.to_json(args.json)
    if args.export_checks:
        suite = report.export_checks(
            max_loss_increase=args.check_max_loss_increase,
            min_samples=args.check_min_samples,
        )
        suite.to_json(args.export_checks)
        print(f"Exported {len(suite.checks)} saved checks to {args.export_checks}.")
    failed = args.fail_on_significant_regression and bool(report.regressions(significant_only=True))
    if (
        args.max_overall_regression is not None
        and report.overall["absolute_delta"] > args.max_overall_regression
    ):
        failed = True
    return 1 if failed else 0


def _run_check(args: argparse.Namespace) -> int:
    suite = CheckSuite.load(args.checks)
    data = pd.read_csv(args.data)
    required = {args.target, args.baseline_pred, args.candidate_pred}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"missing columns: {', '.join(sorted(missing))}")
    excluded = required | set(args.exclude)
    features = data.drop(columns=list(excluded), errors="ignore")
    result = evaluate_checks(
        suite,
        data[args.baseline_pred],
        data[args.candidate_pred],
        features,
        data[args.target],
    )
    print(result.summary())
    if args.json:
        result.to_json(args.json)
    if args.html:
        result.to_html(args.html)
    return 0 if result.overall_status == "pass" else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ModelDiff-ML CLI and return a CI-friendly exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compare":
        return _run_compare(args, parser)
    try:
        return _run_check(args)
    except (ValueError, TypeError, OSError, UnicodeError, pd.errors.ParserError) as error:
        print(f"modeldiff-ml check: error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
