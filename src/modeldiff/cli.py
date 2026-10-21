"""Command-line interface for precomputed-prediction comparisons."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from modeldiff import ModelDiffConfig, compare_predictions


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ModelDiff-ML CLI and return a CI-friendly exit status."""
    parser = build_parser()
    args = parser.parse_args(argv)
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
    failed = args.fail_on_significant_regression and bool(report.regressions(significant_only=True))
    if (
        args.max_overall_regression is not None
        and report.overall["absolute_delta"] > args.max_overall_regression
    ):
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
