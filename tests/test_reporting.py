from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import modeldiff
from modeldiff import ModelDiffConfig
from modeldiff.cli import main


def test_report_formats_are_standalone_and_serializable(tmp_path: Path) -> None:
    X = pd.DataFrame({"x": np.arange(40), "unsafe<name": ["a"] * 40})
    y = np.arange(40, dtype=float)
    config = ModelDiffConfig(n_bootstrap=20, n_permutations=20, max_regions=2)
    report = modeldiff.compare_predictions(y + 1, y + 0.5, X, y, "regression", "mae", config=config)
    text = report.summary()
    assert "ModelDiff-ML Report" in text
    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"
    report.to_json(json_path)
    report.to_html(html_path)
    loaded = json.loads(json_path.read_text())
    assert loaded["schema_version"] == "1.0"
    html = html_path.read_text()
    assert "<!doctype html>" in html
    assert "Methodology and configuration" in html
    assert "unsafe&lt;name" in html or "unsafe<name" not in html


def test_cli_writes_outputs_and_enforces_overall_threshold(tmp_path: Path) -> None:
    data = pd.DataFrame(
        {
            "feature": np.arange(40),
            "y": np.arange(40, dtype=float),
            "old": np.arange(40, dtype=float),
            "new": np.arange(40, dtype=float) + 1,
        }
    )
    csv = tmp_path / "data.csv"
    html = tmp_path / "out.html"
    json_path = tmp_path / "out.json"
    data.to_csv(csv, index=False)
    exit_code = main(
        [
            "compare",
            "--data",
            str(csv),
            "--target",
            "y",
            "--baseline-pred",
            "old",
            "--candidate-pred",
            "new",
            "--task",
            "regression",
            "--metric",
            "mae",
            "--html",
            str(html),
            "--json",
            str(json_path),
            "--max-overall-regression",
            "0.5",
        ]
    )
    assert exit_code == 1
    assert html.exists() and json_path.exists()
