from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import modeldiff
from modeldiff import ModelDiffConfig
from modeldiff.metrics import log_loss


def test_planted_regression_is_found_and_overall_improves(
    planted_classification: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
    fast_config: ModelDiffConfig,
) -> None:
    X, y, baseline, candidate = planted_classification
    report = modeldiff.compare_predictions(
        baseline, candidate, X, y, "binary_classification", "log_loss", config=fast_config
    )
    assert report.overall["absolute_delta"] < 0
    significant = report.regressions(significant_only=True)
    assert significant
    planted = (X["x1"].to_numpy() > 1) & (X["x2"].to_numpy() < 0)
    assert any(
        (
            region.conditions[0].apply(X)
            if len(region.conditions) == 1
            else np.logical_and.reduce([c.apply(X) for c in region.conditions])
        )[planted].mean()
        > 0.5
        for region in significant
    )


def test_validation_statistics_only_use_validation_observations(
    planted_classification: tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray],
    fast_config: ModelDiffConfig,
) -> None:
    X, y, baseline, candidate = planted_classification
    report = modeldiff.compare_predictions(
        baseline, candidate, X, y, "binary_classification", "log_loss", config=fast_config
    )
    region = report.regression_regions[0]
    order = np.random.default_rng(fast_config.random_state).permutation(len(X))
    validation = order[int(np.floor(fast_config.discovery_fraction * len(X))) :]
    validation_X = X.iloc[validation].reset_index(drop=True)
    mask = np.logical_and.reduce([condition.apply(validation_X) for condition in region.conditions])
    expected = np.mean(
        log_loss(y[validation], candidate[validation])[mask]
        - log_loss(y[validation], baseline[validation])[mask]
    )
    assert region.validation_delta == pytest.approx(expected)
    assert region.validation_support == int(mask.sum())


def test_input_validation_and_numpy_columns(fast_config: ModelDiffConfig) -> None:
    X = np.arange(40).reshape(20, 2)
    y = np.array([0, 1] * 10)
    report = modeldiff.compare_predictions(
        np.full(20, 0.4),
        np.full(20, 0.6),
        X,
        y,
        "binary_classification",
        "brier",
        config=fast_config,
    )
    assert report.sample_count == 20
    with pytest.raises(ValueError, match="equal sample lengths"):
        modeldiff.compare_predictions(np.ones(19), np.ones(20), X, y, "regression", "mse")
    with pytest.raises(ValueError, match="probabilities"):
        modeldiff.compare_predictions(
            np.ones(20) * 2, np.ones(20), X, y, "binary_classification", "log_loss"
        )
    duplicated = pd.DataFrame(np.ones((20, 2)), columns=["x", "x"])
    with pytest.raises(ValueError, match="duplicate"):
        modeldiff.compare_predictions(np.ones(20), np.ones(20), duplicated, y, "regression", "mae")


def test_model_adapter() -> None:
    class Classifier:
        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            p = np.full(len(X), 0.6)
            return np.column_stack([1 - p, p])

    X = pd.DataFrame({"x": np.arange(40)})
    report = modeldiff.compare(
        Classifier(),
        Classifier(),
        X,
        [0, 1] * 20,
        "binary_classification",
        "brier",
        config=ModelDiffConfig(n_bootstrap=10, n_permutations=10),
    )
    assert "identical" in " ".join(report.warnings)


def test_group_strategy_is_recorded_and_used_by_public_api(
    fast_config: ModelDiffConfig,
) -> None:
    X = pd.DataFrame({"x": np.arange(80)})
    groups = np.repeat(np.arange(20), 4)
    y = np.linspace(0, 1, 80)
    report = modeldiff.compare_predictions(
        y + 0.2,
        y + 0.1,
        X,
        y,
        "regression",
        "mae",
        config=fast_config,
        validation_strategy="group",
        groups=groups,
    )

    assert report.split["strategy"] == "group"
    assert report.split["group_count"] == 20
    assert report.discovery_count + report.validation_count == len(X)
    assert report.to_dict()["split"] == report.split


def test_null_comparison_reports_no_significant_regions(fast_config: ModelDiffConfig) -> None:
    rng = np.random.default_rng(91)
    X = pd.DataFrame({"x1": rng.normal(size=600), "x2": rng.normal(size=600)})
    y = rng.normal(size=600)
    predictions = y + rng.normal(0, 0.5, size=600)
    report = modeldiff.compare_predictions(
        predictions,
        predictions.copy(),
        X,
        y,
        "regression",
        "squared_error",
        config=fast_config,
    )
    assert report.regressions(significant_only=True) == []
    assert report.improvements(significant_only=True) == []
