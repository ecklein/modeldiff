from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import modeldiff
from modeldiff import CheckSuite, ModelDiffConfig, SavedCheck, StabilityConfig
from modeldiff.analysis.regions import Condition


def _data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    X = pd.DataFrame(
        {
            "risk": rng.normal(size=600),
            "noise": rng.normal(size=600),
        }
    )
    y = np.zeros(len(X))
    baseline = np.zeros(len(X))
    candidate = np.where(X["risk"].to_numpy() > 0, 0.8, 0.0)
    return X, y, baseline, candidate


def _configs() -> tuple[ModelDiffConfig, StabilityConfig]:
    return (
        ModelDiffConfig(
            max_depth=1,
            beam_width=12,
            max_regions=3,
            min_region_fraction=0.10,
            max_region_fraction=0.60,
            n_bootstrap=100,
            n_permutations=99,
            random_state=4,
        ),
        StabilityConfig(
            n_resamples=8,
            subsample_fraction=0.75,
            min_selection_frequency=0.50,
            random_state=4,
        ),
    )


def test_stable_discovery_is_reproducible_and_keeps_validation_untouched() -> None:
    X, y, baseline, candidate = _data()
    config, stability = _configs()

    first = modeldiff.compare_stable_predictions(
        baseline,
        candidate,
        X,
        y,
        "regression",
        "mae",
        config=config,
        stability_config=stability,
    )
    second = modeldiff.compare_stable_predictions(
        baseline,
        candidate,
        X,
        y,
        "regression",
        "mae",
        config=config,
        stability_config=stability,
    )

    assert first.regression_regions
    region = first.regression_regions[0]
    assert region.selection_frequency >= stability.min_selection_frequency
    assert region.total_resamples == stability.n_resamples
    assert 0 <= region.mean_cluster_similarity <= 1
    assert region.triage == "unclassified"
    assert first.to_dict()["analysis"] == "stability_selection"
    assert first.to_json() == second.to_json()
    assert first.split["strategy"] == "random"


def test_stable_region_is_triaged_against_known_checks() -> None:
    X, y, baseline, candidate = _data()
    config, stability = _configs()
    known = CheckSuite(
        task="regression",
        metric="mae",
        checks=(
            SavedCheck(
                check_id="known-risk",
                name="Known high risk",
                conditions=(Condition("risk", ">", -0.5),),
                max_loss_increase=0.0,
                min_samples=10,
                provenance={"source": "test"},
            ),
        ),
    )

    report = modeldiff.compare_stable_predictions(
        baseline,
        candidate,
        X,
        y,
        "regression",
        "mae",
        config=config,
        stability_config=stability,
        known_checks=known,
    )

    assert report.regression_regions[0].triage == "known"
    assert report.regression_regions[0].matched_check_ids == ("known-risk",)
    assert report.to_dict()["triage_counts"]["known"] >= 1


def test_stability_configuration_and_known_suite_are_validated() -> None:
    with pytest.raises(ValueError, match="n_resamples"):
        StabilityConfig(n_resamples=1)

    X, y, baseline, candidate = _data()
    config, stability = _configs()
    mismatched = CheckSuite(task="regression", metric="mse")
    with pytest.raises(ValueError, match="same task and metric"):
        modeldiff.compare_stable_predictions(
            baseline,
            candidate,
            X,
            y,
            "regression",
            "mae",
            config=config,
            stability_config=stability,
            known_checks=mismatched,
        )
