"""Discover, save, and reevaluate a fixed regression slice."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

import modeldiff
from modeldiff import ModelDiffConfig


def data(seed: int, size: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Create a deterministic dataset with a reusable priority segment."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "segment": rng.choice(["standard", "priority"], size=size, p=[0.7, 0.3]),
            "score": rng.normal(size=size),
        }
    )
    y = 2 * X["score"].to_numpy() + rng.normal(0, 0.2, size)
    baseline = y + rng.normal(0, 0.1, size)
    return X, y, baseline


if __name__ == "__main__":
    X_discovery, y_discovery, baseline_discovery = data(7, 1200)
    priority = X_discovery["segment"].eq("priority").to_numpy()
    regressing_candidate = baseline_discovery.copy()
    regressing_candidate[priority] += 1.0

    report = modeldiff.compare_predictions(
        baseline_discovery,
        regressing_candidate,
        X_discovery,
        y_discovery,
        task="regression",
        metric="mae",
        config=ModelDiffConfig(n_bootstrap=200, n_permutations=500, random_state=42),
    )
    regions = report.regressions(significant_only=True)
    if not regions:
        raise RuntimeError("The deterministic example did not discover its planted regression.")
    segment_regions = [
        region
        for region in regions
        if len(region.conditions) == 1 and region.conditions[0].feature == "segment"
    ]
    if not segment_regions:
        raise RuntimeError("The deterministic example did not preserve the planted segment.")

    checks = report.export_checks(
        regions=segment_regions[:1],
        max_loss_increase=0.05,
        min_samples=50,
        names=["Priority segment regression"],
    )

    X_fresh, y_fresh, production = data(8, 400)
    fresh_priority = X_fresh["segment"].eq("priority").to_numpy()
    still_regressing = production.copy()
    still_regressing[fresh_priority] += 1.0
    improved = production.copy()

    with TemporaryDirectory() as directory:
        path = f"{directory}/model_checks.json"
        checks.to_json(path)
        failed = modeldiff.evaluate_checks(path, production, still_regressing, X_fresh, y_fresh)
        passed = modeldiff.evaluate_checks(path, production, improved, X_fresh, y_fresh)

    print("Candidate that still regresses:")
    print(failed.summary())
    print("\nImproved candidate:")
    print(passed.summary())
