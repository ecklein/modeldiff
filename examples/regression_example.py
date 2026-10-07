"""Regression comparison with a planted local degradation."""

from __future__ import annotations

import numpy as np
import pandas as pd

import modeldiff

if __name__ == "__main__":
    rng = np.random.default_rng(7)
    n = 4000
    X = pd.DataFrame({"age": rng.uniform(18, 80, n), "utilization": rng.uniform(0, 1, n)})
    y = 0.5 * X["age"].to_numpy() + 8 * X["utilization"].to_numpy() + rng.normal(0, 2, n)
    baseline = y + rng.normal(0, 3, n)
    candidate = y + rng.normal(0, 2.5, n)
    planted = (X["age"].to_numpy() < 30) & (X["utilization"].to_numpy() > 0.8)
    candidate[planted] += 8

    report = modeldiff.compare_predictions(
        baseline, candidate, X, y, task="regression", metric="mae", random_state=42
    )
    print(report.summary())
