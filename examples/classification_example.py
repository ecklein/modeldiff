"""Candidate improves overall while regressing in a planted feature region."""

from __future__ import annotations

import numpy as np
import pandas as pd

import modeldiff


def make_data(n: int = 5000) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    probability = 1 / (1 + np.exp(-(0.7 * X["x1"] - 0.4 * X["x2"])))
    y = rng.binomial(1, probability.to_numpy())

    baseline = np.clip(probability.to_numpy() + rng.normal(0, 0.08, n), 0.03, 0.97)
    candidate = baseline + np.where(y == 1, 0.08, -0.08)

    # The candidate is deliberately overconfident and wrong in this hidden slice.
    planted_regression = (X["x1"].to_numpy() > 1) & (X["x2"].to_numpy() < 0)
    candidate[planted_regression] = np.where(y[planted_regression] == 1, 0.12, 0.88)
    return X, y, baseline, np.clip(candidate, 0.01, 0.99)


if __name__ == "__main__":
    features, target, old_probabilities, new_probabilities = make_data()
    result = modeldiff.compare_predictions(
        old_probabilities,
        new_probabilities,
        features,
        target,
        task="binary_classification",
        metric="log_loss",
        random_state=42,
    )
    print(result.summary())
    result.to_html("examples/classification_report.html")
