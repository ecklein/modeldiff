"""Aggregate metric comparisons."""

from __future__ import annotations

import numpy as np

from modeldiff.metrics.classification import classification_metrics
from modeldiff.metrics.regression import regression_metrics


def aggregate_metrics(
    task: str,
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    threshold: float,
    epsilon: float,
) -> dict[str, dict[str, float]]:
    """Compute full aggregate metric sets for both models."""
    if task == "binary_classification":
        baseline_metrics = classification_metrics(y, baseline, threshold, epsilon)
        candidate_metrics = classification_metrics(y, candidate, threshold, epsilon)
    else:
        baseline_metrics = regression_metrics(y, baseline)
        candidate_metrics = regression_metrics(y, candidate)
    return {"baseline": baseline_metrics, "candidate": candidate_metrics}
