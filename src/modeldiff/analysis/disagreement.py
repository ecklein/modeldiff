"""Prediction-level behavioral diagnostics."""

from __future__ import annotations

import numpy as np


def classification_disagreement(
    baseline: np.ndarray, candidate: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Summarize probability shifts and hard-label disagreement."""
    shift = np.abs(candidate - baseline)
    baseline_confidence = np.abs(baseline - 0.5) * 2
    candidate_confidence = np.abs(candidate - 0.5) * 2
    return {
        "mean_absolute_probability_shift": float(np.mean(shift)),
        "median_absolute_probability_shift": float(np.median(shift)),
        "p95_absolute_probability_shift": float(np.quantile(shift, 0.95)),
        "hard_classification_disagreement_rate": float(
            np.mean((baseline >= threshold) != (candidate >= threshold))
        ),
        "mean_candidate_confidence_change": float(
            np.mean(candidate_confidence - baseline_confidence)
        ),
    }


def regression_disagreement(baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Summarize regression prediction shifts."""
    difference = np.abs(candidate - baseline)
    if np.std(baseline) == 0 or np.std(candidate) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(baseline, candidate)[0, 1])
    return {
        "mean_absolute_prediction_difference": float(np.mean(difference)),
        "median_absolute_prediction_difference": float(np.median(difference)),
        "p95_absolute_prediction_difference": float(np.quantile(difference, 0.95)),
        "prediction_correlation": correlation,
    }
