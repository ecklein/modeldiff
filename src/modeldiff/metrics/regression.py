"""Regression loss functions."""

from __future__ import annotations

import numpy as np


def squared_error(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Return squared error for each observation."""
    error = np.asarray(predictions, dtype=float) - np.asarray(y_true, dtype=float)
    return np.asarray(error**2, dtype=float)


def absolute_error(y_true: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Return absolute error for each observation."""
    error = np.asarray(predictions, dtype=float) - np.asarray(y_true, dtype=float)
    return np.asarray(np.abs(error), dtype=float)


def regression_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return common aggregate regression metrics."""
    squared = squared_error(y_true, predictions)
    return {
        "mse": float(np.mean(squared)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "mae": float(np.mean(absolute_error(y_true, predictions))),
    }
