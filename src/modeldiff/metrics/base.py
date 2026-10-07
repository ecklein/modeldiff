"""Metric dispatch and aggregate comparison helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .classification import accuracy_loss, brier_loss, log_loss
from .regression import absolute_error, squared_error

LossFunction = Callable[..., np.ndarray]


def get_loss_function(task: str, metric: str) -> LossFunction:
    """Resolve a task/metric pair to its observation-level loss function."""
    if task == "binary_classification":
        return {"log_loss": log_loss, "brier": brier_loss, "accuracy": accuracy_loss}[metric]
    normalized = {"mse": "squared_error", "rmse": "squared_error", "mae": "absolute_error"}.get(
        metric, metric
    )
    return {"squared_error": squared_error, "absolute_error": absolute_error}[normalized]


def primary_metric_value(losses: np.ndarray, metric: str) -> float:
    """Aggregate observation-level loss into the requested primary metric."""
    mean_loss = float(np.mean(losses))
    return float(np.sqrt(mean_loss)) if metric == "rmse" else mean_loss
