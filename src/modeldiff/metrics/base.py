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


def observation_losses(
    task: str,
    metric: str,
    y: np.ndarray,
    predictions: np.ndarray,
    *,
    threshold: float = 0.5,
    epsilon: float = 1e-15,
) -> np.ndarray:
    """Compute the observation-level loss used for region and saved-check deltas."""
    function = get_loss_function(task, metric)
    if task == "binary_classification" and metric == "log_loss":
        return function(y, predictions, epsilon=epsilon)
    if task == "binary_classification" and metric == "accuracy":
        return function(y, predictions, threshold=threshold)
    return function(y, predictions)


def primary_metric_value(losses: np.ndarray, metric: str) -> float:
    """Aggregate observation-level loss into the requested primary metric."""
    mean_loss = float(np.mean(losses))
    return float(np.sqrt(mean_loss)) if metric == "rmse" else mean_loss
