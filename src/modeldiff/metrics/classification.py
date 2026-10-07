"""Binary-classification metrics implemented with NumPy and SciPy."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def log_loss(y_true: np.ndarray, probabilities: np.ndarray, epsilon: float = 1e-15) -> np.ndarray:
    """Return binary cross-entropy for each observation."""
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), epsilon, 1 - epsilon)
    return np.asarray(-(y * np.log(p) + (1 - y) * np.log(1 - p)), dtype=float)


def brier_loss(y_true: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Return squared probability error for each observation."""
    loss = (np.asarray(probabilities, dtype=float) - np.asarray(y_true, dtype=float)) ** 2
    return np.asarray(loss, dtype=float)


def accuracy_loss(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Return a zero-one classification loss for each observation."""
    labels = np.asarray(probabilities, dtype=float) >= threshold
    return np.asarray(labels != np.asarray(y_true, dtype=float), dtype=float)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC with average ranks, correctly handling ties."""
    y = np.asarray(y_true)
    positive = y == 1
    n_positive = int(np.sum(positive))
    n_negative = len(y) - n_positive
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = rankdata(np.asarray(scores, dtype=float), method="average")
    rank_sum = float(np.sum(ranks[positive]))
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def classification_metrics(
    y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5, epsilon: float = 1e-15
) -> dict[str, float]:
    """Return all aggregate binary-classification metrics."""
    losses = accuracy_loss(y_true, probabilities, threshold)
    return {
        "accuracy": float(1 - np.mean(losses)),
        "log_loss": float(np.mean(log_loss(y_true, probabilities, epsilon))),
        "brier": float(np.mean(brier_loss(y_true, probabilities))),
        "roc_auc": roc_auc(y_true, probabilities),
    }
