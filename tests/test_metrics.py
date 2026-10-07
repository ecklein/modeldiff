from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.metrics import (
    log_loss as sklearn_log_loss,
)

from modeldiff.metrics import (
    absolute_error,
    accuracy_loss,
    brier_loss,
    log_loss,
    roc_auc,
    squared_error,
)


def test_classification_metrics_match_sklearn() -> None:
    y = np.array([0, 1, 1, 0, 1, 0], dtype=float)
    p = np.array([0.1, 0.8, 0.8, 0.7, 0.55, 0.2])
    assert np.mean(log_loss(y, p)) == pytest.approx(sklearn_log_loss(y, p))
    assert np.mean(brier_loss(y, p)) == pytest.approx(brier_score_loss(y, p))
    assert 1 - np.mean(accuracy_loss(y, p)) == pytest.approx(accuracy_score(y, p >= 0.5))
    assert roc_auc(y, p) == pytest.approx(roc_auc_score(y, p))


def test_roc_auc_ties_and_single_class() -> None:
    assert roc_auc(np.array([0, 1, 0, 1]), np.array([0.1, 0.5, 0.5, 0.9])) == 0.875
    assert np.isnan(roc_auc(np.ones(3), np.arange(3)))


def test_regression_metrics_match_sklearn() -> None:
    y = np.array([1.0, 2.0, 4.0])
    p = np.array([0.5, 2.5, 5.0])
    assert np.mean(squared_error(y, p)) == pytest.approx(mean_squared_error(y, p))
    assert np.mean(absolute_error(y, p)) == pytest.approx(mean_absolute_error(y, p))


def test_log_loss_clips_extreme_probabilities() -> None:
    losses = log_loss(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert np.all(np.isfinite(losses))
    assert np.all(losses > 30)
