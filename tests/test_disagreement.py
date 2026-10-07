from __future__ import annotations

import numpy as np
import pytest

from modeldiff.analysis.disagreement import classification_disagreement, regression_disagreement


def test_classification_disagreement() -> None:
    result = classification_disagreement(np.array([0.1, 0.6, 0.9]), np.array([0.2, 0.4, 0.7]))
    assert result["mean_absolute_probability_shift"] == pytest.approx(1 / 6)
    assert result["hard_classification_disagreement_rate"] == pytest.approx(1 / 3)


def test_regression_disagreement_and_constant_correlation() -> None:
    result = regression_disagreement(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]))
    assert result["prediction_correlation"] == pytest.approx(1.0)
    constant = regression_disagreement(np.ones(3), np.arange(3.0))
    assert np.isnan(constant["prediction_correlation"])
