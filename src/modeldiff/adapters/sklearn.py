"""Small model protocol adapter compatible with sklearn-style estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def predict_model(model: Any, X: pd.DataFrame, task: str) -> np.ndarray:
    """Obtain one-dimensional predictions without transforming model scores."""
    if task == "regression":
        if not hasattr(model, "predict"):
            raise TypeError("regression models must provide predict(X)")
        predictions = np.asarray(model.predict(X))
    else:
        if not hasattr(model, "predict_proba"):
            if hasattr(model, "decision_function"):
                raise TypeError(
                    "probability-based classification metrics require predict_proba(X); "
                    "decision_function scores are not silently transformed"
                )
            raise TypeError("binary classification models must provide predict_proba(X)")
        probabilities = np.asarray(model.predict_proba(X))
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError("predict_proba(X) must return shape (n_samples, 2)")
        predictions = probabilities[:, 1]
    if predictions.ndim != 1:
        raise ValueError("model predictions must be one-dimensional")
    return predictions.astype(float)
