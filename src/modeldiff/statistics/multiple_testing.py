"""Multiple-hypothesis correction."""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values: list[float] | np.ndarray) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or np.any((values < 0) | (values > 1)):
        raise ValueError("p_values must be a one-dimensional collection in [0, 1]")
    if values.size == 0:
        return values.copy()
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * values.size / np.arange(1, values.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted
