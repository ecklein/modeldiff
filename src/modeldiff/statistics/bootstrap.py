"""Bootstrap confidence intervals."""

from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int | np.random.Generator | None = None,
) -> tuple[float, float]:
    """Estimate a percentile confidence interval for a sample mean."""
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or sample.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    means = np.empty(n_bootstrap, dtype=float)
    # Batches cap memory for large validation regions.
    batch_size = max(1, min(n_bootstrap, 1_000_000 // sample.size))
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        indices = rng.integers(0, sample.size, size=(stop - start, sample.size))
        means[start:stop] = sample[indices].mean(axis=1)
    alpha = 1 - confidence_level
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)
