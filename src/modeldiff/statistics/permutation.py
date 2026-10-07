"""Paired randomization tests."""

from __future__ import annotations

from typing import Literal

import numpy as np

Alternative = Literal["two-sided", "greater", "less"]


def sign_flip_test(
    delta: np.ndarray,
    n_permutations: int = 2000,
    alternative: Alternative = "two-sided",
    random_state: int | np.random.Generator | None = None,
) -> float:
    """Test paired mean differences by randomly flipping their signs."""
    values = np.asarray(delta, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("delta must be a non-empty one-dimensional array")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    observed = float(np.mean(values))
    extreme = 0
    batch_size = max(1, min(n_permutations, 1_000_000 // values.size))
    for start in range(0, n_permutations, batch_size):
        count = min(batch_size, n_permutations - start)
        signs = rng.integers(0, 2, size=(count, values.size), dtype=np.int8) * 2 - 1
        permuted = (signs * values).mean(axis=1)
        if alternative == "two-sided":
            extreme += int(np.sum(np.abs(permuted) >= abs(observed)))
        elif alternative == "greater":
            extreme += int(np.sum(permuted >= observed))
        else:
            extreme += int(np.sum(permuted <= observed))
    return (extreme + 1) / (n_permutations + 1)
