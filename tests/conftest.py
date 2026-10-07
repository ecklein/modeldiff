from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeldiff import ModelDiffConfig


@pytest.fixture
def fast_config() -> ModelDiffConfig:
    return ModelDiffConfig(
        max_depth=2,
        beam_width=15,
        max_regions=5,
        min_region_fraction=0.04,
        n_bootstrap=120,
        n_permutations=199,
        random_state=42,
    )


@pytest.fixture
def planted_classification() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    n = 2400
    X = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "segment": rng.choice(["a", "b", "c"], size=n),
        }
    )
    latent = 0.7 * X["x1"].to_numpy() - 0.4 * X["x2"].to_numpy()
    true_probability = 1 / (1 + np.exp(-latent))
    y = rng.binomial(1, true_probability).astype(float)
    baseline = np.clip(true_probability + rng.normal(0, 0.08, n), 0.03, 0.97)
    candidate = baseline + np.where(y == 1, 0.08, -0.08)
    planted = (X["x1"].to_numpy() > 1) & (X["x2"].to_numpy() < 0)
    candidate[planted] = np.where(y[planted] == 1, 0.12, 0.88)
    return X, y, baseline, np.clip(candidate, 0.01, 0.99)
