from __future__ import annotations

import numpy as np
import pytest

from modeldiff.statistics import benjamini_hochberg, bootstrap_mean_ci, sign_flip_test


def test_bootstrap_mean_ci_is_deterministic_and_sensible() -> None:
    values = np.arange(1.0, 101.0)
    first = bootstrap_mean_ci(values, 1000, random_state=10)
    second = bootstrap_mean_ci(values, 1000, random_state=10)
    assert first == second
    assert first[0] < np.mean(values) < first[1]


def test_sign_flip_detects_positive_and_not_centered_noise() -> None:
    rng = np.random.default_rng(4)
    positive = rng.normal(1, 0.2, size=80)
    assert sign_flip_test(positive, 999, "greater", random_state=1) < 0.01
    assert sign_flip_test(positive, 999, "less", random_state=1) > 0.9
    centered = np.array([-1.0, 1.0] * 30)
    assert sign_flip_test(centered, 499, "two-sided", random_state=1) == 1.0


def test_benjamini_hochberg_known_values_and_validation() -> None:
    actual = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])
    assert actual == pytest.approx([0.02, 0.04, 0.04, 0.008])
    assert benjamini_hochberg([]).size == 0
    with pytest.raises(ValueError):
        benjamini_hochberg([1.2])
