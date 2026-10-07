from __future__ import annotations

import numpy as np
import pandas as pd

from modeldiff import ModelDiffConfig
from modeldiff.analysis.regions import (
    Condition,
    Region,
    deduplicate_regions,
    discover_regions,
    generate_conditions,
)


def test_numeric_and_categorical_conditions_treat_missing_as_false() -> None:
    X = pd.DataFrame({"number": [1.0, 2.0, np.nan], "kind": ["a", "b", None]})
    assert Condition("number", "<=", 1.5).apply(X).tolist() == [True, False, False]
    assert Condition("number", ">", 1.5).apply(X).tolist() == [False, True, False]
    assert Condition("kind", "==", "a").apply(X).tolist() == [True, False, False]
    assert Condition("kind", "!=", "a").apply(X).tolist() == [False, True, False]


def test_generate_conditions_respects_support_and_category_limit() -> None:
    X = pd.DataFrame({"x": np.arange(100), "category": list("abcdefghij") * 10})
    config = ModelDiffConfig(max_categories=2, max_region_fraction=0.9)
    conditions = generate_conditions(X, config, minimum_support=5)
    categorical_equalities = [
        c for c in conditions if c.feature == "category" and c.operator == "=="
    ]
    assert len(categorical_equalities) == 2
    assert all(5 <= condition.apply(X).sum() <= 90 for condition in conditions)


def test_beam_search_finds_multicondition_and_is_deterministic() -> None:
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.normal(size=2000), "b": rng.normal(size=2000)})
    delta = rng.normal(0, 0.05, size=2000)
    planted = (X["a"].to_numpy() > 0.8) & (X["b"].to_numpy() < -0.2)
    delta[planted] += 1.0
    config = ModelDiffConfig(max_depth=2, min_region_fraction=0.02, max_regions=8)
    first = discover_regions(X, delta, config, "regression")
    second = discover_regions(X, delta, config, "regression")
    assert first == second
    assert any(len(region.conditions) == 2 for region in first)
    assert all(
        len({c.feature for c in region.conditions}) == len(region.conditions) for region in first
    )
    best_mask = first[0].apply(X)
    overlap = (best_mask & planted).sum() / (best_mask | planted).sum()
    assert overlap > 0.35


def test_depth_and_overlap_deduplication() -> None:
    X = pd.DataFrame({"x": np.arange(100), "y": np.arange(100)})
    delta = np.where(X["x"] > 70, 1.0, 0.0)
    config = ModelDiffConfig(max_depth=1, max_region_fraction=0.5)
    assert all(len(region.conditions) == 1 for region in discover_regions(X, delta, config))
    a = Region((Condition("x", ">", 50),), 49, 0.49, 1.0, 1.0)
    b = Region((Condition("x", ">", 51),), 48, 0.48, 0.9, 0.9)
    selected = deduplicate_regions([(a, a.apply(X)), (b, b.apply(X))], 0.8, 10)
    assert selected == [a]
