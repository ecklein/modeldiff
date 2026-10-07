"""Interpretable condition generation and beam-search region discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from modeldiff.config import ModelDiffConfig
from modeldiff.utils import python_value

Direction = Literal["regression", "improvement", "both"]


@dataclass(frozen=True)
class Condition:
    """A single human-readable feature predicate."""

    feature: str
    operator: str
    value: Any

    def apply(self, X: pd.DataFrame) -> np.ndarray:
        """Evaluate the condition, treating missing values as non-members."""
        series = X[self.feature]
        present = series.notna().to_numpy()
        if self.operator == "<=":
            result = series.le(self.value).to_numpy()
        elif self.operator == ">":
            result = series.gt(self.value).to_numpy()
        elif self.operator == "==":
            result = series.eq(self.value).to_numpy()
        elif self.operator == "!=":
            result = series.ne(self.value).to_numpy()
        else:
            raise ValueError(f"unsupported condition operator: {self.operator}")
        return np.asarray(result, dtype=bool) & present

    def __str__(self) -> str:
        value = repr(self.value) if isinstance(self.value, str) else f"{self.value:.6g}"
        return f"{self.feature} {self.operator} {value}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "feature": self.feature,
            "operator": self.operator,
            "value": python_value(self.value),
        }


@dataclass(frozen=True)
class Region:
    """A conjunction of conditions found on discovery data."""

    conditions: tuple[Condition, ...]
    discovery_support: int
    discovery_fraction: float
    discovery_delta: float
    score: float

    def apply(self, X: pd.DataFrame) -> np.ndarray:
        """Evaluate all conditions against a feature frame."""
        mask = np.ones(len(X), dtype=bool)
        for condition in self.conditions:
            mask &= condition.apply(X)
        return mask

    def __str__(self) -> str:
        return " AND ".join(str(condition) for condition in self.conditions)


def generate_conditions(
    X: pd.DataFrame, config: ModelDiffConfig, minimum_support: int
) -> list[Condition]:
    """Generate bounded numeric and categorical candidate predicates."""
    conditions: list[Condition] = []
    n = len(X)
    maximum_support = int(np.floor(config.max_region_fraction * n))
    for feature in X.columns:
        series = X[feature]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            values = series.dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            thresholds = np.unique(np.quantile(values, config.quantiles))
            for threshold in thresholds:
                for operator in ("<=", ">"):
                    condition = Condition(feature, operator, float(threshold))
                    support = int(condition.apply(X).sum())
                    if minimum_support <= support <= maximum_support:
                        conditions.append(condition)
        else:
            counts = series.dropna().value_counts()
            for category in counts.head(config.max_categories).index:
                for operator in ("==", "!="):
                    condition = Condition(feature, operator, python_value(category))
                    support = int(condition.apply(X).sum())
                    if minimum_support <= support <= maximum_support:
                        conditions.append(condition)
    return conditions


def _direction_score(mean_delta: float, support_fraction: float, direction: Direction) -> float:
    if direction == "regression":
        effect = mean_delta
    elif direction == "improvement":
        effect = -mean_delta
    else:
        effect = abs(mean_delta)
    return max(effect, 0.0) * float(np.sqrt(support_fraction))


def _region_key(conditions: tuple[Condition, ...]) -> tuple[str, ...]:
    return tuple(sorted(str(condition) for condition in conditions))


def discover_regions(
    X: pd.DataFrame,
    delta: np.ndarray,
    config: ModelDiffConfig,
    direction: Direction = "both",
) -> list[Region]:
    """Find high-effect regions using deterministic breadth-limited beam search."""
    n = len(X)
    minimum_support = max(config.min_region_size or 0, int(np.ceil(config.min_region_fraction * n)))
    candidates = generate_conditions(X, config, minimum_support)
    candidate_masks = {condition: condition.apply(X) for condition in candidates}
    beam: list[tuple[Region, np.ndarray]] = []
    all_regions: list[tuple[Region, np.ndarray]] = []
    seen: set[tuple[str, ...]] = set()

    for depth in range(1, config.max_depth + 1):
        expanded: list[tuple[Region, np.ndarray]] = []
        parents: list[tuple[tuple[Condition, ...], np.ndarray]]
        if depth == 1:
            parents = [((), np.ones(n, dtype=bool))]
        else:
            parents = [(region.conditions, mask) for region, mask in beam]
        for parent_conditions, parent_mask in parents:
            used_features = {condition.feature for condition in parent_conditions}
            for condition in candidates:
                # One predicate per feature is simple, readable, and avoids contradictions.
                if condition.feature in used_features:
                    continue
                conditions = (*parent_conditions, condition)
                key = _region_key(conditions)
                if key in seen:
                    continue
                seen.add(key)
                mask = parent_mask & candidate_masks[condition]
                support = int(mask.sum())
                fraction = support / n
                if support < minimum_support or fraction > config.max_region_fraction:
                    continue
                mean_delta = float(np.mean(delta[mask]))
                score = _direction_score(mean_delta, fraction, direction)
                if score <= 0:
                    continue
                expanded.append(
                    (
                        Region(conditions, support, fraction, mean_delta, score),
                        mask,
                    )
                )
        expanded.sort(
            key=lambda item: (
                -item[0].score,
                -item[0].discovery_support,
                len(item[0].conditions),
                str(item[0]),
            )
        )
        beam = expanded[: config.beam_width]
        all_regions.extend(beam)
        if not beam:
            break

    return deduplicate_regions(all_regions, config.jaccard_threshold, config.max_regions)


def jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Return Jaccard similarity between two boolean membership masks."""
    union = int(np.sum(mask_a | mask_b))
    return float(np.sum(mask_a & mask_b) / union) if union else 1.0


def deduplicate_regions(
    regions_with_masks: list[tuple[Region, np.ndarray]], threshold: float, maximum: int
) -> list[Region]:
    """Keep the strongest region among highly overlapping candidates."""
    ranked = sorted(
        regions_with_masks,
        key=lambda item: (
            -item[0].score,
            -item[0].discovery_support,
            len(item[0].conditions),
            str(item[0]),
        ),
    )
    selected: list[tuple[Region, np.ndarray]] = []
    for region, mask in ranked:
        if any(jaccard(mask, existing_mask) > threshold for _, existing_mask in selected):
            continue
        selected.append((region, mask))
        if len(selected) >= maximum:
            break
    return [region for region, _ in selected]
