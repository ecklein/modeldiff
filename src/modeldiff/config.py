"""Configuration for ModelDiff-ML analyses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelDiffConfig:
    """Controls region discovery and independent statistical validation."""

    discovery_fraction: float = 0.5
    max_depth: int = 3
    beam_width: int = 25
    max_regions: int = 10
    min_region_size: int | None = None
    min_region_fraction: float = 0.02
    max_region_fraction: float = 0.50
    quantiles: tuple[float, ...] = tuple(i / 10 for i in range(1, 10))
    max_categories: int = 20
    jaccard_threshold: float = 0.80
    n_bootstrap: int = 1000
    n_permutations: int = 2000
    confidence_level: float = 0.95
    significance_level: float = 0.05
    random_state: int = 42
    threshold: float = 0.5
    epsilon: float = 1e-15

    def __post_init__(self) -> None:
        if not 0 < self.discovery_fraction < 1:
            raise ValueError("discovery_fraction must be between 0 and 1")
        if self.max_depth < 1 or self.beam_width < 1 or self.max_regions < 1:
            raise ValueError("max_depth, beam_width, and max_regions must be positive")
        if self.min_region_size is not None and self.min_region_size < 1:
            raise ValueError("min_region_size must be positive")
        if not 0 <= self.min_region_fraction <= self.max_region_fraction <= 1:
            raise ValueError("region fractions must satisfy 0 <= min <= max <= 1")
        if any(not 0 < q < 1 for q in self.quantiles):
            raise ValueError("quantiles must be between 0 and 1")
        if not 0 <= self.jaccard_threshold <= 1:
            raise ValueError("jaccard_threshold must be between 0 and 1")
        if self.n_bootstrap < 1 or self.n_permutations < 1:
            raise ValueError("resampling counts must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        if not 0 < self.significance_level < 1:
            raise ValueError("significance_level must be between 0 and 1")
        if not 0 <= self.threshold <= 1 or not 0 < self.epsilon < 0.5:
            raise ValueError("threshold and epsilon are invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration mapping."""
        data = asdict(self)
        data["quantiles"] = list(self.quantiles)
        return data


@dataclass(frozen=True)
class StabilityConfig:
    """Controls repeated discovery and stable-region selection.

    Resampling is confined to the discovery partition. The untouched validation
    partition is used once, after stable regions have been selected.
    """

    n_resamples: int = 30
    subsample_fraction: float = 0.75
    min_selection_frequency: float = 0.50
    cluster_jaccard_threshold: float = 0.70
    known_coverage_threshold: float = 0.80
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_resamples < 2:
            raise ValueError("n_resamples must be at least 2")
        if not 0 < self.subsample_fraction <= 1:
            raise ValueError("subsample_fraction must be between 0 and 1")
        if not 0 < self.min_selection_frequency <= 1:
            raise ValueError("min_selection_frequency must be between 0 and 1")
        if not 0 <= self.cluster_jaccard_threshold <= 1:
            raise ValueError("cluster_jaccard_threshold must be between 0 and 1")
        if not 0 <= self.known_coverage_threshold <= 1:
            raise ValueError("known_coverage_threshold must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible configuration mapping."""
        return asdict(self)
