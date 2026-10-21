"""Leakage-aware discovery and validation splitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

SplitStrategy = Literal["random", "group", "time", "explicit"]


@dataclass(frozen=True)
class SplitResult:
    """Indices and audit metadata for a discovery/validation partition."""

    discovery_indices: np.ndarray
    validation_indices: np.ndarray
    strategy: SplitStrategy
    requested_discovery_fraction: float | None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return compact, JSON-compatible split metadata (without row indices)."""
        total = len(self.discovery_indices) + len(self.validation_indices)
        return {
            "strategy": self.strategy,
            "requested_discovery_fraction": self.requested_discovery_fraction,
            "actual_discovery_fraction": len(self.discovery_indices) / total,
            "discovery_count": len(self.discovery_indices),
            "validation_count": len(self.validation_indices),
            **self.metadata,
        }


def _check_length(values: Any, name: str, n_samples: int) -> np.ndarray:
    # Object dtype preserves heterogeneous Python inputs so invalid mixtures
    # (for example integers and strings used as time values) are not silently
    # coerced into a lexicographically sortable string array.
    array = np.asarray(values, dtype=object)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(array) != n_samples:
        raise ValueError(f"{name} must contain exactly {n_samples} values")
    return array


def _partition_warnings(
    discovery_count: int,
    validation_count: int,
    requested_fraction: float | None,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if discovery_count < 30:
        warnings.append(
            "Discovery set has fewer than 30 observations; region search may be unstable."
        )
    if validation_count < 30:
        warnings.append("Validation set has fewer than 30 observations; inference may be unstable.")
    if requested_fraction is not None:
        actual = discovery_count / (discovery_count + validation_count)
        if abs(actual - requested_fraction) > 0.1:
            warnings.append(
                "The achieved discovery fraction differs from the requested fraction by more "
                "than 10 percentage points."
            )
    return tuple(warnings)


def _result(
    discovery: np.ndarray,
    validation: np.ndarray,
    strategy: SplitStrategy,
    requested_fraction: float | None,
    metadata: dict[str, Any] | None = None,
) -> SplitResult:
    if len(discovery) == 0 or len(validation) == 0:
        raise ValueError("both discovery and validation partitions must contain observations")
    return SplitResult(
        discovery_indices=np.asarray(discovery, dtype=np.intp),
        validation_indices=np.asarray(validation, dtype=np.intp),
        strategy=strategy,
        requested_discovery_fraction=requested_fraction,
        warnings=_partition_warnings(len(discovery), len(validation), requested_fraction),
        metadata=metadata or {},
    )


def _best_boundary(cumulative_counts: np.ndarray, target: int) -> int:
    """Return a non-empty boundary closest to the requested row count."""
    candidates = cumulative_counts[:-1]
    if len(candidates) == 0:
        raise ValueError("at least two distinct groups are required to create a split")
    return int(np.argmin(np.abs(candidates - target))) + 1


def _normalize_explicit_indices(values: Any, name: str, n_samples: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional index array or boolean mask")
    if np.issubdtype(array.dtype, np.bool_):
        if len(array) != n_samples:
            raise ValueError(f"boolean {name} must contain exactly {n_samples} values")
        return np.flatnonzero(array)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must contain integer indices or boolean mask values")
    normalized = array.astype(np.intp, copy=False)
    if np.any((normalized < 0) | (normalized >= n_samples)):
        raise ValueError(f"{name} contains an index outside the range [0, {n_samples})")
    if len(np.unique(normalized)) != len(normalized):
        raise ValueError(f"{name} contains duplicate indices")
    return normalized


def make_split(
    n_samples: int,
    *,
    strategy: SplitStrategy = "random",
    discovery_fraction: float = 0.5,
    random_state: int = 42,
    groups: Any | None = None,
    time_values: Any | None = None,
    discovery_indices: Any | None = None,
    validation_indices: Any | None = None,
) -> SplitResult:
    """Create a validated discovery/validation split.

    Group splits keep every group entirely in one partition. Time splits place
    earlier observations in discovery and later observations in validation,
    while keeping equal timestamps together. Explicit splits accept integer
    index arrays or full-length boolean masks and must partition every row.
    """
    if n_samples < 2:
        raise ValueError("at least two observations are required to create a split")
    if not 0 < discovery_fraction < 1:
        raise ValueError("discovery_fraction must be between 0 and 1")
    if strategy not in {"random", "group", "time", "explicit"}:
        raise ValueError("strategy must be one of: random, group, time, explicit")

    supplied = {
        "groups": groups is not None,
        "time_values": time_values is not None,
        "discovery_indices": discovery_indices is not None,
        "validation_indices": validation_indices is not None,
    }
    allowed = {
        "random": set(),
        "group": {"groups"},
        "time": {"time_values"},
        "explicit": {"discovery_indices", "validation_indices"},
    }[strategy]
    unexpected = [name for name, present in supplied.items() if present and name not in allowed]
    if unexpected:
        raise ValueError(f"{', '.join(unexpected)} cannot be used with the {strategy!r} strategy")

    target = int(np.floor(discovery_fraction * n_samples))
    if strategy == "random":
        order = np.random.default_rng(random_state).permutation(n_samples)
        return _result(
            order[:target],
            order[target:],
            strategy,
            discovery_fraction,
            {"random_state": random_state},
        )

    if strategy == "group":
        if groups is None:
            raise ValueError("groups are required for the 'group' strategy")
        group_array = _check_length(groups, "groups", n_samples)
        if bool(np.asarray(pd.isna(group_array)).any()):
            raise ValueError("groups cannot contain missing values")
        codes, unique_groups = pd.factorize(group_array, sort=False)
        if len(unique_groups) < 2:
            raise ValueError("at least two distinct groups are required to create a split")
        counts = np.bincount(codes, minlength=len(unique_groups))
        group_order = np.random.default_rng(random_state).permutation(len(unique_groups))
        boundary = _best_boundary(np.cumsum(counts[group_order]), target)
        discovery_group_codes = group_order[:boundary]
        discovery_mask = np.isin(codes, discovery_group_codes)
        return _result(
            np.flatnonzero(discovery_mask),
            np.flatnonzero(~discovery_mask),
            strategy,
            discovery_fraction,
            {"random_state": random_state, "group_count": len(unique_groups)},
        )

    if strategy == "time":
        if time_values is None:
            raise ValueError("time_values are required for the 'time' strategy")
        time_array = _check_length(time_values, "time_values", n_samples)
        if bool(np.asarray(pd.isna(time_array)).any()):
            raise ValueError("time_values cannot contain missing values")
        series = pd.Series(time_array)
        try:
            order = series.sort_values(kind="mergesort").index.to_numpy(dtype=np.intp)
        except (TypeError, ValueError) as error:
            raise ValueError("time_values must be mutually orderable") from error
        ordered_values = series.iloc[order].reset_index(drop=True)
        distinct_boundaries = np.flatnonzero(
            ordered_values.iloc[1:].to_numpy() != ordered_values.iloc[:-1].to_numpy()
        ) + 1
        if len(distinct_boundaries) == 0:
            raise ValueError("at least two distinct time_values are required to create a split")
        boundary = int(distinct_boundaries[np.argmin(np.abs(distinct_boundaries - target))])
        return _result(
            order[:boundary],
            order[boundary:],
            strategy,
            discovery_fraction,
            {"time_order": "ascending", "distinct_time_count": len(distinct_boundaries) + 1},
        )

    if discovery_indices is None or validation_indices is None:
        raise ValueError(
            "discovery_indices and validation_indices are required for the 'explicit' strategy"
        )
    discovery = _normalize_explicit_indices(
        discovery_indices, "discovery_indices", n_samples
    )
    validation = _normalize_explicit_indices(
        validation_indices, "validation_indices", n_samples
    )
    if np.intersect1d(discovery, validation).size:
        raise ValueError("discovery_indices and validation_indices cannot overlap")
    if len(discovery) + len(validation) != n_samples:
        raise ValueError("explicit indices must assign every observation to exactly one partition")
    return _result(discovery, validation, strategy, None)
