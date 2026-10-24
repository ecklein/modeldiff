"""Stability-selected region discovery on an untouched validation holdout."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from modeldiff.adapters import predict_model
from modeldiff.analysis.regions import Condition, Region, discover_regions, jaccard
from modeldiff.api import _validate_regions, compare_predictions
from modeldiff.checks import CheckSuite
from modeldiff.config import ModelDiffConfig, StabilityConfig
from modeldiff.metrics.base import observation_losses
from modeldiff.result import RegionResult, StabilityReport, StableRegionResult
from modeldiff.splitting import SplitStrategy, make_split
from modeldiff.statistics import benjamini_hochberg
from modeldiff.utils import normalize_X, validate_inputs


@dataclass
class _Occurrence:
    region: Region
    mask: np.ndarray
    resample: int


def _resample_rows(
    discovery_rows: np.ndarray,
    strategy: SplitStrategy,
    fraction: float,
    rng: np.random.Generator,
    *,
    groups: Any | None,
    time_values: Any | None,
) -> np.ndarray:
    """Sample discovery rows while respecting the outer split's structure."""
    if strategy == "group":
        labels = np.asarray(groups, dtype=object)[discovery_rows]
        unique = np.unique(labels)
        count = max(1, int(np.ceil(fraction * len(unique))))
        chosen = rng.choice(unique, size=count, replace=False)
        return discovery_rows[np.isin(labels, chosen)]
    count = max(1, int(np.ceil(fraction * len(discovery_rows))))
    if strategy == "time":
        values = np.asarray(time_values, dtype=object)[discovery_rows]
        order = np.argsort(values, kind="stable")
        start = int(rng.integers(0, len(order) - count + 1))
        return discovery_rows[order[start : start + count]]
    return np.sort(rng.choice(discovery_rows, size=count, replace=False))


def _cluster_occurrences(
    occurrences: list[_Occurrence], threshold: float
) -> list[list[_Occurrence]]:
    clusters: list[list[_Occurrence]] = []
    for occurrence in sorted(
        occurrences,
        key=lambda item: (-item.region.score, item.resample, str(item.region)),
    ):
        best_index: int | None = None
        best_similarity = -1.0
        for index, cluster in enumerate(clusters):
            similarity = max(jaccard(occurrence.mask, item.mask) for item in cluster)
            if similarity >= threshold and similarity > best_similarity:
                best_index = index
                best_similarity = similarity
        if best_index is None:
            clusters.append([occurrence])
        else:
            clusters[best_index].append(occurrence)
    return clusters


def _stable_candidates(
    X: Any,
    delta: np.ndarray,
    discovery_rows: np.ndarray,
    direction: Literal["regression", "improvement"],
    config: ModelDiffConfig,
    stability: StabilityConfig,
    strategy: SplitStrategy,
    *,
    groups: Any | None,
    time_values: Any | None,
) -> list[tuple[Region, dict[str, Any]]]:
    rng = np.random.default_rng(stability.random_state)
    full_X = X.iloc[discovery_rows].reset_index(drop=True)
    occurrences: list[_Occurrence] = []
    for resample in range(stability.n_resamples):
        sampled_rows = _resample_rows(
            discovery_rows,
            strategy,
            stability.subsample_fraction,
            rng,
            groups=groups,
            time_values=time_values,
        )
        sampled_X = X.iloc[sampled_rows].reset_index(drop=True)
        sampled_delta = delta[sampled_rows]
        for region in discover_regions(sampled_X, sampled_delta, config, direction):
            mask = region.apply(full_X)
            if mask.any():
                occurrences.append(_Occurrence(region, mask, resample))

    stable: list[tuple[Region, dict[str, Any]]] = []
    for cluster in _cluster_occurrences(occurrences, stability.cluster_jaccard_threshold):
        by_resample: dict[int, _Occurrence] = {}
        for item in cluster:
            current = by_resample.get(item.resample)
            if current is None or item.region.score > current.region.score:
                by_resample[item.resample] = item
        cluster = list(by_resample.values())
        selected_resamples = len(cluster)
        frequency = selected_resamples / stability.n_resamples
        if frequency < stability.min_selection_frequency:
            continue
        similarities = [
            np.mean([jaccard(item.mask, other.mask) for other in cluster]) for item in cluster
        ]
        medoid_index = int(np.argmax(similarities))
        medoid = cluster[medoid_index]
        mask = medoid.mask
        support = int(mask.sum())
        full_delta = float(np.mean(delta[discovery_rows][mask]))
        resample_deltas = [item.region.discovery_delta for item in cluster]
        resample_supports = [item.region.discovery_support for item in cluster]
        desired_signs = [
            value > 0 if direction == "regression" else value < 0 for value in resample_deltas
        ]
        region = Region(
            conditions=medoid.region.conditions,
            discovery_support=support,
            discovery_fraction=support / len(discovery_rows),
            discovery_delta=full_delta,
            score=medoid.region.score,
        )
        stable.append(
            (
                region,
                {
                    "selection_frequency": frequency,
                    "selected_resamples": selected_resamples,
                    "total_resamples": stability.n_resamples,
                    "sign_consistency": float(np.mean(desired_signs)),
                    "mean_cluster_similarity": float(similarities[medoid_index]),
                    "discovery_delta_range": (
                        float(min(resample_deltas)),
                        float(max(resample_deltas)),
                    ),
                    "discovery_support_range": (
                        int(min(resample_supports)),
                        int(max(resample_supports)),
                    ),
                },
            )
        )
    stable.sort(key=lambda item: (-item[1]["selection_frequency"], -item[0].score, str(item[0])))
    return stable[: config.max_regions]


def _triage(
    region: Any,
    X_validation: Any,
    known_checks: CheckSuite | None,
    threshold: float,
) -> tuple[str, float, tuple[str, ...]]:
    if known_checks is None:
        return "unclassified", 0.0, ()
    region_mask = _conditions_mask(X_validation, region.conditions)
    matched: list[str] = []
    covered = np.zeros(len(X_validation), dtype=bool)
    for check in known_checks.checks:
        check_mask = _conditions_mask(X_validation, check.conditions)
        overlap = region_mask & check_mask
        if overlap.any():
            matched.append(check.check_id)
            covered |= overlap
    coverage = float(int(covered.sum()) / int(region_mask.sum())) if region_mask.any() else 0.0
    status = "known" if coverage >= threshold else "partially_covered" if coverage > 0 else "new"
    return status, coverage, tuple(matched)


def _conditions_mask(X: Any, conditions: tuple[Condition, ...]) -> np.ndarray:
    mask = np.ones(len(X), dtype=bool)
    for condition in conditions:
        mask &= condition.apply(X)
    return mask


def compare_stable_predictions(
    baseline_predictions: Any,
    candidate_predictions: Any,
    X: Any,
    y: Any,
    task: str,
    metric: str,
    *,
    config: ModelDiffConfig | None = None,
    stability_config: StabilityConfig | None = None,
    known_checks: CheckSuite | str | Path | None = None,
    validation_strategy: SplitStrategy = "random",
    groups: Any | None = None,
    time_values: Any | None = None,
    discovery_indices: Any | None = None,
    validation_indices: Any | None = None,
) -> StabilityReport:
    """Discover repeatable regions, then validate them once on untouched data."""
    effective = config or ModelDiffConfig()
    stability = stability_config or StabilityConfig(random_state=effective.random_state)
    frame, target, baseline, candidate = validate_inputs(
        X, y, baseline_predictions, candidate_predictions, task, metric
    )
    split = make_split(
        len(frame),
        strategy=validation_strategy,
        discovery_fraction=effective.discovery_fraction,
        random_state=effective.random_state,
        groups=groups,
        time_values=time_values,
        discovery_indices=discovery_indices,
        validation_indices=validation_indices,
    )
    base = compare_predictions(
        baseline,
        candidate,
        frame,
        target,
        task,
        metric,
        config=effective,
        validation_strategy="explicit",
        discovery_indices=split.discovery_indices,
        validation_indices=split.validation_indices,
    )
    base.split = split.to_dict()
    base.warnings = list(dict.fromkeys([*split.warnings, *base.warnings]))
    baseline_loss = observation_losses(
        task, metric, target, baseline, threshold=effective.threshold, epsilon=effective.epsilon
    )
    candidate_loss = observation_losses(
        task, metric, target, candidate, threshold=effective.threshold, epsilon=effective.epsilon
    )
    delta = candidate_loss - baseline_loss
    suite = CheckSuite.load(known_checks) if isinstance(known_checks, (str, Path)) else known_checks
    if suite is not None and (suite.task != task or suite.metric != metric):
        raise ValueError("known_checks must use the same task and metric as the comparison")
    if suite is not None:
        missing_features = sorted(
            {
                condition.feature
                for check in suite.checks
                for condition in check.conditions
                if condition.feature not in frame.columns
            }
        )
        if missing_features:
            raise ValueError(
                "known_checks require missing feature columns: " + ", ".join(missing_features)
            )

    selected_regressions = _stable_candidates(
        frame,
        delta,
        split.discovery_indices,
        "regression",
        effective,
        stability,
        validation_strategy,
        groups=groups,
        time_values=time_values,
    )
    selected_improvements = _stable_candidates(
        frame,
        delta,
        split.discovery_indices,
        "improvement",
        effective,
        stability,
        validation_strategy,
        groups=groups,
        time_values=time_values,
    )
    X_validation = frame.iloc[split.validation_indices].reset_index(drop=True)
    rng = np.random.default_rng(effective.random_state)
    warnings = base.warnings
    validated_regressions = _validate_regions(
        [region for region, _ in selected_regressions],
        "regression",
        X_validation,
        baseline_loss[split.validation_indices],
        candidate_loss[split.validation_indices],
        effective,
        rng,
        warnings,
    )
    validated_improvements = _validate_regions(
        [region for region, _ in selected_improvements],
        "improvement",
        X_validation,
        baseline_loss[split.validation_indices],
        candidate_loss[split.validation_indices],
        effective,
        rng,
        warnings,
    )
    validated = [*validated_regressions, *validated_improvements]
    q_values = benjamini_hochberg([region.p_value for region in validated])
    validated = [
        replace(
            region,
            q_value=float(q),
            significant=float(q) <= effective.significance_level,
        )
        for region, q in zip(validated, q_values, strict=True)
    ]
    metadata = {
        (direction, tuple(str(condition) for condition in region.conditions)): details
        for direction, selected in (
            ("regression", selected_regressions),
            ("improvement", selected_improvements),
        )
        for region, details in selected
    }
    stable_results: list[StableRegionResult] = []
    for region in validated:
        details = metadata[
            (region.direction, tuple(str(condition) for condition in region.conditions))
        ]
        triage, coverage, matched = _triage(
            region, X_validation, suite, stability.known_coverage_threshold
        )
        stable_results.append(
            StableRegionResult(
                **{name: getattr(region, name) for name in region.__dataclass_fields__},
                **details,
                triage=triage,
                known_coverage=coverage,
                matched_check_ids=matched,
            )
        )
    regressions = sorted(
        [
            item
            for item in stable_results
            if item.direction == "regression" and item.validation_delta > 0
        ],
        key=lambda item: (not item.significant, -item.selection_frequency, -item.validation_delta),
    )
    improvements = sorted(
        [
            item
            for item in stable_results
            if item.direction == "improvement" and item.validation_delta < 0
        ],
        key=lambda item: (not item.significant, -item.selection_frequency, item.validation_delta),
    )
    if not regressions and not improvements:
        warnings.append("No regions met the configured stability-selection threshold.")
    return StabilityReport(
        task=base.task,
        metric=base.metric,
        sample_count=base.sample_count,
        discovery_count=base.discovery_count,
        validation_count=base.validation_count,
        overall=base.overall,
        aggregate_metrics=base.aggregate_metrics,
        disagreement=base.disagreement,
        split=base.split,
        regression_regions=cast(list[RegionResult], regressions),
        improvement_regions=cast(list[RegionResult], improvements),
        warnings=warnings,
        configuration=effective,
        stability_configuration=stability,
    )


def compare_stable(
    baseline: Any,
    candidate: Any,
    X: Any,
    y: Any,
    task: str,
    metric: str,
    **kwargs: Any,
) -> StabilityReport:
    """Compare fitted models using stability-selected region discovery."""
    frame = normalize_X(X)
    return compare_stable_predictions(
        predict_model(baseline, frame, task),
        predict_model(candidate, frame, task),
        frame,
        y,
        task,
        metric,
        **kwargs,
    )
