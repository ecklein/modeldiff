"""Public comparison pipeline."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import numpy as np

from modeldiff.adapters import predict_model
from modeldiff.analysis.disagreement import (
    classification_disagreement,
    regression_disagreement,
)
from modeldiff.analysis.distributions import aggregate_metrics
from modeldiff.analysis.regions import Region, discover_regions
from modeldiff.config import ModelDiffConfig
from modeldiff.metrics.base import get_loss_function, primary_metric_value
from modeldiff.result import ModelDiffReport, RegionResult
from modeldiff.splitting import SplitStrategy, make_split
from modeldiff.statistics import benjamini_hochberg, bootstrap_mean_ci, sign_flip_test
from modeldiff.utils import normalize_X, validate_inputs


def _observation_losses(
    task: str,
    metric: str,
    y: np.ndarray,
    predictions: np.ndarray,
    config: ModelDiffConfig,
) -> np.ndarray:
    function = get_loss_function(task, metric)
    if task == "binary_classification" and metric == "log_loss":
        return function(y, predictions, epsilon=config.epsilon)
    if task == "binary_classification" and metric == "accuracy":
        return function(y, predictions, threshold=config.threshold)
    return function(y, predictions)


def _validate_regions(
    regions: list[Region],
    direction: str,
    X_validation: Any,
    baseline_loss: np.ndarray,
    candidate_loss: np.ndarray,
    config: ModelDiffConfig,
    rng: np.random.Generator,
    warnings: list[str],
) -> list[RegionResult]:
    results: list[RegionResult] = []
    delta = candidate_loss - baseline_loss
    for region in regions:
        mask = region.apply(X_validation)
        support = int(mask.sum())
        if support == 0:
            warnings.append(f"Region disappeared in validation: {region}")
            continue
        if support < 5:
            warnings.append(f"Region has fewer than five validation observations: {region}")
        region_delta = delta[mask]
        alternative: Literal["greater", "less"] = "greater" if direction == "regression" else "less"
        ci = bootstrap_mean_ci(
            region_delta,
            n_bootstrap=config.n_bootstrap,
            confidence_level=config.confidence_level,
            random_state=rng,
        )
        p_value = sign_flip_test(
            region_delta,
            n_permutations=config.n_permutations,
            alternative=alternative,
            random_state=rng,
        )
        results.append(
            RegionResult(
                conditions=region.conditions,
                direction=direction,
                discovery_support=region.discovery_support,
                discovery_fraction=region.discovery_fraction,
                validation_support=support,
                validation_fraction=support / len(X_validation),
                discovery_delta=region.discovery_delta,
                validation_delta=float(np.mean(region_delta)),
                confidence_interval=ci,
                p_value=p_value,
                q_value=1.0,
                score=region.score,
                baseline_mean_loss=float(np.mean(baseline_loss[mask])),
                candidate_mean_loss=float(np.mean(candidate_loss[mask])),
            )
        )
    return results


def compare_predictions(
    baseline_predictions: Any,
    candidate_predictions: Any,
    X: Any,
    y: Any,
    task: str,
    metric: str,
    *,
    config: ModelDiffConfig | None = None,
    random_state: int | None = None,
    validation_strategy: SplitStrategy = "random",
    groups: Any | None = None,
    time_values: Any | None = None,
    discovery_indices: Any | None = None,
    validation_indices: Any | None = None,
) -> ModelDiffReport:
    """Compare precomputed model predictions and discover validated behavior changes.

    Positive loss deltas mean the candidate is worse. Regions are discovered on
    one partition and statistically evaluated on the independent holdout. Use
    group or time splitting when random rows would leak related observations.
    """
    effective_config = config or ModelDiffConfig()
    if random_state is not None:
        effective_config = replace(effective_config, random_state=random_state)
    frame, target, baseline, candidate = validate_inputs(
        X, y, baseline_predictions, candidate_predictions, task, metric
    )
    baseline_loss = _observation_losses(task, metric, target, baseline, effective_config)
    candidate_loss = _observation_losses(task, metric, target, candidate, effective_config)
    delta = candidate_loss - baseline_loss

    baseline_primary = primary_metric_value(baseline_loss, metric)
    candidate_primary = primary_metric_value(candidate_loss, metric)
    absolute_delta = candidate_primary - baseline_primary
    relative_delta = absolute_delta / baseline_primary if baseline_primary != 0 else None
    overall = {
        "baseline": baseline_primary,
        "candidate": candidate_primary,
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
    }
    aggregates = aggregate_metrics(
        task, target, baseline, candidate, effective_config.threshold, effective_config.epsilon
    )
    if task == "binary_classification":
        diagnostics = classification_disagreement(baseline, candidate, effective_config.threshold)
    else:
        diagnostics = regression_disagreement(baseline, candidate)

    split = make_split(
        len(frame),
        strategy=validation_strategy,
        discovery_fraction=effective_config.discovery_fraction,
        random_state=effective_config.random_state,
        groups=groups,
        time_values=time_values,
        discovery_indices=discovery_indices,
        validation_indices=validation_indices,
    )
    discovery_rows = split.discovery_indices
    validation_rows = split.validation_indices
    X_discovery = frame.iloc[discovery_rows].reset_index(drop=True)
    X_validation = frame.iloc[validation_rows].reset_index(drop=True)
    discovery_delta = delta[discovery_rows]
    rng = np.random.default_rng(effective_config.random_state)

    warnings = list(split.warnings)
    if np.array_equal(baseline, candidate):
        warnings.append("Candidate predictions are identical to baseline predictions.")
    if np.std(baseline) == 0 or np.std(candidate) == 0:
        warnings.append("At least one model has constant predictions.")
    for feature in frame.columns:
        if frame[feature].isna().mean() > 0.5:
            warnings.append(f"Feature {feature!r} contains more than 50% missing values.")

    discovered_regressions = discover_regions(
        X_discovery, discovery_delta, effective_config, "regression"
    )
    discovered_improvements = discover_regions(
        X_discovery, discovery_delta, effective_config, "improvement"
    )
    validated = _validate_regions(
        discovered_regressions,
        "regression",
        X_validation,
        baseline_loss[validation_rows],
        candidate_loss[validation_rows],
        effective_config,
        rng,
        warnings,
    )
    validated += _validate_regions(
        discovered_improvements,
        "improvement",
        X_validation,
        baseline_loss[validation_rows],
        candidate_loss[validation_rows],
        effective_config,
        rng,
        warnings,
    )
    q_values = benjamini_hochberg([region.p_value for region in validated])
    validated = [
        replace(
            region,
            q_value=float(q_value),
            significant=float(q_value) <= effective_config.significance_level,
        )
        for region, q_value in zip(validated, q_values, strict=True)
    ]
    regressions = sorted(
        [
            region
            for region in validated
            if region.direction == "regression" and region.validation_delta > 0
        ],
        key=lambda region: (not region.significant, -region.validation_delta, region.q_value),
    )
    improvements = sorted(
        [
            region
            for region in validated
            if region.direction == "improvement" and region.validation_delta < 0
        ],
        key=lambda region: (not region.significant, region.validation_delta, region.q_value),
    )
    return ModelDiffReport(
        task=task,
        metric=metric,
        sample_count=len(frame),
        discovery_count=len(discovery_rows),
        validation_count=len(validation_rows),
        split=split.to_dict(),
        overall=overall,
        aggregate_metrics=aggregates,
        disagreement=diagnostics,
        regression_regions=regressions,
        improvement_regions=improvements,
        warnings=warnings,
        configuration=effective_config,
    )


def compare(
    baseline: Any,
    candidate: Any,
    X: Any,
    y: Any,
    task: str,
    metric: str,
    *,
    config: ModelDiffConfig | None = None,
    random_state: int | None = None,
    validation_strategy: SplitStrategy = "random",
    groups: Any | None = None,
    time_values: Any | None = None,
    discovery_indices: Any | None = None,
    validation_indices: Any | None = None,
) -> ModelDiffReport:
    """Compare two fitted sklearn-style models on a shared evaluation dataset."""
    frame = normalize_X(X)
    baseline_predictions = predict_model(baseline, frame, task)
    candidate_predictions = predict_model(candidate, frame, task)
    return compare_predictions(
        baseline_predictions,
        candidate_predictions,
        frame,
        y,
        task,
        metric,
        config=config,
        random_state=random_state,
        validation_strategy=validation_strategy,
        groups=groups,
        time_values=time_values,
        discovery_indices=discovery_indices,
        validation_indices=validation_indices,
    )
