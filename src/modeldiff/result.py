"""Serializable result objects for ModelDiff-ML."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from modeldiff.analysis.regions import Condition
from modeldiff.config import ModelDiffConfig, StabilityConfig

if TYPE_CHECKING:
    from modeldiff.checks import CheckSuite


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    return value


@dataclass(frozen=True)
class RegionResult:
    """A discovered region evaluated on the independent validation set."""

    conditions: tuple[Condition, ...]
    direction: str
    discovery_support: int
    discovery_fraction: float
    validation_support: int
    validation_fraction: float
    discovery_delta: float
    validation_delta: float
    confidence_interval: tuple[float, float]
    p_value: float
    q_value: float
    score: float
    baseline_mean_loss: float
    candidate_mean_loss: float
    significant: bool = False

    @property
    def description(self) -> str:
        """Return a concise textual conjunction."""
        return " AND ".join(str(condition) for condition in self.conditions)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible region mapping."""
        return {
            "conditions": [condition.to_dict() for condition in self.conditions],
            "description": self.description,
            "direction": self.direction,
            "discovery_support": self.discovery_support,
            "discovery_fraction": self.discovery_fraction,
            "validation_support": self.validation_support,
            "validation_fraction": self.validation_fraction,
            "discovery_delta": self.discovery_delta,
            "validation_delta": self.validation_delta,
            "confidence_interval": list(self.confidence_interval),
            "p_value": self.p_value,
            "q_value": self.q_value,
            "score": self.score,
            "baseline_mean_loss": self.baseline_mean_loss,
            "candidate_mean_loss": self.candidate_mean_loss,
            "significant": self.significant,
        }


@dataclass(frozen=True)
class StableRegionResult(RegionResult):
    """A validated region that repeatedly appeared during discovery resampling."""

    selection_frequency: float = 0.0
    selected_resamples: int = 0
    total_resamples: int = 0
    sign_consistency: float = 0.0
    mean_cluster_similarity: float = 0.0
    discovery_delta_range: tuple[float, float] = (0.0, 0.0)
    discovery_support_range: tuple[int, int] = (0, 0)
    triage: str = "unclassified"
    known_coverage: float = 0.0
    matched_check_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["stability"] = {
            "selection_frequency": self.selection_frequency,
            "selected_resamples": self.selected_resamples,
            "total_resamples": self.total_resamples,
            "sign_consistency": self.sign_consistency,
            "mean_cluster_similarity": self.mean_cluster_similarity,
            "discovery_delta_range": list(self.discovery_delta_range),
            "discovery_support_range": list(self.discovery_support_range),
        }
        data["triage"] = {
            "status": self.triage,
            "known_coverage": self.known_coverage,
            "matched_check_ids": list(self.matched_check_ids),
        }
        return data


@dataclass
class ModelDiffReport:
    """Complete, model-independent output of a ModelDiff-ML comparison."""

    task: str
    metric: str
    sample_count: int
    discovery_count: int
    validation_count: int
    overall: dict[str, Any]
    aggregate_metrics: dict[str, dict[str, float]]
    disagreement: dict[str, float]
    split: dict[str, Any] = field(default_factory=dict)
    regression_regions: list[RegionResult] = field(default_factory=list)
    improvement_regions: list[RegionResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    configuration: ModelDiffConfig = field(default_factory=ModelDiffConfig)
    schema_version: str = "1.0"

    def regressions(self, *, significant_only: bool = False) -> list[RegionResult]:
        """Return validated regression regions, optionally requiring q <= alpha."""
        if significant_only:
            return [region for region in self.regression_regions if region.significant]
        return list(self.regression_regions)

    def improvements(self, *, significant_only: bool = False) -> list[RegionResult]:
        """Return validated improvement regions, optionally requiring q <= alpha."""
        if significant_only:
            return [region for region in self.improvement_regions if region.significant]
        return list(self.improvement_regions)

    def export_checks(
        self,
        regions: Sequence[RegionResult] | None = None,
        *,
        max_loss_increase: float = 0.0,
        min_samples: int = 30,
        names: Sequence[str] | None = None,
    ) -> CheckSuite:
        """Export selected regression regions as a reusable, fixed-condition suite.

        When ``regions`` is omitted, statistically significant regression regions
        are exported. Passing an explicit empty sequence creates a valid empty suite.
        """
        from modeldiff.checks import checks_from_regions

        selected = self.regressions(significant_only=True) if regions is None else regions
        return checks_from_regions(
            task=self.task,
            metric=self.metric,
            threshold=self.configuration.threshold,
            epsilon=self.configuration.epsilon,
            report_schema_version=self.schema_version,
            regions=selected,
            max_loss_increase=max_loss_increase,
            min_samples=min_samples,
            names=names,
        )

    def summary(self) -> str:
        """Render a readable plain-text report."""
        from modeldiff.reporting.console import render_console

        return render_console(self)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-compatible report mapping."""
        data = {
            "schema_version": self.schema_version,
            "task": self.task,
            "metric": self.metric,
            "sample_counts": {
                "total": self.sample_count,
                "discovery": self.discovery_count,
                "validation": self.validation_count,
            },
            "split": self.split,
            "overall": self.overall,
            "aggregate_metrics": self.aggregate_metrics,
            "prediction_diagnostics": self.disagreement,
            "regression_regions": [region.to_dict() for region in self.regression_regions],
            "improvement_regions": [region.to_dict() for region in self.improvement_regions],
            "warnings": self.warnings,
            "configuration": self.configuration.to_dict(),
        }
        return cast(dict[str, Any], _finite_or_none(data))

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialize the report and optionally write it to a path."""
        rendered = json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)
        if path is not None:
            Path(path).write_text(rendered + "\n", encoding="utf-8")
        return rendered

    def to_html(self, path: str | Path | None = None) -> str:
        """Render a standalone HTML report and optionally write it to a path."""
        from modeldiff.reporting.html import render_html

        rendered = render_html(self)
        if path is not None:
            Path(path).write_text(rendered, encoding="utf-8")
        return rendered


@dataclass
class StabilityReport(ModelDiffReport):
    """Comparison report whose regions passed repeated-discovery selection."""

    stability_configuration: StabilityConfig = field(default_factory=StabilityConfig)
    schema_version: str = "1.1"

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["analysis"] = "stability_selection"
        data["stability_configuration"] = self.stability_configuration.to_dict()
        data["triage_counts"] = {
            status: sum(
                isinstance(region, StableRegionResult) and region.triage == status
                for region in self.regression_regions
            )
            for status in ("new", "known", "partially_covered", "unclassified")
        }
        return data
