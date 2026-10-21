"""Serializable result objects for ModelDiff-ML."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from modeldiff.analysis.regions import Condition
from modeldiff.config import ModelDiffConfig


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
