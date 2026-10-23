"""Versioned, reusable regression-slice checks."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from modeldiff.analysis.regions import Condition
from modeldiff.metrics.base import observation_losses
from modeldiff.utils import VALID_METRICS, VALID_TASKS, python_value, validate_inputs

if TYPE_CHECKING:
    from modeldiff.result import RegionResult

CHECK_SUITE_SCHEMA_VERSION = "1.0"
CHECK_RESULT_SCHEMA_VERSION = "1.0"
SUPPORTED_OPERATORS = {"<=", ">", "==", "!="}

CheckStatus = Literal["pass", "fail", "insufficient_data"]
OverallCheckStatus = Literal["pass", "fail", "insufficient_data", "no_checks"]


def loss_unit(task: str, metric: str) -> str:
    """Return the explicit unit used by a saved check's mean loss delta."""
    if task == "binary_classification":
        return {
            "accuracy": "misclassification_rate",
            "brier": "mean_brier_loss",
            "log_loss": "mean_log_loss",
        }[metric]
    return (
        "mean_squared_error"
        if metric in {"squared_error", "mse", "rmse"}
        else "mean_absolute_error"
    )


def _validate_tolerance(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("max_loss_increase must be a finite, nonnegative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("max_loss_increase must be a finite, nonnegative number")
    return result


def _validate_min_samples(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 1:
        raise ValueError("min_samples must be a positive integer")
    return int(value)


def _validate_json_data(value: Any, name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain NaN or Infinity")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_data(item, name)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{name} keys must be strings")
        for item in value.values():
            _validate_json_data(item, name)
        return
    raise ValueError(f"{name} must contain only JSON-compatible values")


def _validate_condition(condition: Condition) -> None:
    if not isinstance(condition.feature, str) or not condition.feature:
        raise ValueError("condition feature must be a nonempty string")
    if not isinstance(condition.operator, str) or condition.operator not in SUPPORTED_OPERATORS:
        raise ValueError(f"unsupported condition operator: {condition.operator!r}")
    value = python_value(condition.value)
    if value is None or not isinstance(value, (str, int, float, bool)):
        raise ValueError("condition value must be a non-null JSON scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("condition numeric values must be finite")
    if condition.operator in {"<=", ">"} and (
        isinstance(value, bool) or not isinstance(value, (int, float))
    ):
        raise ValueError(f"operator {condition.operator!r} requires a numeric condition value")


def _condition_from_dict(data: Any) -> Condition:
    if not isinstance(data, dict):
        raise ValueError("each condition must be an object")
    required = {"feature", "operator", "value"}
    if not required.issubset(data):
        raise ValueError("condition requires feature, operator, and value")
    condition = Condition(data["feature"], data["operator"], data["value"])
    _validate_condition(condition)
    return condition


def _stable_check_id(task: str, metric: str, conditions: tuple[Condition, ...]) -> str:
    payload = json.dumps(
        {
            "task": task,
            "metric": metric,
            "conditions": [condition.to_dict() for condition in conditions],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"slice-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


@dataclass(frozen=True)
class SavedCheck:
    """One fixed region and its observed-loss tolerance."""

    check_id: str
    name: str
    conditions: tuple[Condition, ...]
    max_loss_increase: float
    min_samples: int
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.check_id, str) or not self.check_id:
            raise ValueError("check_id must be a nonempty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("check name must be a nonempty string")
        object.__setattr__(self, "conditions", tuple(self.conditions))
        if not self.conditions:
            raise ValueError("a saved check must contain at least one condition")
        for condition in self.conditions:
            if not isinstance(condition, Condition):
                raise ValueError("saved check conditions must be Condition objects")
            _validate_condition(condition)
        object.__setattr__(self, "max_loss_increase", _validate_tolerance(self.max_loss_increase))
        object.__setattr__(self, "min_samples", _validate_min_samples(self.min_samples))
        if not isinstance(self.provenance, dict):
            raise ValueError("check provenance must be an object")
        _validate_json_data(self.provenance, "check provenance")

    @property
    def id(self) -> str:
        """Return the stable check identifier."""
        return self.check_id

    @property
    def description(self) -> str:
        """Return the fixed conjunction in human-readable form."""
        return " AND ".join(str(condition) for condition in self.conditions)

    def with_settings(
        self,
        *,
        name: str | None = None,
        max_loss_increase: float | None = None,
        min_samples: int | None = None,
    ) -> SavedCheck:
        """Return a copy with user-facing policy settings changed."""
        return replace(
            self,
            name=self.name if name is None else name,
            max_loss_increase=(
                self.max_loss_increase
                if max_loss_increase is None
                else _validate_tolerance(max_loss_increase)
            ),
            min_samples=(
                self.min_samples if min_samples is None else _validate_min_samples(min_samples)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible check mapping."""
        return {
            "id": self.check_id,
            "name": self.name,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "max_loss_increase": self.max_loss_increase,
            "min_samples": self.min_samples,
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, data: Any) -> SavedCheck:
        """Validate and load one check mapping."""
        if not isinstance(data, dict):
            raise ValueError("each saved check must be an object")
        required = {
            "id",
            "name",
            "conditions",
            "max_loss_increase",
            "min_samples",
            "provenance",
        }
        missing = required - set(data)
        if missing:
            raise ValueError(f"saved check is missing fields: {', '.join(sorted(missing))}")
        if not isinstance(data["conditions"], list):
            raise ValueError("saved check conditions must be an array")
        return cls(
            check_id=data["id"],
            name=data["name"],
            conditions=tuple(_condition_from_dict(item) for item in data["conditions"]),
            max_loss_increase=data["max_loss_increase"],
            min_samples=data["min_samples"],
            provenance=data["provenance"],
        )


@dataclass(frozen=True)
class CheckSuite:
    """A versioned collection of fixed regression-slice checks."""

    task: str
    metric: str
    checks: tuple[SavedCheck, ...] = ()
    threshold: float = 0.5
    epsilon: float = 1e-15
    schema_version: str = CHECK_SUITE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECK_SUITE_SCHEMA_VERSION:
            raise ValueError(f"unsupported check-suite schema version: {self.schema_version!r}")
        if not isinstance(self.task, str) or self.task not in VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(VALID_TASKS)}")
        if not isinstance(self.metric, str) or self.metric not in VALID_METRICS[self.task]:
            raise ValueError(f"metric {self.metric!r} is not supported for task {self.task!r}")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, numbers.Real):
            raise ValueError("metric threshold must be a finite number between 0 and 1")
        if not math.isfinite(float(self.threshold)) or not 0 <= float(self.threshold) <= 1:
            raise ValueError("metric threshold must be a finite number between 0 and 1")
        if isinstance(self.epsilon, bool) or not isinstance(self.epsilon, numbers.Real):
            raise ValueError("metric epsilon must be a finite number between 0 and 0.5")
        if not math.isfinite(float(self.epsilon)) or not 0 < float(self.epsilon) < 0.5:
            raise ValueError("metric epsilon must be a finite number between 0 and 0.5")
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "epsilon", float(self.epsilon))
        object.__setattr__(self, "checks", tuple(self.checks))
        if any(not isinstance(check, SavedCheck) for check in self.checks):
            raise ValueError("checks must contain SavedCheck objects")
        identifiers = [check.check_id for check in self.checks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("saved check IDs must be unique")

    def with_check_settings(
        self,
        check_id: str,
        *,
        name: str | None = None,
        max_loss_increase: float | None = None,
        min_samples: int | None = None,
    ) -> CheckSuite:
        """Return a suite with one named check's policy settings updated."""
        if check_id not in {check.check_id for check in self.checks}:
            raise ValueError(f"unknown check ID: {check_id!r}")
        updated = tuple(
            check.with_settings(
                name=name,
                max_loss_increase=max_loss_increase,
                min_samples=min_samples,
            )
            if check.check_id == check_id
            else check
            for check in self.checks
        )
        return replace(self, checks=updated)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-compatible suite mapping."""
        return {
            "schema_version": self.schema_version,
            "task": self.task,
            "metric": self.metric,
            "metric_settings": {"threshold": self.threshold, "epsilon": self.epsilon},
            "checks": [check.to_dict() for check in self.checks],
        }

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialize the suite and optionally write it to a file."""
        rendered = json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)
        if path is not None:
            Path(path).write_text(rendered + "\n", encoding="utf-8")
        return rendered

    @classmethod
    def from_dict(cls, data: Any) -> CheckSuite:
        """Validate and load a suite mapping."""
        if not isinstance(data, dict):
            raise ValueError("check suite must be a JSON object")
        required = {"schema_version", "task", "metric", "metric_settings", "checks"}
        missing = required - set(data)
        if missing:
            raise ValueError(f"check suite is missing fields: {', '.join(sorted(missing))}")
        settings = data["metric_settings"]
        if not isinstance(settings, dict) or not {"threshold", "epsilon"}.issubset(settings):
            raise ValueError("metric_settings requires threshold and epsilon")
        if not isinstance(data["checks"], list):
            raise ValueError("checks must be an array")
        return cls(
            task=data["task"],
            metric=data["metric"],
            checks=tuple(SavedCheck.from_dict(item) for item in data["checks"]),
            threshold=settings["threshold"],
            epsilon=settings["epsilon"],
            schema_version=data["schema_version"],
        )

    @classmethod
    def from_json(cls, text: str) -> CheckSuite:
        """Parse and validate a suite from a JSON string."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid check-suite JSON: {error.msg}") from error
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: str | Path) -> CheckSuite:
        """Load and validate a suite from a UTF-8 JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class CheckOutcome:
    """Observed result for one saved check on fresh data."""

    check_id: str
    name: str
    conditions: tuple[Condition, ...]
    support_count: int
    support_fraction: float
    baseline_mean_loss: float | None
    candidate_mean_loss: float | None
    loss_delta: float | None
    max_loss_increase: float
    min_samples: int
    status: CheckStatus
    explanation: str

    @property
    def id(self) -> str:
        return self.check_id

    @property
    def description(self) -> str:
        return " AND ".join(str(condition) for condition in self.conditions)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible outcome mapping."""
        return {
            "id": self.check_id,
            "name": self.name,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "description": self.description,
            "support_count": self.support_count,
            "support_fraction": self.support_fraction,
            "baseline_mean_loss": self.baseline_mean_loss,
            "candidate_mean_loss": self.candidate_mean_loss,
            "loss_delta": self.loss_delta,
            "max_loss_increase": self.max_loss_increase,
            "min_samples": self.min_samples,
            "status": self.status,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class CheckEvaluationResult:
    """Complete result of evaluating a saved check suite."""

    task: str
    metric: str
    sample_count: int
    outcomes: tuple[CheckOutcome, ...]
    suite_schema_version: str
    threshold: float
    epsilon: float
    schema_version: str = CHECK_RESULT_SCHEMA_VERSION

    @property
    def passed_count(self) -> int:
        return sum(outcome.status == "pass" for outcome in self.outcomes)

    @property
    def failed_count(self) -> int:
        return sum(outcome.status == "fail" for outcome in self.outcomes)

    @property
    def insufficient_data_count(self) -> int:
        return sum(outcome.status == "insufficient_data" for outcome in self.outcomes)

    @property
    def any_evaluated(self) -> bool:
        return any(outcome.status != "insufficient_data" for outcome in self.outcomes)

    @property
    def overall_status(self) -> OverallCheckStatus:
        if not self.outcomes:
            return "no_checks"
        if self.failed_count:
            return "fail"
        if self.insufficient_data_count:
            return "insufficient_data"
        return "pass"

    @property
    def loss_unit(self) -> str:
        return loss_unit(self.task, self.metric)

    def summary(self) -> str:
        """Render a concise text summary."""
        from modeldiff.reporting.checks import render_check_console

        return render_check_console(self)

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned result mapping."""
        data = {
            "schema_version": self.schema_version,
            "suite_schema_version": self.suite_schema_version,
            "task": self.task,
            "metric": self.metric,
            "loss_unit": self.loss_unit,
            "metric_settings": {"threshold": self.threshold, "epsilon": self.epsilon},
            "sample_count": self.sample_count,
            "overall_status": self.overall_status,
            "any_checks_evaluated": self.any_evaluated,
            "counts": {
                "total": len(self.outcomes),
                "passed": self.passed_count,
                "failed": self.failed_count,
                "insufficient_data": self.insufficient_data_count,
            },
            "checks": [outcome.to_dict() for outcome in self.outcomes],
        }
        return cast(dict[str, Any], _finite_or_none(data))

    def to_json(self, path: str | Path | None = None, *, indent: int = 2) -> str:
        """Serialize the result and optionally write it to a file."""
        rendered = json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)
        if path is not None:
            Path(path).write_text(rendered + "\n", encoding="utf-8")
        return rendered

    def to_html(self, path: str | Path | None = None) -> str:
        """Render a standalone HTML result and optionally write it to a file."""
        from modeldiff.reporting.checks import render_check_html

        rendered = render_check_html(self)
        if path is not None:
            Path(path).write_text(rendered, encoding="utf-8")
        return rendered


def _finite_or_none(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_or_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_or_none(item) for item in value]
    return value


def _feature_kind(series: pd.Series[Any]) -> str | None:
    nonmissing = series.dropna()
    if nonmissing.empty:
        return None
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    values = nonmissing.to_list()
    if all(isinstance(value, (bool, np.bool_)) for value in values):
        return "boolean"
    if all(
        isinstance(value, numbers.Real) and not isinstance(value, (bool, np.bool_))
        for value in values
    ):
        return "numeric"
    if all(isinstance(value, str) for value in values):
        return "string"
    return "incompatible"


def _value_kind(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, numbers.Real):
        return "numeric"
    return "string"


def _validate_feature_schema(frame: pd.DataFrame, check: SavedCheck) -> None:
    for condition in check.conditions:
        if condition.feature not in frame.columns:
            raise ValueError(
                f"check {check.check_id!r} requires missing feature column {condition.feature!r}"
            )
        actual_kind = _feature_kind(frame[condition.feature])
        expected_kind = _value_kind(condition.value)
        if actual_kind is not None and actual_kind != expected_kind:
            raise ValueError(
                f"feature {condition.feature!r} has incompatible type for check "
                f"{check.check_id!r}: expected {expected_kind}, found {actual_kind}"
            )


def evaluate_checks(
    checks: CheckSuite | str | Path,
    baseline_predictions: Any,
    candidate_predictions: Any,
    X: Any,
    y: Any,
) -> CheckEvaluationResult:
    """Evaluate fixed saved conditions without discovery, optimization, or splitting."""
    suite = checks if isinstance(checks, CheckSuite) else CheckSuite.load(checks)
    frame, target, baseline, candidate = validate_inputs(
        X,
        y,
        baseline_predictions,
        candidate_predictions,
        suite.task,
        suite.metric,
    )
    baseline_loss = observation_losses(
        suite.task,
        suite.metric,
        target,
        baseline,
        threshold=suite.threshold,
        epsilon=suite.epsilon,
    )
    candidate_loss = observation_losses(
        suite.task,
        suite.metric,
        target,
        candidate,
        threshold=suite.threshold,
        epsilon=suite.epsilon,
    )
    outcomes: list[CheckOutcome] = []
    for check in suite.checks:
        _validate_feature_schema(frame, check)
        mask = np.ones(len(frame), dtype=bool)
        for condition in check.conditions:
            mask &= condition.apply(frame)
        support = int(mask.sum())
        fraction = support / len(frame)
        if support < check.min_samples:
            outcomes.append(
                CheckOutcome(
                    check_id=check.check_id,
                    name=check.name,
                    conditions=check.conditions,
                    support_count=support,
                    support_fraction=fraction,
                    baseline_mean_loss=None,
                    candidate_mean_loss=None,
                    loss_delta=None,
                    max_loss_increase=check.max_loss_increase,
                    min_samples=check.min_samples,
                    status="insufficient_data",
                    explanation=(
                        f"Matched {support} observations; at least "
                        f"{check.min_samples} are required."
                    ),
                )
            )
            continue
        baseline_mean = float(np.mean(baseline_loss[mask]))
        candidate_mean = float(np.mean(candidate_loss[mask]))
        delta = candidate_mean - baseline_mean
        failed = delta > check.max_loss_increase
        outcomes.append(
            CheckOutcome(
                check_id=check.check_id,
                name=check.name,
                conditions=check.conditions,
                support_count=support,
                support_fraction=fraction,
                baseline_mean_loss=baseline_mean,
                candidate_mean_loss=candidate_mean,
                loss_delta=delta,
                max_loss_increase=check.max_loss_increase,
                min_samples=check.min_samples,
                status="fail" if failed else "pass",
                explanation=(
                    f"Loss increase {delta:.6g} exceeds tolerance "
                    f"{check.max_loss_increase:.6g}."
                    if failed
                    else f"Loss increase {delta:.6g} is within tolerance "
                    f"{check.max_loss_increase:.6g}."
                ),
            )
        )
    return CheckEvaluationResult(
        task=suite.task,
        metric=suite.metric,
        sample_count=len(frame),
        outcomes=tuple(outcomes),
        suite_schema_version=suite.schema_version,
        threshold=suite.threshold,
        epsilon=suite.epsilon,
    )


def checks_from_regions(
    *,
    task: str,
    metric: str,
    threshold: float,
    epsilon: float,
    report_schema_version: str,
    regions: Sequence[RegionResult],
    max_loss_increase: float,
    min_samples: int,
    names: Sequence[str] | None = None,
) -> CheckSuite:
    """Build deterministic saved checks from selected RegionResult objects."""
    tolerance = _validate_tolerance(max_loss_increase)
    minimum = _validate_min_samples(min_samples)
    selected = list(regions)
    if names is not None and len(names) != len(selected):
        raise ValueError("names must contain one value for each exported region")
    used_ids: dict[str, int] = {}
    saved: list[SavedCheck] = []
    for index, region in enumerate(selected, 1):
        if region.direction != "regression":
            raise ValueError("only regression regions can be exported as regression checks")
        conditions = tuple(region.conditions)
        base_id = _stable_check_id(task, metric, conditions)
        occurrence = used_ids.get(base_id, 0) + 1
        used_ids[base_id] = occurrence
        check_id = base_id if occurrence == 1 else f"{base_id}-{occurrence}"
        name = names[index - 1] if names is not None else f"Regression slice {index}"
        saved.append(
            SavedCheck(
                check_id=check_id,
                name=name,
                conditions=conditions,
                max_loss_increase=tolerance,
                min_samples=minimum,
                provenance={
                    "source": "modeldiff_report",
                    "report_schema_version": report_schema_version,
                    "direction": region.direction,
                    "discovery_support": region.discovery_support,
                    "validation_support": region.validation_support,
                },
            )
        )
    return CheckSuite(
        task=task,
        metric=metric,
        checks=tuple(saved),
        threshold=threshold,
        epsilon=epsilon,
    )
