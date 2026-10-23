from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import modeldiff
from modeldiff import CheckSuite, ModelDiffConfig, SavedCheck
from modeldiff.analysis.regions import Condition
from modeldiff.result import ModelDiffReport, RegionResult


def _saved_check(
    *,
    check_id: str = "high-x",
    name: str = "High x",
    condition: Condition | None = None,
    tolerance: float = 0.1,
    min_samples: int = 3,
) -> SavedCheck:
    return SavedCheck(
        check_id=check_id,
        name=name,
        conditions=(condition or Condition("x", ">", 2.0),),
        max_loss_increase=tolerance,
        min_samples=min_samples,
        provenance={"source": "test"},
    )


def _suite(*checks: SavedCheck, metric: str = "mae", threshold: float = 0.5) -> CheckSuite:
    return CheckSuite(
        task="regression" if metric != "accuracy" else "binary_classification",
        metric=metric,
        checks=tuple(checks),
        threshold=threshold,
    )


def _report_with_region(condition: Condition) -> tuple[ModelDiffReport, RegionResult]:
    region = RegionResult(
        conditions=(condition,),
        direction="regression",
        discovery_support=20,
        discovery_fraction=0.4,
        validation_support=18,
        validation_fraction=0.36,
        discovery_delta=0.2,
        validation_delta=0.18,
        confidence_interval=(0.1, 0.3),
        p_value=0.001,
        q_value=0.002,
        score=0.1,
        baseline_mean_loss=0.2,
        candidate_mean_loss=0.38,
        significant=True,
    )
    report = ModelDiffReport(
        task="binary_classification",
        metric="accuracy",
        sample_count=100,
        discovery_count=50,
        validation_count=50,
        overall={},
        aggregate_metrics={},
        disagreement={},
        regression_regions=[region],
        configuration=ModelDiffConfig(threshold=0.7123456789012345, epsilon=1e-12),
    )
    return report, region


def test_export_and_load_round_trip_preserves_conditions_and_settings(tmp_path: Path) -> None:
    threshold = 0.12345678901234568
    report, region = _report_with_region(Condition("score", ">", threshold))
    suite = report.export_checks(
        regions=[region],
        max_loss_increase=0.0123456789012345,
        min_samples=17,
        names=["Critical score slice"],
    )
    path = tmp_path / "checks.json"
    suite.to_json(path)
    loaded = CheckSuite.load(path)

    assert loaded == suite
    assert loaded.schema_version == "1.0"
    assert loaded.threshold == 0.7123456789012345
    assert loaded.epsilon == 1e-12
    assert loaded.checks[0].conditions[0].value == threshold
    assert loaded.checks[0].name == "Critical score slice"
    assert "p_value" not in loaded.checks[0].provenance
    assert suite.checks[0].id == report.export_checks(regions=[region]).checks[0].id
    assert json.loads(suite.to_json()) == suite.to_dict()


def test_user_can_update_one_checks_name_and_policy() -> None:
    suite = _suite(_saved_check())
    updated = suite.with_check_settings(
        "high-x", name="Priority customers", max_loss_increase=0.25, min_samples=10
    )

    assert updated.checks[0].name == "Priority customers"
    assert updated.checks[0].max_loss_increase == 0.25
    assert updated.checks[0].min_samples == 10
    assert suite.checks[0].name == "High x"
    with pytest.raises(ValueError, match="unknown check ID"):
        suite.with_check_settings("missing", name="Nope")


def test_fixed_check_fails_then_passes_without_rediscovery() -> None:
    X = pd.DataFrame({"x": np.arange(6)})
    y = np.zeros(6)
    baseline = np.zeros(6)
    regressing = np.array([0, 0, 0, 0.3, 0.3, 0.3])
    improved = np.array([0, 0, 0, 0.05, 0.05, 0.05])
    suite = _suite(_saved_check())

    failed = modeldiff.evaluate_checks(suite, baseline, regressing, X, y)
    passed = modeldiff.evaluate_checks(suite, baseline, improved, X, y)

    assert failed.overall_status == "fail"
    assert failed.outcomes[0].loss_delta == pytest.approx(0.3)
    assert passed.overall_status == "pass"
    assert passed.outcomes[0].loss_delta == pytest.approx(0.05)
    assert failed.outcomes[0].description == "x > 2"


def test_delta_equal_to_tolerance_passes() -> None:
    suite = _suite(
        _saved_check(
            condition=Condition("x", ">", -1.0), tolerance=1.0, min_samples=4
        )
    )
    result = modeldiff.evaluate_checks(
        suite,
        np.zeros(4),
        np.ones(4),
        pd.DataFrame({"x": np.arange(4)}),
        np.zeros(4),
    )

    assert result.outcomes[0].loss_delta == 1.0
    assert result.outcomes[0].status == "pass"


@pytest.mark.parametrize(
    ("condition", "minimum", "support"),
    [
        (Condition("x", ">", 100.0), 1, 0),
        (Condition("x", ">", 3.0), 3, 2),
    ],
)
def test_zero_and_low_support_are_insufficient_data(
    condition: Condition, minimum: int, support: int
) -> None:
    suite = _suite(_saved_check(condition=condition, min_samples=minimum))
    result = modeldiff.evaluate_checks(
        suite,
        np.zeros(6),
        np.ones(6),
        pd.DataFrame({"x": np.arange(6)}),
        np.zeros(6),
    )
    outcome = result.outcomes[0]

    assert outcome.status == "insufficient_data"
    assert outcome.support_count == support
    assert outcome.baseline_mean_loss is None
    assert outcome.candidate_mean_loss is None
    assert outcome.loss_delta is None
    assert result.overall_status == "insufficient_data"
    assert result.any_evaluated is False


def test_mixed_status_counts_and_failure_precedence() -> None:
    suite = _suite(
        _saved_check(check_id="pass", condition=Condition("x", "<=", 1), min_samples=2),
        _saved_check(check_id="fail", condition=Condition("x", ">", 1), min_samples=2),
        _saved_check(
            check_id="small", condition=Condition("x", ">", 10), min_samples=1
        ),
    )
    candidate = np.array([0.0, 0.0, 0.4, 0.4])
    result = modeldiff.evaluate_checks(
        suite,
        np.zeros(4),
        candidate,
        pd.DataFrame({"x": np.arange(4)}),
        np.zeros(4),
    )

    assert [outcome.status for outcome in result.outcomes] == [
        "pass",
        "fail",
        "insufficient_data",
    ]
    assert result.passed_count == result.failed_count == result.insufficient_data_count == 1
    assert result.overall_status == "fail"
    assert result.any_evaluated is True


def test_empty_suite_is_not_reported_as_pass() -> None:
    result = modeldiff.evaluate_checks(
        _suite(),
        np.zeros(4),
        np.zeros(4),
        pd.DataFrame({"x": np.arange(4)}),
        np.zeros(4),
    )

    assert result.overall_status == "no_checks"
    assert result.any_evaluated is False
    assert "No checks evaluated" in result.summary()


def test_missing_and_incompatible_feature_schemas_raise_errors() -> None:
    suite = _suite(_saved_check())
    values = np.zeros(4)
    with pytest.raises(ValueError, match="missing feature column 'x'"):
        modeldiff.evaluate_checks(suite, values, values, pd.DataFrame({"z": range(4)}), values)
    with pytest.raises(ValueError, match="incompatible type"):
        modeldiff.evaluate_checks(
            suite,
            values,
            values,
            pd.DataFrame({"x": ["a", "b", "c", "d"]}),
            values,
        )


def test_malformed_suites_are_rejected() -> None:
    valid = _suite(_saved_check()).to_dict()
    malformed: list[tuple[dict[str, object], str]] = []

    unsupported = json.loads(json.dumps(valid))
    unsupported["schema_version"] = "9.0"
    malformed.append((unsupported, "unsupported"))

    invalid_task = json.loads(json.dumps(valid))
    invalid_task["task"] = "multiclass"
    malformed.append((invalid_task, "task must be one of"))

    invalid_metric = json.loads(json.dumps(valid))
    invalid_metric["metric"] = "r2"
    malformed.append((invalid_metric, "not supported"))

    duplicate = json.loads(json.dumps(valid))
    duplicate["checks"].append(dict(duplicate["checks"][0]))  # type: ignore[union-attr,index]
    malformed.append((duplicate, "unique"))

    invalid_tolerance = json.loads(json.dumps(valid))
    invalid_tolerance["checks"][0]["max_loss_increase"] = -1  # type: ignore[index]
    malformed.append((invalid_tolerance, "nonnegative"))

    boolean_minimum = json.loads(json.dumps(valid))
    boolean_minimum["checks"][0]["min_samples"] = True  # type: ignore[index]
    malformed.append((boolean_minimum, "positive integer"))

    invalid_operator = json.loads(json.dumps(valid))
    invalid_operator["checks"][0]["conditions"][0]["operator"] = "eval"  # type: ignore[index]
    malformed.append((invalid_operator, "unsupported condition operator"))

    invalid_value = json.loads(json.dumps(valid))
    invalid_value["checks"][0]["conditions"][0]["value"] = None  # type: ignore[index]
    malformed.append((invalid_value, "non-null JSON scalar"))

    missing_settings = json.loads(json.dumps(valid))
    del missing_settings["metric_settings"]["epsilon"]  # type: ignore[index]
    malformed.append((missing_settings, "requires threshold and epsilon"))

    invalid_threshold = json.loads(json.dumps(valid))
    invalid_threshold["metric_settings"]["threshold"] = 2  # type: ignore[index]
    malformed.append((invalid_threshold, "threshold"))

    for data, message in malformed:
        with pytest.raises(ValueError, match=message):
            CheckSuite.from_dict(data)

    with pytest.raises(ValueError, match="invalid check-suite JSON"):
        CheckSuite.from_json("not json")
    with pytest.raises(ValueError, match="NaN"):
        SavedCheck(
            check_id="x",
            name="x",
            conditions=(Condition("x", ">", 0),),
            max_loss_increase=0,
            min_samples=1,
            provenance={"bad": float("nan")},
        )


def test_accuracy_threshold_is_preserved() -> None:
    check = _saved_check(condition=Condition("x", ">", -1), tolerance=0, min_samples=4)
    suite = CheckSuite(
        task="binary_classification",
        metric="accuracy",
        checks=(check,),
        threshold=0.7,
    )
    result = modeldiff.evaluate_checks(
        suite,
        np.full(4, 0.6),
        np.full(4, 0.8),
        pd.DataFrame({"x": np.arange(4)}),
        np.zeros(4),
    )

    assert result.outcomes[0].loss_delta == 1.0
    assert result.outcomes[0].status == "fail"
    assert result.loss_unit == "misclassification_rate"


def test_rmse_checks_use_mean_squared_error_loss_units() -> None:
    suite = CheckSuite(
        task="regression",
        metric="rmse",
        checks=(
            _saved_check(
                condition=Condition("x", ">", -1), tolerance=3.0, min_samples=4
            ),
        ),
    )
    result = modeldiff.evaluate_checks(
        suite,
        np.zeros(4),
        np.full(4, 2.0),
        pd.DataFrame({"x": np.arange(4)}),
        np.zeros(4),
    )

    assert result.outcomes[0].loss_delta == 4.0
    assert result.outcomes[0].status == "fail"
    assert result.loss_unit == "mean_squared_error"


def test_result_json_uses_null_and_html_escapes_user_text(tmp_path: Path) -> None:
    suite = _suite(
        _saved_check(
            name="<script>alert('x')</script>",
            condition=Condition("<feature>", ">", 100),
            min_samples=1,
        )
    )
    result = modeldiff.evaluate_checks(
        suite,
        np.zeros(4),
        np.zeros(4),
        pd.DataFrame({"<feature>": np.arange(4)}),
        np.zeros(4),
    )
    json_path = tmp_path / "result.json"
    html_path = tmp_path / "result.html"
    result.to_json(json_path)
    result.to_html(html_path)
    loaded = json.loads(json_path.read_text())
    html = html_path.read_text()

    assert loaded["checks"][0]["loss_delta"] is None
    assert loaded["overall_status"] == "insufficient_data"
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;feature&gt;" in html


def test_exporting_explicit_empty_regions_creates_valid_empty_suite() -> None:
    report, _ = _report_with_region(Condition("x", ">", 1))
    suite = report.export_checks(regions=[])

    assert suite.checks == ()
    assert CheckSuite.from_json(suite.to_json()) == suite
