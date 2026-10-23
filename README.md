# ModelDiff-ML

Your candidate model improved AUC from 0.84 to 0.85. But did it get better everywhere?

ModelDiff-ML compares two machine-learning models at the observation level and automatically
discovers interpretable, statistically validated regions where the candidate behaves differently
from the baseline. It is a small Python library built around NumPy, pandas, and SciPy—not an MLOps
platform.

```text
ordinary evaluation:  "The average metric improved."
ModelDiff-ML:          "It improved overall, but regressed for high-x1, low-x2 observations."
```

## Installation

ModelDiff-ML requires Python 3.11 or newer. The distribution is named `modeldiff-ml`; its
Python import package remains the concise `modeldiff`.

```bash
pip install modeldiff-ml
```

For local development:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Quickstart

ModelDiff-ML supports fitted sklearn-style models:

```python
import modeldiff

report = modeldiff.compare(
    baseline=old_model,
    candidate=new_model,
    X=X_test,
    y=y_test,
    task="binary_classification",
    metric="log_loss",
    random_state=42,
)

print(report.summary())
report.to_json("modeldiff-ml.json")
report.to_html("modeldiff-ml.html")
```

Binary classifiers must expose `predict_proba(X)` with two columns. ModelDiff-ML deliberately does not
turn an estimator's uncalibrated `decision_function` into a probability. Regression estimators need
only `predict(X)`.

## Raw predictions

The framework is model-agnostic when predictions are already available:

```python
from modeldiff import ModelDiffConfig, compare_predictions

config = ModelDiffConfig(
    max_depth=3,
    min_region_fraction=0.02,
    n_bootstrap=1000,
    n_permutations=2000,
    random_state=42,
)

report = compare_predictions(
    baseline_predictions=old_probabilities,
    candidate_predictions=new_probabilities,
    X=X_test,
    y=y_test,
    task="binary_classification",
    metric="log_loss",
    config=config,
)

significant_regressions = report.regressions(significant_only=True)
```

`X` may be a pandas DataFrame or two-dimensional NumPy array. DataFrames produce more readable
conditions. Arrays receive the names `x0`, `x1`, and so on.

## Leakage-aware validation splits

Random splitting remains the default. For repeated entities or ordered data, select a strategy that
keeps related observations out of both sides of the discovery/validation boundary:

```python
# All rows for a customer remain in one partition.
group_report = compare_predictions(
    old_predictions,
    new_predictions,
    X,
    y,
    task="regression",
    metric="mae",
    validation_strategy="group",
    groups=customer_ids,
)

# Earlier timestamps are used for discovery and later timestamps for validation.
time_report = compare_predictions(
    old_predictions,
    new_predictions,
    X,
    y,
    task="regression",
    metric="mae",
    validation_strategy="time",
    time_values=event_timestamps,
)

# Use an existing partition. Integer indices and boolean masks are both accepted.
explicit_report = compare_predictions(
    old_predictions,
    new_predictions,
    X,
    y,
    task="regression",
    metric="mae",
    validation_strategy="explicit",
    discovery_indices=development_mask,
    validation_indices=holdout_mask,
)
```

Group and time boundaries are chosen as close as possible to `discovery_fraction` without splitting
a group or equal timestamp. Explicit partitions must assign every observation exactly once. Invalid
or overlapping partitions raise `ValueError`; small or substantially unbalanced partitions add a
report warning. `report.split` and serialized reports contain the strategy, actual fraction, counts,
and strategy-specific audit metadata.

## Saved regression checks

Turn selected discoveries into fixed checks that can be applied to future model versions without
running region discovery again:

```python
checks = report.export_checks(
    regions=report.regressions(significant_only=True),
    max_loss_increase=0.01,
    min_samples=100,
    names=["High-utilization applicants"],  # one optional name per region
)
checks.to_json("model_checks.json")

result = modeldiff.evaluate_checks(
    checks="model_checks.json",
    baseline_predictions=production_predictions,
    candidate_predictions=new_predictions,
    X=X_fresh,
    y=y_fresh,
)

print(result.summary())
result.to_json("check_results.json")
result.to_html("check_results.html")
```

Calling `report.export_checks()` without `regions` exports significant regression regions. Passing
an explicit empty list creates a valid empty suite. Each saved check has a deterministic ID, readable
name, exact structured conditions, an absolute `max_loss_increase`, and `min_samples`. To tune one
policy after export without mutating the original suite:

```python
check_id = checks.checks[0].id
checks = checks.with_check_settings(
    check_id,
    name="Priority production segment",
    max_loss_increase=0.005,
    min_samples=250,
)
```

Evaluation statuses are `fail` when the observed mean loss increase is above the tolerance, `pass`
when it is equal to or below the tolerance, and `insufficient_data` when support is below the saved
minimum. Overall status is `fail` if any check fails, otherwise `insufficient_data` if any check lacks
support, `no_checks` for an empty suite, and `pass` only when every check in a nonempty suite passes.
Zero-support slices never pass, and unavailable loss values serialize as `null`.

Tolerance units are observation-level mean loss: accuracy uses misclassification-rate increase;
`mae` uses mean absolute-error increase; and `mse`, `squared_error`, and `rmse` checks use mean
squared-error increase. In particular, an RMSE check does **not** compare aggregate RMSE values.
The suite preserves the classification threshold and log-loss epsilon used by the source report.
Saved-suite and check-result JSON each use their own `1.0` schema, independently of comparison
report schemas.

These statuses are observed tolerance checks, not hypothesis-test conclusions. Evaluate against
fresh data when possible: using discovery data is descriptive, and repeatedly tuning against the
same evaluation set can overfit it. Exported checks cover only the selected findings, not every
possible future regression. See [`examples/saved_checks_example.py`](examples/saved_checks_example.py)
for a complete discover → save → fail/pass workflow.

## Interpreting delta

ModelDiff-ML uses one convention everywhere:

```text
delta_i = candidate_loss_i - baseline_loss_i
```

- Positive delta: the candidate is worse.
- Negative delta: the candidate is better.
- Zero: equal loss.

The supported classification losses are log loss, Brier loss (`"brier"`), and accuracy loss
(`"accuracy"`). Classification reports also include accuracy, log loss, Brier score, and rank-based
ROC AUC. Regression supports squared error (`"squared_error"` or `"mse"`), RMSE (`"rmse"`), and
absolute error (`"absolute_error"` or `"mae"`).

For RMSE, the overall metric is square-rooted after aggregation; region discovery still operates on
observation-level squared error. For accuracy, the primary loss is misclassification rate while the
aggregate metrics table also reports conventional accuracy.

## How region discovery works

1. ModelDiff-ML computes a loss difference for every observation.
2. It deterministically divides the data into discovery and validation partitions.
3. Numeric candidate conditions come from empirical quantiles. Categorical candidates use frequent
   equality and inequality conditions.
4. A from-scratch beam search builds conjunctions up to `max_depth`, scoring each as
   `abs(mean_delta) * sqrt(region_fraction)` (with sign-aware variants for each direction).
5. Highly overlapping regions are removed using Jaccard similarity.
6. The exact selected conditions are applied unchanged to the validation partition.
7. Validation effects receive percentile bootstrap confidence intervals and paired sign-flip
   p-values. Benjamini–Hochberg correction produces q-values across all tested regions.

This separation matters: searching for a threshold and reporting a p-value from those same rows
would understate selection bias. ModelDiff-ML's independent holdout is a practical first defense.

## Classification example

[`examples/classification_example.py`](examples/classification_example.py) creates a candidate that
is better overall but deliberately worse when `x1` is high and `x2` is low:

```bash
python examples/classification_example.py
```

This is the central ModelDiff-ML use case: aggregate improvement can hide a meaningful local
regression. A generated standalone report is available at
[`examples/classification_report.html`](examples/classification_report.html).

## Regression example

[`examples/regression_example.py`](examples/regression_example.py) compares two arrays of regression
predictions and discovers a planted high-error slice.

## CLI and CI

The CLI reads precomputed predictions from CSV. Unless excluded, columns other than the target and
prediction columns are features:

```bash
modeldiff-ml compare \
  --data validation.csv \
  --target y \
  --baseline-pred old_prediction \
  --candidate-pred new_prediction \
  --task binary_classification \
  --metric log_loss \
  --html report.html \
  --json report.json
```

The CLI supports the same split strategies with `--validation-strategy` and a control column:

```bash
modeldiff-ml compare ... --validation-strategy group --group-column customer_id
modeldiff-ml compare ... --validation-strategy time --time-column event_time
modeldiff-ml compare ... --validation-strategy explicit --split-column partition
```

An explicit split column must contain only `discovery` and `validation`. Split-control columns are
automatically excluded from the region-search features.

Use `--exclude COLUMN` repeatedly to omit non-feature columns. CI-oriented checks return exit code 1:

```bash
modeldiff-ml compare ... --fail-on-significant-regression
modeldiff-ml compare ... --max-overall-regression 0.01
```

Export significant regression regions during comparison, then evaluate them on a fresh CSV:

```bash
modeldiff-ml compare \
  --data discovery.csv \
  --target y --baseline-pred old --candidate-pred new \
  --task regression --metric mae \
  --export-checks model_checks.json \
  --check-max-loss-increase 0.01 \
  --check-min-samples 100

modeldiff-ml check \
  --checks model_checks.json \
  --data fresh_predictions.csv \
  --target y \
  --baseline-pred production_prediction \
  --candidate-pred new_prediction \
  --json check_results.json \
  --html check_results.html
```

The `check` command infers task, metric, threshold, and epsilon from the suite. Its exit codes are:

- `0`: a nonempty suite and every check passes.
- `1`: at least one failure, insufficient-data check, or an empty suite.
- `2`: invalid suite, invalid inputs, or command usage.

For valid evaluations, requested JSON and HTML reports are written before exit code `1` is returned.
The existing `compare` exit-code behavior is unchanged.

Comparison JSON output has schema version `1.0`, contains no model objects, and is suitable for CI
artifact processing.

## Configuration

`ModelDiffConfig` controls the discovery fraction, quantile grid, beam width, maximum depth, minimum
and maximum support, overlap threshold, resampling counts, confidence and significance levels,
classification threshold, numerical epsilon, and random seed. Defaults favor understandable output
and avoid tiny slices. Missing feature values are never imputed; they evaluate as false for V1
conditions.

## Statistical scope and limitations

Discovered subgroups are exploratory. Independent validation and FDR correction reduce false
discoveries, but do not eliminate them. Results can still be unstable with small data, distribution
shift, correlated repeated observations, or extensive repeated analyses by a human. A fresh final
test set is appropriate before high-stakes deployment decisions.

V1 supports binary classification and regression only. It does not include multiclass evaluation,
causal claims, fairness-specific analysis, distributed execution, SHAP, LLM evaluation, monitoring
infrastructure, model registries, dashboards, or automated remediation.

## Roadmap

- Multiclass observation-level losses
- Cross-validated discovery and stability summaries
- Pluggable, documented region scoring policies
- Richer calibrated-score adapters

## License

MIT
