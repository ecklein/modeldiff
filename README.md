# ModelDiff

Your candidate model improved AUC from 0.84 to 0.85. But did it get better everywhere?

ModelDiff compares two machine-learning models at the observation level and automatically
discovers interpretable, statistically validated regions where the candidate behaves differently
from the baseline. It is a small Python library built around NumPy, pandas, and SciPy—not an MLOps
platform.

```text
ordinary evaluation:  "The average metric improved."
ModelDiff:             "It improved overall, but regressed for high-x1, low-x2 observations."
```

## Installation

ModelDiff requires Python 3.11 or newer.

```bash
pip install modeldiff
```

For local development:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Quickstart

ModelDiff supports fitted sklearn-style models:

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
report.to_json("modeldiff.json")
report.to_html("modeldiff.html")
```

Binary classifiers must expose `predict_proba(X)` with two columns. ModelDiff deliberately does not
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

## Interpreting delta

ModelDiff uses one convention everywhere:

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

1. ModelDiff computes a loss difference for every observation.
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
would understate selection bias. ModelDiff's independent holdout is a practical first defense.

## Classification example

[`examples/classification_example.py`](examples/classification_example.py) creates a candidate that
is better overall but deliberately worse when `x1` is high and `x2` is low:

```bash
python examples/classification_example.py
```

This is the central ModelDiff use case: aggregate improvement can hide a meaningful local
regression. A generated standalone report is available at
[`examples/classification_report.html`](examples/classification_report.html).

## Regression example

[`examples/regression_example.py`](examples/regression_example.py) compares two arrays of regression
predictions and discovers a planted high-error slice.

## CLI and CI

The CLI reads precomputed predictions from CSV. Unless excluded, columns other than the target and
prediction columns are features:

```bash
modeldiff compare \
  --data validation.csv \
  --target y \
  --baseline-pred old_prediction \
  --candidate-pred new_prediction \
  --task binary_classification \
  --metric log_loss \
  --html report.html \
  --json report.json
```

Use `--exclude COLUMN` repeatedly to omit non-feature columns. CI-oriented checks return exit code 1:

```bash
modeldiff compare ... --fail-on-significant-regression
modeldiff compare ... --max-overall-regression 0.01
```

The JSON output has schema version `1.0`, contains no model objects, and is suitable for CI artifact
processing.

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
causal claims, fairness-specific analysis, time/group-aware splitting, distributed execution, SHAP,
LLM evaluation, monitoring infrastructure, model registries, dashboards, or automated remediation.

## Roadmap

- Group-aware and time-based validation strategies
- Multiclass observation-level losses
- Cross-validated discovery and stability summaries
- Pluggable, documented region scoring policies
- Richer calibrated-score adapters

## License

MIT

