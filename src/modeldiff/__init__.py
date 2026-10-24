"""ModelDiff-ML: behavioral regression testing for machine-learning models."""

from .api import compare, compare_predictions
from .checks import (
    CheckDecisionPolicy,
    CheckEvaluationResult,
    CheckOutcome,
    CheckSuite,
    SavedCheck,
    evaluate_checks,
)
from .config import ModelDiffConfig, StabilityConfig
from .result import ModelDiffReport, RegionResult, StabilityReport, StableRegionResult
from .splitting import SplitResult, SplitStrategy, make_split
from .stability import compare_stable, compare_stable_predictions

__all__ = [
    "ModelDiffConfig",
    "StabilityConfig",
    "ModelDiffReport",
    "CheckEvaluationResult",
    "CheckDecisionPolicy",
    "CheckOutcome",
    "CheckSuite",
    "RegionResult",
    "StableRegionResult",
    "StabilityReport",
    "SavedCheck",
    "SplitResult",
    "SplitStrategy",
    "compare",
    "compare_predictions",
    "compare_stable",
    "compare_stable_predictions",
    "evaluate_checks",
    "make_split",
]
__version__ = "0.4.0"
