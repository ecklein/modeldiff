"""ModelDiff-ML: behavioral regression testing for machine-learning models."""

from .api import compare, compare_predictions
from .checks import (
    CheckEvaluationResult,
    CheckOutcome,
    CheckSuite,
    SavedCheck,
    evaluate_checks,
)
from .config import ModelDiffConfig
from .result import ModelDiffReport, RegionResult
from .splitting import SplitResult, SplitStrategy, make_split

__all__ = [
    "ModelDiffConfig",
    "ModelDiffReport",
    "CheckEvaluationResult",
    "CheckOutcome",
    "CheckSuite",
    "RegionResult",
    "SavedCheck",
    "SplitResult",
    "SplitStrategy",
    "compare",
    "compare_predictions",
    "evaluate_checks",
    "make_split",
]
__version__ = "0.3.0"
