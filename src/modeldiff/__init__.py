"""ModelDiff-ML: behavioral regression testing for machine-learning models."""

from .api import compare, compare_predictions
from .config import ModelDiffConfig
from .result import ModelDiffReport, RegionResult
from .splitting import SplitResult, SplitStrategy, make_split

__all__ = [
    "ModelDiffConfig",
    "ModelDiffReport",
    "RegionResult",
    "SplitResult",
    "SplitStrategy",
    "compare",
    "compare_predictions",
    "make_split",
]
__version__ = "0.2.0"
