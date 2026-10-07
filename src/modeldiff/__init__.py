"""ModelDiff: behavioral regression testing for machine-learning models."""

from .api import compare, compare_predictions
from .config import ModelDiffConfig
from .result import ModelDiffReport, RegionResult

__all__ = [
    "ModelDiffConfig",
    "ModelDiffReport",
    "RegionResult",
    "compare",
    "compare_predictions",
]
__version__ = "0.1.0"
