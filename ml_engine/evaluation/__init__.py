"""Forecasting evaluation package."""

from .adaptive_selector import AdaptiveModelSelector, AdaptiveSelectorError
from .metrics import (
    EvaluationError,
    evaluate_forecast,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)
from .model_comparison import ModelComparisonError, compare_models

__all__ = [
    "AdaptiveModelSelector",
    "AdaptiveSelectorError",
    "EvaluationError",
    "ModelComparisonError",
    "compare_models",
    "evaluate_forecast",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "r2_score",
    "root_mean_squared_error",
]
