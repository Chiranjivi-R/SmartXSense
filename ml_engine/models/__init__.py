"""Forecasting models package."""

from .arima import ARIMAForecaster, ARIMAModelError
from .baseline import BaselineModelError, NaiveForecaster
from .random_forest import RandomForestForecaster, RandomForestModelError
from .tabular_preprocessing import PreprocessingError, TabularFeatureTransformer
from .xgboost_model import XGBoostForecaster, XGBoostModelError

__all__ = [
    "ARIMAForecaster",
    "ARIMAModelError",
    "BaselineModelError",
    "NaiveForecaster",
    "PreprocessingError",
    "RandomForestForecaster",
    "RandomForestModelError",
    "TabularFeatureTransformer",
    "TrainerError",
    "XGBoostForecaster",
    "XGBoostModelError",
    "run_training_pipeline",
]


def __getattr__(name: str):
    if name in ("run_training_pipeline", "TrainerError"):
        from .trainer import TrainerError, run_training_pipeline

        if name == "run_training_pipeline":
            return run_training_pipeline
        return TrainerError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
