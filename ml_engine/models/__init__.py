"""Forecasting models package."""

from .arima import ARIMAForecaster, ARIMAModelError
from .baseline import BaselineModelError, NaiveForecaster

__all__ = [
    "ARIMAForecaster",
    "ARIMAModelError",
    "BaselineModelError",
    "NaiveForecaster",
    "TrainerError",
    "run_training_pipeline",
]


def __getattr__(name: str):
    if name in ("run_training_pipeline", "TrainerError"):
        from .trainer import TrainerError, run_training_pipeline

        if name == "run_training_pipeline":
            return run_training_pipeline
        return TrainerError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
