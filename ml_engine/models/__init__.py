"""Forecasting models package."""

from .arima import ARIMAForecaster, ARIMAModelError
from .baseline import BaselineModelError, NaiveForecaster
from .cnn_lstm import CNNLSTMForecaster, CNNLSTMModelError, DEFAULT_LOOKBACK_CNN_LSTM
from .hybrid import DEFAULT_HYBRID_COMPONENTS, SUPPORTED_COMPONENTS, HybridForecaster, HybridModelError
from .lstm import DEFAULT_LOOKBACK, LSTMForecaster, LSTMModelError
from .random_forest import RETAIL_MACRO_EXCLUSIONS, RandomForestForecaster, RandomForestModelError
from .sequence_preprocessing import SequencePreprocessingError, SequencePreprocessor
from .tabular_preprocessing import PreprocessingError, TabularFeatureTransformer
from .xgboost_model import XGBoostForecaster, XGBoostModelError

__all__ = [
    "ARIMAForecaster",
    "ARIMAModelError",
    "BaselineModelError",
    "CNNLSTMForecaster",
    "CNNLSTMModelError",
    "DEFAULT_HYBRID_COMPONENTS",
    "DEFAULT_LOOKBACK",
    "DEFAULT_LOOKBACK_CNN_LSTM",
    "HybridForecaster",
    "HybridModelError",
    "LSTMForecaster",
    "LSTMModelError",
    "NaiveForecaster",
    "PreprocessingError",
    "RandomForestForecaster",
    "RandomForestModelError",
    "RETAIL_MACRO_EXCLUSIONS",
    "SequencePreprocessingError",
    "SequencePreprocessor",
    "SUPPORTED_COMPONENTS",
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
