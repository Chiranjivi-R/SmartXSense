"""Forecasting evaluation metrics for model performance assessment."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class EvaluationError(ValueError):
    """Raised when evaluation metrics cannot be computed due to invalid inputs."""


def _to_numpy_array(data: Any, name: str) -> np.ndarray:
    """Convert input series, array, or list to a 1D float numpy array."""
    if data is None:
        raise EvaluationError(f"Input '{name}' cannot be None")

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.to_numpy()

    try:
        arr = np.asarray(data, dtype=np.float64)
    except (ValueError, TypeError) as err:
        raise EvaluationError(f"Input '{name}' must be convertible to a numeric array") from err

    if arr.ndim > 1:
        arr = arr.squeeze()
        if arr.ndim > 1:
            raise EvaluationError(f"Input '{name}' must be 1-dimensional, got shape {arr.shape}")

    if arr.size == 0:
        raise EvaluationError(f"Input '{name}' cannot be empty")

    if np.isnan(arr).any() or np.isinf(arr).any():
        raise EvaluationError(f"Input '{name}' contains NaN or infinite values")

    return arr


def _validate_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert y_true and y_pred pair."""
    true_arr = _to_numpy_array(y_true, "y_true")
    pred_arr = _to_numpy_array(y_pred, "y_pred")

    if true_arr.shape != pred_arr.shape:
        raise EvaluationError(
            f"Input length mismatch: y_true has shape {true_arr.shape}, y_pred has shape {pred_arr.shape}"
        )

    return true_arr, pred_arr


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """Compute Mean Absolute Error (MAE)."""
    true_arr, pred_arr = _validate_pair(y_true, y_pred)
    return float(np.mean(np.abs(true_arr - pred_arr)))


def root_mean_squared_error(y_true: Any, y_pred: Any) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    true_arr, pred_arr = _validate_pair(y_true, y_pred)
    return float(np.sqrt(np.mean((true_arr - pred_arr) ** 2)))


def mean_absolute_percentage_error(y_true: Any, y_pred: Any, *, zero_threshold: float = 1e-8) -> float:
    """Compute Mean Absolute Percentage Error (MAPE) as a fraction (0.0 to inf).

    Zero-Target Policy:
    Observations where |y_true| < zero_threshold are excluded from the mean
    to prevent division by zero or infinite values. If all y_true observations are zero,
    0.0 is returned if y_pred matches y_true, or 0.0 as fallback.
    """
    true_arr, pred_arr = _validate_pair(y_true, y_pred)
    non_zero_mask = np.abs(true_arr) >= zero_threshold

    if not np.any(non_zero_mask):
        # All target values are zero.
        return 0.0

    percentage_errors = np.abs((true_arr[non_zero_mask] - pred_arr[non_zero_mask]) / true_arr[non_zero_mask])
    return float(np.mean(percentage_errors))


def r2_score(y_true: Any, y_pred: Any) -> float:
    """Compute R^2 (coefficient of determination).

    Returns 1.0 if total variance is zero and predictions match targets perfectly,
    or 0.0 if total variance is zero and predictions differ.
    """
    true_arr, pred_arr = _validate_pair(y_true, y_pred)
    ss_res = np.sum((true_arr - pred_arr) ** 2)
    mean_true = np.mean(true_arr)
    ss_tot = np.sum((true_arr - mean_true) ** 2)

    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else 0.0

    return float(1.0 - (ss_res / ss_tot))


def evaluate_forecast(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Compute all forecasting evaluation metrics and return standard metrics dictionary."""
    true_arr, pred_arr = _validate_pair(y_true, y_pred)
    return {
        "mae": mean_absolute_error(true_arr, pred_arr),
        "rmse": root_mean_squared_error(true_arr, pred_arr),
        "mape": mean_absolute_percentage_error(true_arr, pred_arr),
        "r2": r2_score(true_arr, pred_arr),
    }
