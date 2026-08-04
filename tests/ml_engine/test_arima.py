"""Tests for ARIMA statistical forecasting model implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.arima import ARIMAForecaster, ARIMAModelError


def test_arima_fit_and_prediction_alignment() -> None:
    np.random.seed(42)
    timestamps = pd.date_range("2025-01-01", periods=100, freq="h")
    # Synthetic trending time series
    values = 100.0 + np.cumsum(np.random.randn(100))
    df = pd.DataFrame({"timestamp": timestamps, "demand_mw": values})

    train_df = df.iloc[:70].reset_index(drop=True)
    val_df = df.iloc[70:85].reset_index(drop=True)
    test_df = df.iloc[85:].reset_index(drop=True)

    forecaster = ARIMAForecaster(order=(1, 1, 1))
    forecaster.fit(train_df, target_column="demand_mw")

    val_preds = forecaster.predict(val_df, target_column="demand_mw", history=train_df["demand_mw"])
    assert len(val_preds) == len(val_df)
    assert not val_preds.isna().any()

    test_preds = forecaster.predict(test_df, target_column="demand_mw", history=df.iloc[:85]["demand_mw"])
    assert len(test_preds) == len(test_df)
    assert not test_preds.isna().any()


test_arima_order_config = [(1, 0, 0), (0, 1, 1), (1, 1, 1)]


@pytest.mark.parametrize("order", test_arima_order_config)
def test_configurable_arima_orders(order: tuple[int, int, int]) -> None:
    np.random.seed(42)
    train_series = pd.Series(10.0 + np.cumsum(np.random.randn(40)), name="demand_mw")
    eval_series = pd.Series(10.0 + np.cumsum(np.random.randn(10)), name="demand_mw")

    forecaster = ARIMAForecaster(order=order)
    forecaster.fit(train_series)
    preds = forecaster.predict(eval_series)

    assert len(preds) == len(eval_series)
    assert forecaster.order == order


def test_invalid_order_tuple() -> None:
    with pytest.raises(ARIMAModelError, match="3-element tuple"):
        ARIMAForecaster(order=(1, 1))  # type: ignore

    with pytest.raises(ARIMAModelError, match="non-negative integer"):
        ARIMAForecaster(order=(-1, 1, 1))


def test_predict_before_fit_raises() -> None:
    forecaster = ARIMAForecaster(order=(1, 1, 1))
    with pytest.raises(ARIMAModelError, match="fitted before calling predict"):
        forecaster.predict(pd.Series([1.0, 2.0]))


def test_empty_training_data_raises() -> None:
    forecaster = ARIMAForecaster()
    with pytest.raises(ARIMAModelError, match="empty"):
        forecaster.fit(pd.DataFrame())
