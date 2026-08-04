"""Tests for baseline forecasting models (Naive and Seasonal Naive)."""

from __future__ import annotations

import pandas as pd
import pytest

from ml_engine.models.baseline import BaselineModelError, NaiveForecaster


def test_naive_forecaster_fit_predict_series() -> None:
    train_target = pd.Series([10.0, 20.0, 30.0, 40.0], name="demand_mw")
    eval_target = pd.Series([50.0, 60.0, 70.0], name="demand_mw")

    forecaster = NaiveForecaster(seasonal_period=1)
    forecaster.fit(train_target)

    # In naive forecasting, pred(0) of eval is train_target[-1] = 40.0
    # pred(1) is eval_target[0] = 50.0, pred(2) is eval_target[1] = 60.0
    preds = forecaster.predict(eval_target)

    assert len(preds) == len(eval_target)
    assert list(preds.values) == [40.0, 50.0, 60.0]


def test_electricity_seasonal_naive_24_hour_history() -> None:
    timestamps = pd.date_range("2025-01-01", periods=48, freq="h")
    # Hourly values 1..48
    values = [float(i) for i in range(1, 49)]
    df = pd.DataFrame({"timestamp": timestamps, "demand_mw": values})

    train_df = df.iloc[:30].reset_index(drop=True)
    val_df = df.iloc[30:].reset_index(drop=True)

    forecaster = NaiveForecaster(seasonal_period=24)
    forecaster.fit(train_df, target_column="demand_mw")
    preds = forecaster.predict(val_df, target_column="demand_mw", history=train_df["demand_mw"])

    assert len(preds) == len(val_df)
    # Val starts at index 30 (timestamp hour 30, value 31.0).
    # Seasonal period is 24, so pred for index 30 should equal value at index (30 - 24 = 6) which is 7.0!
    assert preds.iloc[0] == 7.0
    # pred for index 31 should equal value at index (31 - 24 = 7) which is 8.0!
    assert preds.iloc[1] == 8.0


def test_no_future_target_leakage() -> None:
    train_series = pd.Series([10.0, 20.0, 30.0])
    # Future values contain large numbers (e.g. 1000.0)
    eval_series = pd.Series([1000.0, 2000.0, 3000.0])

    forecaster = NaiveForecaster(seasonal_period=1)
    forecaster.fit(train_series)
    preds = forecaster.predict(eval_series)

    # First prediction must use last train observation (30.0), NOT future 1000.0 or 2000.0
    assert preds.iloc[0] == 30.0
    assert preds.iloc[1] == 1000.0
    assert preds.iloc[2] == 2000.0


def test_prediction_timestamp_and_count_alignment() -> None:
    timestamps = pd.date_range("2025-01-01", periods=10, freq="D")
    train_df = pd.DataFrame({"timestamp": timestamps[:6], "sales": [10, 20, 30, 40, 50, 60]})
    eval_df = pd.DataFrame({"timestamp": timestamps[6:], "sales": [70, 80, 90, 100]})

    forecaster = NaiveForecaster(seasonal_period=1)
    forecaster.fit(train_df, target_column="sales")
    preds = forecaster.predict(eval_df, target_column="sales")

    assert len(preds) == len(eval_df)
    assert (preds.index == eval_df.index).all()


def test_invalid_seasonal_period() -> None:
    with pytest.raises(BaselineModelError, match="positive integer"):
        NaiveForecaster(seasonal_period=0)

    with pytest.raises(BaselineModelError, match="positive integer"):
        NaiveForecaster(seasonal_period=-5)


def test_predict_unfitted_or_empty_raises() -> None:
    forecaster = NaiveForecaster()
    with pytest.raises(BaselineModelError, match="empty"):
        forecaster.predict(pd.DataFrame())
