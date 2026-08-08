"""Unit tests for Phase B future forecasting feature generation and recursion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.future_forecasting import (
    FutureForecastError,
    _lag_rolling_from_history,
    generate_future_forecast,
    is_future_forecast_supported,
    load_chronological_history,
    recursive_tabular_future_forecast,
    validate_horizon,
)
from ml_engine.models.random_forest import RandomForestForecaster
from ml_engine.models.tabular_preprocessing import TabularFeatureTransformer


def _make_elec_features(features_dir: Path, n: int = 400) -> Path:
    features_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    demand = 500 + np.sin(np.linspace(0, 12, n)) * 40 + rng.normal(0, 3, n)
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": demand,
            "year": ts.year,
            "month": ts.month,
            "day": ts.day,
            "day_of_week": ts.dayofweek,
            "quarter": ts.quarter,
            "is_weekend": (ts.dayofweek >= 5).astype(int),
            "hour": ts.hour,
        }
    )
    for lag in (1, 24, 48, 168):
        frame[f"lag_{lag}"] = frame["demand_mw"].shift(lag)
    for window in (24, 168):
        hist = frame["demand_mw"].shift(1)
        frame[f"rolling_mean_{window}"] = hist.rolling(window, min_periods=window).mean()
        frame[f"rolling_std_{window}"] = hist.rolling(window, min_periods=window).std()
    frame = frame.dropna().reset_index(drop=True)
    n = len(frame)
    a, b = int(n * 0.6), int(n * 0.8)
    frame.iloc[:a].to_parquet(features_dir / "train.parquet", index=False)
    frame.iloc[a:b].to_parquet(features_dir / "validation.parquet", index=False)
    frame.iloc[b:].to_parquet(features_dir / "test.parquet", index=False)
    return features_dir


def _make_retail_features(features_dir: Path, n: int = 100) -> Path:
    features_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    sales = 50 + np.sin(np.linspace(0, 8, n)) * 12 + rng.normal(0, 2, n)
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "sales": sales,
            "year": ts.year.astype(float),
            "month": ts.month.astype(float),
            "day": ts.day.astype(float),
            "day_of_week": ts.dayofweek.astype(float),
            "quarter": ts.quarter.astype(float),
            "is_weekend": (ts.dayofweek >= 5).astype(float),
            "week_of_month": (((ts.day - 1) // 7) + 1).astype(float),
            "price": 100.0,
            "discount_pct": 5.0,
            "temperature_c": 28.0,
            "cpi_combined": 140.0,
            "city": "MetroA",
            "category": "Electronics",
        }
    )
    for lag in (1, 7, 14, 28):
        frame[f"lag_{lag}"] = frame["sales"].shift(lag)
    for window in (7, 14, 28):
        hist = frame["sales"].shift(1)
        frame[f"rolling_mean_{window}"] = hist.rolling(window, min_periods=window).mean()
        frame[f"rolling_std_{window}"] = hist.rolling(window, min_periods=window).std()
    frame = frame.dropna().reset_index(drop=True)
    n = len(frame)
    a, b = int(n * 0.6), int(n * 0.8)
    frame.iloc[:a].to_parquet(features_dir / "train.parquet", index=False)
    frame.iloc[a:b].to_parquet(features_dir / "validation.parquet", index=False)
    frame.iloc[b:].to_parquet(features_dir / "test.parquet", index=False)
    return features_dir


@pytest.fixture()
def elec_features(tmp_path: Path) -> Path:
    return _make_elec_features(tmp_path / "features_elec_future")


@pytest.fixture()
def retail_features(tmp_path: Path) -> Path:
    return _make_retail_features(tmp_path / "features_retail_future")


def test_horizon_validation() -> None:
    assert validate_horizon(7, domain="retail") == 7
    with pytest.raises(FutureForecastError):
        validate_horizon(0, domain="retail")
    with pytest.raises(FutureForecastError):
        validate_horizon(-3, domain="retail")
    with pytest.raises(FutureForecastError):
        validate_horizon(True, domain="retail")
    with pytest.raises(FutureForecastError):
        validate_horizon(7.5, domain="retail")
    with pytest.raises(FutureForecastError):
        validate_horizon(500, domain="retail")


def test_support_matrix() -> None:
    assert is_future_forecast_supported("retail", "random_forest")
    assert is_future_forecast_supported("electricity", "arima")
    assert not is_future_forecast_supported("retail", "lstm")
    assert not is_future_forecast_supported("retail", "hybrid")
    assert not is_future_forecast_supported("retail", "arima")


def test_lag_rolling_uses_only_past_targets() -> None:
    history = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    values = _lag_rolling_from_history(history, domain="retail")
    assert values["lag_1"] == 80.0
    assert values["lag_7"] == 20.0
    assert values["rolling_mean_7"] == pytest.approx(np.mean(history[-7:]))
    # Append a "future prediction" and ensure next lag_1 uses it (not a fabricated actual).
    history2 = history + [99.0]
    values2 = _lag_rolling_from_history(history2, domain="retail")
    assert values2["lag_1"] == 99.0


def test_future_timestamps_after_history_and_chronological(elec_features: Path) -> None:
    history = load_chronological_history(elec_features, "demand_mw")
    train = pd.read_parquet(elec_features / "train.parquet")
    model = RandomForestForecaster(n_estimators=20, random_state=42)
    model.fit(train, target_column="demand_mw")

    result = generate_future_forecast(
        model=model,
        model_key="random_forest",
        domain="electricity",
        features_dir=elec_features,
        horizon=24,
    )
    assert result["forecast_type"] == "future"
    assert result["horizon"] == 24
    assert len(result["predictions"]) == 24
    last_hist = pd.Timestamp(history["timestamp"].iloc[-1])
    timestamps = [pd.Timestamp(p["timestamp"]) for p in result["predictions"]]
    assert timestamps[0] > last_hist
    assert timestamps == sorted(timestamps)
    for item in result["predictions"]:
        assert set(item.keys()) == {"timestamp", "predicted"}
        assert "actual" not in item
        assert np.isfinite(item["predicted"])


def test_no_actual_fabricated_retail(retail_features: Path) -> None:
    train = pd.read_parquet(retail_features / "train.parquet")
    model = RandomForestForecaster(n_estimators=25, random_state=42)
    model.fit(train, target_column="sales")
    result = generate_future_forecast(
        model=model,
        model_key="random_forest",
        domain="retail",
        features_dir=retail_features,
        horizon=7,
    )
    assert result["exogenous_policy"] == "last_observation_carried_forward"
    assert all("actual" not in p for p in result["predictions"])
    assert result["prediction_count"] == 7


def test_preprocessing_not_refitted(elec_features: Path) -> None:
    train = pd.read_parquet(elec_features / "train.parquet")
    model = RandomForestForecaster(n_estimators=15, random_state=42)
    model.fit(train, target_column="demand_mw")
    transformer = model._transformer
    assert transformer is not None
    original_fit = TabularFeatureTransformer.fit

    def _fail_fit(self, *args, **kwargs):
        raise AssertionError("fit() must not be called during future forecasting")

    TabularFeatureTransformer.fit = _fail_fit
    try:
        preds = recursive_tabular_future_forecast(
            model=model,
            history_df=load_chronological_history(elec_features, "demand_mw"),
            domain="electricity",
            target_column="demand_mw",
            horizon=5,
        )
        assert len(preds) == 5
    finally:
        TabularFeatureTransformer.fit = original_fit


def test_unsupported_model_raises(elec_features: Path) -> None:
    train = pd.read_parquet(elec_features / "train.parquet")
    model = RandomForestForecaster(n_estimators=10, random_state=42)
    model.fit(train, target_column="demand_mw")
    with pytest.raises(FutureForecastError, match="not currently supported"):
        generate_future_forecast(
            model=model,
            model_key="lstm",
            domain="electricity",
            features_dir=elec_features,
            horizon=24,
        )
