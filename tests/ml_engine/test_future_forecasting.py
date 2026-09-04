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
    assert validate_horizon(2160, domain="electricity") == 2160
    with pytest.raises(FutureForecastError):
        validate_horizon(2161, domain="electricity")


def test_support_matrix() -> None:
    assert is_future_forecast_supported("retail", "random_forest")
    assert is_future_forecast_supported("electricity", "arima")
    assert is_future_forecast_supported("electricity", "hybrid")
    assert not is_future_forecast_supported("retail", "lstm")
    assert not is_future_forecast_supported("retail", "hybrid")
    assert not is_future_forecast_supported("retail", "arima")


def test_future_calendar_region_is_recomputed_not_carried_forward() -> None:
    from ml_engine.models.future_forecasting import build_future_feature_frame

    history = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-11-06T00:00:00Z", "2026-11-07T00:00:00Z"]),
        "demand_mw": [100.0, 101.0],
        "calendar_region": ["india", "india"],
    })
    future, _ = build_future_feature_frame(history, domain="electricity", target_column="demand_mw", horizon=1)
    assert future.loc[0, "calendar_region"] == "india"
    assert future.loc[0, "is_festival"] == 0
    assert future.loc[0, "holiday_name"] is pd.NA or pd.isna(future.loc[0, "holiday_name"])


def test_future_calendar_uses_declared_india_uk_or_other_context() -> None:
    from ml_engine.models.future_forecasting import build_future_feature_frame

    def future_for(region: str, last_timestamp: str):
        history = pd.DataFrame({"timestamp": pd.to_datetime([last_timestamp]), "demand_mw": [100.0], "calendar_region": [region]})
        return build_future_feature_frame(history, domain="electricity", target_column="demand_mw", horizon=1)[0].iloc[0]

    india = future_for("india", "2026-03-03T23:00:00Z")
    uk = future_for("uk", "2026-12-24T23:00:00Z")
    other = future_for("other", "2026-03-03T23:00:00Z")
    assert india["holiday_name"] == "Holi"
    assert india["is_festival"] == 1
    assert uk["holiday_name"] == "Christmas Day"
    assert "is_festival" not in uk.index
    assert "is_holiday" not in other.index


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


def test_electricity_hybrid_future_forecasting(elec_features: Path) -> None:
    """Verify Electricity Hybrid combines valid future forecasts using validation weights."""
    from ml_engine.models.arima import ARIMAForecaster
    from ml_engine.models.hybrid import HybridForecaster
    from ml_engine.models.xgboost_model import XGBoostForecaster
    from ml_engine.evaluation.metrics import evaluate_forecast

    train = pd.read_parquet(elec_features / "train.parquet")
    val = pd.read_parquet(elec_features / "validation.parquet")

    # Fit component models on train
    m_arima = ARIMAForecaster(order=(1, 1, 1)).fit(train, target_column="demand_mw")
    m_rf = RandomForestForecaster(n_estimators=15, random_state=42).fit(train, target_column="demand_mw")
    m_xgb = XGBoostForecaster(n_estimators=15, max_depth=3, learning_rate=0.05, random_state=42).fit(train, target_column="demand_mw")

    # Evaluate validation predictions to derive frozen weights
    v_arima = m_arima.predict(val, target_column="demand_mw", history=train["demand_mw"])
    v_rf = m_rf.predict(val)
    v_xgb = m_xgb.predict(val)

    comp_val = {
        "arima": evaluate_forecast(val["demand_mw"], v_arima),
        "random_forest": evaluate_forecast(val["demand_mw"], v_rf),
        "xgboost": evaluate_forecast(val["demand_mw"], v_xgb),
    }
    weights = HybridForecaster.calculate_weights(comp_val, weighting_strategy="inverse_rmse")

    hyb = HybridForecaster(components=["arima", "random_forest", "xgboost"], domain="electricity")
    hyb.weights = weights
    hyb.component_models = {"arima": m_arima, "random_forest": m_rf, "xgboost": m_xgb}

    history = load_chronological_history(elec_features, "demand_mw")
    last_hist = pd.Timestamp(history["timestamp"].iloc[-1])

    result = generate_future_forecast(
        model=hyb,
        model_key="hybrid",
        domain="electricity",
        features_dir=elec_features,
        horizon=24,
    )

    assert result["forecast_type"] == "future"
    assert result["domain"] == "electricity"
    assert result["model"] == "hybrid"
    assert result["horizon"] == 24
    assert len(result["predictions"]) == 24
    assert result["prediction_count"] == 24

    timestamps = [pd.Timestamp(p["timestamp"]) for p in result["predictions"]]
    assert timestamps[0] > last_hist
    assert timestamps == sorted(timestamps)

    for p in result["predictions"]:
        assert "timestamp" in p
        assert "predicted" in p
        assert "actual" not in p
        assert np.isfinite(p["predicted"])


def test_electricity_hybrid_future_forecast_rejects_unsupported_components(elec_features: Path) -> None:
    from ml_engine.models.hybrid import HybridForecaster

    train = pd.read_parquet(elec_features / "train.parquet")
    m_rf = RandomForestForecaster(n_estimators=10, random_state=42).fit(train, target_column="demand_mw")

    hyb = HybridForecaster(components=["random_forest", "lstm"], domain="electricity")
    hyb.weights = {"random_forest": 0.5, "lstm": 0.5}
    hyb.component_models = {"random_forest": m_rf, "lstm": "mock_lstm"}

    with pytest.raises(FutureForecastError, match="does not support future forecasting"):
        generate_future_forecast(
            model=hyb,
            model_key="hybrid",
            domain="electricity",
            features_dir=elec_features,
            horizon=12,
        )


def test_hybrid_future_forecast_rejects_retail(retail_features: Path) -> None:
    from ml_engine.models.hybrid import HybridForecaster

    train = pd.read_parquet(retail_features / "train.parquet")
    m_rf = RandomForestForecaster(n_estimators=10, random_state=42).fit(train, target_column="sales")

    hyb = HybridForecaster(components=["random_forest", "xgboost"], domain="retail")
    hyb.weights = {"random_forest": 0.5, "xgboost": 0.5}
    hyb.component_models = {"random_forest": m_rf, "xgboost": m_rf}

    with pytest.raises(FutureForecastError, match="not currently supported"):
        generate_future_forecast(
            model=hyb,
            model_key="hybrid",
            domain="retail",
            features_dir=retail_features,
            horizon=7,
        )


def test_domain_specific_horizon_bounds() -> None:
    """Verify domain-specific horizon limits (90 for retail, 2160 for electricity)."""
    assert validate_horizon(90, domain="retail") == 90
    with pytest.raises(FutureForecastError, match="exceeds maximum allowed value of 90 for domain 'retail'"):
        validate_horizon(91, domain="retail")
    assert validate_horizon(2160, domain="electricity") == 2160
    with pytest.raises(FutureForecastError, match="exceeds maximum allowed value of 2160 for domain 'electricity'"):
        validate_horizon(2161, domain="electricity")


def test_ewma3_recursive_evolution() -> None:
    """Verify ewma_3 recursively incorporates y_T at step 1 and predicted targets thereafter."""
    from ml_engine.models.future_forecasting import build_future_feature_frame

    # History of 10 rows ending at sales=100.0 and ewma_3=80.0
    history = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC"),
        "sales": [50.0] * 9 + [100.0],
        "ewma_3": [50.0] * 9 + [80.0],
        "calendar_region": ["india"] * 10,
    })

    # Suppose predicted targets are [120.0, 140.0, 160.0]
    preds = [120.0, 140.0, 160.0]
    future, _ = build_future_feature_frame(
        history,
        domain="retail",
        target_column="sales",
        horizon=3,
        predicted_targets=preds,
    )

    # Step 1 (t+1): ewma incorporates y_T (100.0) -> 0.5 * 100.0 + 0.5 * 80.0 = 90.0
    # Step 2 (t+2): ewma incorporates pred_1 (120.0) -> 0.5 * 120.0 + 0.5 * 90.0 = 105.0
    # Step 3 (t+3): ewma incorporates pred_2 (140.0) -> 0.5 * 140.0 + 0.5 * 105.0 = 122.5
    assert future.loc[0, "ewma_3"] == pytest.approx(90.0)
    assert future.loc[1, "ewma_3"] == pytest.approx(105.0)
    assert future.loc[2, "ewma_3"] == pytest.approx(122.5)


def test_payday_week_dynamic_recomputation() -> None:
    """Verify payday_week is deterministically recomputed from date, not frozen via LOCF."""
    from ml_engine.models.future_forecasting import build_future_feature_frame

    # April 30 is in payday_week (last 2 days of month -> payday_week = 1.0)
    history = pd.DataFrame({
        "timestamp": pd.date_range("2024-04-20", "2024-04-30", freq="D", tz="UTC"),
        "sales": [50.0] * 11,
        "payday_week": [0.0] * 8 + [1.0, 1.0, 1.0],
        "calendar_region": ["india"] * 11,
    })

    # Horizon 15 days: May 1 to May 15
    future, _ = build_future_feature_frame(
        history,
        domain="retail",
        target_column="sales",
        horizon=15,
        predicted_targets=[50.0] * 15,
    )

    # May 1..7: payday_week == 1.0 (day <= 7)
    for i in range(7):
        assert future.loc[i, "payday_week"] == 1.0, f"May {i+1} should be payday_week"

    # May 8..15: payday_week == 0.0 (day > 7 and not end of month)
    for i in range(7, 15):
        assert future.loc[i, "payday_week"] == 0.0, f"May {i+1} should NOT be payday_week"


def test_festival_proximity_flags_and_interactions_evolution() -> None:
    """Verify festival_distance, flags, and interaction terms evolve dynamically without leakage or sentinel."""
    from ml_engine.models.future_forecasting import build_future_feature_frame

    # Holi in 2025 is on 2025-03-14 (defined in INDIA_FESTIVALS). History ends on 2025-03-09.
    history = pd.DataFrame({
        "timestamp": pd.date_range("2025-02-20", "2025-03-09", freq="D", tz="UTC"),
        "sales": [50.0] * 18,
        "festival_distance": [4.0] * 18,
        "festival_flag": [0.0] * 18,
        "holiday_flag": [0.0] * 18,
        "festival_season_flag": [0.0] * 18,
        "festival_name": [None] * 18,
        "calendar_region": ["india"] * 18,
        "lag_1": [50.0] * 18,
        "fest_x_lag_1": [0.0] * 18,
        "fest_season_x_lag1": [0.0] * 18,
    })

    preds = [60.0 + i * 5 for i in range(10)]  # 2025-03-10 to 2025-03-19
    future, _ = build_future_feature_frame(
        history,
        domain="retail",
        target_column="sales",
        horizon=10,
        predicted_targets=preds,
    )

    # March 10: 4 days before Holi -> -4.0 (saturated)
    assert future.loc[0, "festival_distance"] == -4.0
    assert future.loc[0, "festival_flag"] == 0.0
    assert future.loc[0, "festival_season_flag"] == 0.0
    assert future.loc[0, "fest_x_lag_1"] == 0.0
    assert future.loc[0, "fest_season_x_lag1"] == 0.0

    # March 11: 3 days before Holi -> -3.0, season_flag=1.0
    assert future.loc[1, "festival_distance"] == -3.0
    assert future.loc[1, "festival_flag"] == 0.0
    assert future.loc[1, "festival_season_flag"] == 1.0
    assert future.loc[1, "fest_x_lag_1"] == 0.0
    assert future.loc[1, "fest_season_x_lag1"] == pytest.approx(future.loc[1, "lag_1"])

    # March 12: 2 days before -> -2.0
    assert future.loc[2, "festival_distance"] == -2.0
    assert future.loc[2, "festival_season_flag"] == 1.0

    # March 13: 1 day before -> -1.0
    assert future.loc[3, "festival_distance"] == -1.0
    assert future.loc[3, "festival_season_flag"] == 1.0

    # March 14: Holi day -> distance = 0.0, festival_flag = 1.0, holiday_flag = 1.0
    assert future.loc[4, "festival_distance"] == 0.0
    assert future.loc[4, "festival_flag"] == 1.0
    assert future.loc[4, "holiday_flag"] == 1.0
    assert future.loc[4, "festival_name"] == "Holi"
    assert future.loc[4, "festival_season_flag"] == 1.0
    # fest_x_lag_1 should be 1.0 * lag_1
    assert future.loc[4, "fest_x_lag_1"] == pytest.approx(future.loc[4, "lag_1"])
    assert future.loc[4, "fest_season_x_lag1"] == pytest.approx(future.loc[4, "lag_1"])

    # March 15: 1 day after -> 1.0
    assert future.loc[5, "festival_distance"] == 1.0
    assert future.loc[5, "festival_flag"] == 0.0
    assert future.loc[5, "festival_season_flag"] == 1.0

    # March 16: 2 days after -> 2.0
    assert future.loc[6, "festival_distance"] == 2.0
    assert future.loc[6, "festival_season_flag"] == 1.0

    # March 17: 3 days after -> 3.0
    assert future.loc[7, "festival_distance"] == 3.0
    assert future.loc[7, "festival_season_flag"] == 1.0

    # March 18: 4 days after -> 4.0 (saturated)
    assert future.loc[8, "festival_distance"] == 4.0
    assert future.loc[8, "festival_flag"] == 0.0
    assert future.loc[8, "festival_season_flag"] == 0.0

    # March 19: 5 days after -> 4.0 (saturated, NOT 99.0)
    assert future.loc[9, "festival_distance"] == 4.0
    assert future.loc[9, "festival_flag"] == 0.0
    assert future.loc[9, "festival_season_flag"] == 0.0


def test_electricity_future_forecasting_remains_unaffected(elec_features: Path) -> None:
    """Verify electricity future forecasting matrices do not include retail-only features."""
    from ml_engine.models.future_forecasting import build_future_feature_frame

    history = load_chronological_history(elec_features, "demand_mw")
    future, _ = build_future_feature_frame(
        history,
        domain="electricity",
        target_column="demand_mw",
        horizon=12,
    )
    # Electricity must NOT have festival_distance, ewma_3, payday_week injected
    assert "festival_distance" not in future.columns
    assert "ewma_3" not in future.columns
    assert "payday_week" not in future.columns
    assert len(future) == 12
