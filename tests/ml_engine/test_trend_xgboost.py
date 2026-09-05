"""Tests for Trend XGBoost (Linear Detrended Residual XGBoost Forecaster).

Verifies:
1. Linear detrending fit/predict on trended series and out-of-sample extrapolation.
2. Canonical lag detrending against historical trend offsets.
3. Conservative spike weighting on residuals.
4. Safe domain restriction (electricity-only, strictly rejected for retail).
5. Artifact persistence round-trip and exact numerical reproducibility.
6. Recursive future forecasting and capability matrix registration.
7. Explainability support via ForecastExplainer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_engine.explainability import ForecastExplainer, SUPPORTED_EXPLAINABLE_MODELS
from ml_engine.models.artifact_store import (
    compute_dataset_fingerprint,
    derive_features_id,
    is_persistence_supported,
    save_model_artifact,
    try_load_compatible_artifact,
)
from ml_engine.models.capability_matrix import (
    DEFAULT_COMPARE_MODELS,
    build_model_capability_matrix,
)
from ml_engine.models.future_forecasting import (
    FUTURE_SUPPORTED_MODELS,
    is_future_forecast_supported,
)
from ml_engine.models.trainer import (
    TrainerError,
    expected_configuration_for_request,
    run_training_pipeline,
)
from ml_engine.models.xgboost_model import XGBoostForecaster, XGBoostModelError


def test_trend_xgboost_synthetic_trend_extrapolation() -> None:
    """Verify that trend_mode='linear' extrapolates beyond training range where raw XGBoost bounds."""
    np.random.seed(42)
    periods = 100
    timestamps = pd.date_range("2024-01-01", periods=periods, freq="h")
    # Linear upward trend: slope 10 per hour
    trend_vals = 1000.0 + 10.0 * np.arange(periods)
    seasonal_vals = 50.0 * np.sin(2 * np.pi * np.arange(periods) / 24.0)
    target = trend_vals + seasonal_vals + np.random.randn(periods) * 2.0

    df = pd.DataFrame({
        "timestamp": timestamps,
        "demand_mw": target,
        "lag_1": np.roll(target, 1),
        "hour": timestamps.hour,
    })
    df.iloc[0, df.columns.get_loc("lag_1")] = target[0]

    train_df = df.iloc[:70].copy()
    eval_df = df.iloc[70:].copy()

    # Model with linear detrending
    trend_model = XGBoostForecaster(
        n_estimators=30,
        max_depth=4,
        learning_rate=0.1,
        trend_mode="linear",
        sample_weight_strategy="spike_sensitive",
        random_state=42,
    )
    trend_model.fit(train_df, target_column="demand_mw")

    assert trend_model.trend_params is not None
    assert "slope" in trend_model.trend_params
    assert "intercept" in trend_model.trend_params
    assert trend_model.trend_params["slope"] > 0

    preds_trend = trend_model.predict(eval_df)
    assert len(preds_trend) == len(eval_df)
    assert not preds_trend.isna().any()

    # In eval split, target values are well above training max:
    eval_min_target = eval_df["demand_mw"].min()
    train_max_target = train_df["demand_mw"].max()
    assert eval_min_target > train_max_target * 0.95

    # Trend model should predict above train max on late eval points
    assert preds_trend.iloc[-1] > train_max_target

    # Standard model without detrending flatlines near/below train max
    raw_model = XGBoostForecaster(
        n_estimators=30,
        max_depth=4,
        learning_rate=0.1,
        trend_mode="none",
        random_state=42,
    )
    raw_model.fit(train_df, target_column="demand_mw")
    preds_raw = raw_model.predict(eval_df)

    # Detrended model should have significantly lower RMSE on trended extrapolation
    rmse_trend = np.sqrt(np.mean((preds_trend.values - eval_df["demand_mw"].values) ** 2))
    rmse_raw = np.sqrt(np.mean((preds_raw.values - eval_df["demand_mw"].values) ** 2))
    assert rmse_trend < rmse_raw


def test_trend_xgboost_lag_detrending() -> None:
    """Verify that canonical lags are offset by their specific historical trend."""
    timestamps = pd.date_range("2024-01-01", periods=200, freq="h")
    slope = 5.0
    intercept = 500.0
    t_numeric = (timestamps - timestamps[0]).total_seconds() / 3600.0
    trend = intercept + slope * t_numeric

    train_df = pd.DataFrame({
        "timestamp": timestamps[:150],
        "demand_mw": trend[:150],
        "lag_1": np.roll(trend[:150], 1),
        "lag_24": np.roll(trend[:150], 24),
        "hour": timestamps[:150].hour,
    })

    model = XGBoostForecaster(
        n_estimators=10,
        trend_mode="linear",
        random_state=42,
    )
    model.fit(train_df, target_column="demand_mw")

    assert model.trend_params is not None
    # Slope should be approximately 5.0
    np.testing.assert_allclose(model.trend_params["slope"], 5.0, atol=0.1)

    eval_df = pd.DataFrame({
        "timestamp": timestamps[150:],
        "demand_mw": trend[150:],
        "lag_1": np.roll(trend, 1)[150:],
        "lag_24": np.roll(trend, 24)[150:],
        "hour": timestamps[150:].hour,
    })
    preds = model.predict(eval_df)
    # On a pure linear trend, detrended model should predict with near-zero error
    np.testing.assert_allclose(preds.values, eval_df["demand_mw"].values, atol=1.0)


def test_trend_xgboost_retail_domain_rejection(tmp_path) -> None:
    """Verify that trend_xgboost is rejected for retail domain at all entry points."""
    # 1. expected_configuration_for_request
    with pytest.raises(TrainerError, match="only supported for the electricity domain"):
        expected_configuration_for_request(model_key="trend_xgboost", domain="retail")

    # 2. run_training_pipeline
    dummy_features = tmp_path / "dummy_retail"
    dummy_features.mkdir(parents=True)
    with pytest.raises(TrainerError, match="only supported for the electricity domain"):
        run_training_pipeline(domain="retail", features_dir=dummy_features, model_name="trend_xgboost")

    # 3. future_forecasting support
    assert not is_future_forecast_supported("retail", "trend_xgboost")
    assert is_future_forecast_supported("electricity", "trend_xgboost")


def test_trend_xgboost_capability_registration() -> None:
    """Verify that trend_xgboost is registered in capability matrix and comparison defaults."""
    assert "trend_xgboost" in DEFAULT_COMPARE_MODELS["electricity"]
    assert "trend_xgboost" not in DEFAULT_COMPARE_MODELS["retail"]

    matrix = build_model_capability_matrix()
    trend_entry = next((item for item in matrix if item["key"] == "trend_xgboost"), None)
    assert trend_entry is not None
    assert trend_entry["electricity"] is True
    assert trend_entry["retail"] is False
    assert trend_entry["future_forecast_supported"] is True
    assert trend_entry["persistence_supported"] is True
    assert trend_entry["compare_default"] is True


def test_trend_xgboost_artifact_persistence_roundtrip(tmp_path) -> None:
    """Verify that trend_xgboost saves and loads cleanly with exact prediction parity."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-01-01", periods=100, freq="h")
    trend = 2000.0 + 3.0 * np.arange(100)
    target = trend + np.random.randn(100) * 10.0

    df = pd.DataFrame({
        "timestamp": timestamps,
        "demand_mw": target,
        "lag_1": np.roll(target, 1),
        "hour": timestamps.hour,
    })
    train_df = df.iloc[:60].copy()
    val_df = df.iloc[60:80].copy()
    test_df = df.iloc[80:].copy()

    # Save splits as parquet for fingerprinting
    feats_dir = tmp_path / "features"
    feats_dir.mkdir(parents=True)
    train_df.to_parquet(feats_dir / "train.parquet")
    val_df.to_parquet(feats_dir / "validation.parquet")
    test_df.to_parquet(feats_dir / "test.parquet")

    model = XGBoostForecaster(
        n_estimators=15,
        max_depth=3,
        learning_rate=0.1,
        trend_mode="linear",
        sample_weight_strategy="spike_sensitive",
        random_state=42,
    )
    model.fit(train_df, target_column="demand_mw")
    orig_val_pred = model.predict(val_df)
    orig_test_pred = model.predict(test_df)

    config = {
        "n_estimators": 15,
        "max_depth": 3,
        "learning_rate": 0.1,
        "trend_mode": "linear",
        "sample_weight_strategy": "spike_sensitive",
    }
    features_id = derive_features_id(feats_dir)
    ds_fp = compute_dataset_fingerprint(feats_dir)
    artifacts_dir = tmp_path / "artifacts"

    # Save
    save_info = save_model_artifact(
        model=model,
        domain="electricity",
        model_key="trend_xgboost",
        target="demand_mw",
        features_id=features_id,
        dataset_fingerprint=ds_fp,
        feature_names=["lag_1", "hour"],
        configuration=config,
        validation_metrics={"rmse": 50.0},
        test_metrics={"rmse": 55.0},
        artifacts_root=artifacts_dir,
    )
    assert save_info["metadata"]["model"] == "trend_xgboost"

    # Load
    loaded = try_load_compatible_artifact(
        domain="electricity",
        model_key="trend_xgboost",
        features_id=features_id,
        dataset_fingerprint=ds_fp,
        target="demand_mw",
        configuration=config,
        artifacts_root=artifacts_dir,
    )
    assert loaded is not None
    loaded_model = loaded["model"]
    assert isinstance(loaded_model, XGBoostForecaster)
    assert loaded_model.trend_mode == "linear"
    assert loaded_model.trend_params is not None

    loaded_val_pred = loaded_model.predict(val_df)
    loaded_test_pred = loaded_model.predict(test_df)

    pd.testing.assert_series_equal(orig_val_pred, loaded_val_pred)
    pd.testing.assert_series_equal(orig_test_pred, loaded_test_pred)


def test_trend_xgboost_explainability_support() -> None:
    """Verify that trend_xgboost integrates with ForecastExplainer and SHAP."""
    assert "trend_xgboost" in SUPPORTED_EXPLAINABLE_MODELS

    timestamps = pd.date_range("2024-01-01", periods=60, freq="h")
    df = pd.DataFrame({
        "timestamp": timestamps,
        "demand_mw": 1000.0 + np.arange(60) * 2.0,
        "lag_1": 1000.0 + np.roll(np.arange(60) * 2.0, 1),
        "hour": timestamps.hour,
    })
    train_df = df.iloc[:40].copy()
    eval_df = df.iloc[40:].copy()

    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="linear",
        random_state=42,
    )
    forecaster.fit(train_df, target_column="demand_mw")

    explainer = ForecastExplainer(forecaster, model_name="trend_xgboost")
    global_exp = explainer.explain_global(eval_df)
    assert global_exp["model"] == "trend_xgboost"
    assert "natural_language_explanation" in global_exp


def test_trend_xgboost_generate_future_forecast(tmp_path: Path) -> None:
    """Verify recursive future forecasting works natively for trend_xgboost on electricity."""
    from ml_engine.models.future_forecasting import generate_future_forecast

    feats_dir = tmp_path / "elec_feats"
    feats_dir.mkdir(parents=True, exist_ok=True)
    n = 300
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    # Linear trend + diurnal pattern
    trend = 1000.0 + 2.0 * np.arange(n)
    demand = trend + 50.0 * np.sin(2 * np.pi * np.arange(n) / 24.0)
    frame = pd.DataFrame({
        "timestamp": ts,
        "demand_mw": demand,
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "month": ts.month,
    })
    for lag in (1, 24, 48):
        frame[f"lag_{lag}"] = frame["demand_mw"].shift(lag)
    frame = frame.dropna().reset_index(drop=True)

    a, b = int(len(frame) * 0.6), int(len(frame) * 0.8)
    train_df = frame.iloc[:a].copy()
    val_df = frame.iloc[a:b].copy()
    test_df = frame.iloc[b:].copy()

    train_df.to_parquet(feats_dir / "train.parquet", index=False)
    val_df.to_parquet(feats_dir / "validation.parquet", index=False)
    test_df.to_parquet(feats_dir / "test.parquet", index=False)

    model = XGBoostForecaster(
        n_estimators=15,
        max_depth=3,
        learning_rate=0.1,
        trend_mode="linear",
        sample_weight_strategy="spike_sensitive",
        random_state=42,
    )
    model.fit(train_df, target_column="demand_mw")

    result = generate_future_forecast(
        model=model,
        model_key="trend_xgboost",
        domain="electricity",
        features_dir=feats_dir,
        horizon=24,
    )
    assert result["horizon"] == 24
    assert len(result["predictions"]) == 24
    for p in result["predictions"]:
        assert "timestamp" in p
        assert np.isfinite(p["predicted"])
        # In a growing series, predictions should be above the baseline level
        assert p["predicted"] > 800.0
