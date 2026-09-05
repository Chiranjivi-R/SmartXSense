"""Tests for SARIMA (0,1,1)(0,1,1,24) Safe Production Hardening.

Verifies:
1. Configuration fingerprinting and persistence uniqueness vs baseline ARIMA.
2. Parameter validation (order, seasonal_order, training_window).
3. Domain safety (electricity-only, strictly rejected for retail).
4. Artifact persistence round-trip and exact numerical reproducibility.
5. Horizon-aware capability restriction (h <= 168 allowed; h > 168 rejected).
6. Auto Select horizon-aware candidate filtering.
7. Hybrid ensemble registration and validation.
8. Capability matrix registration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_engine.explainability import SUPPORTED_EXPLAINABLE_MODELS
from ml_engine.models.arima import ARIMAForecaster, ARIMAModelError
from ml_engine.models.artifact_store import (
    config_fingerprint,
    is_persistence_supported,
    predict_with_fitted_model,
    save_model_artifact,
    try_load_compatible_artifact,
)
from ml_engine.models.capability_matrix import (
    DEFAULT_COMPARE_MODELS,
    build_model_capability_matrix,
)
from ml_engine.models.future_forecasting import (
    FUTURE_SUPPORTED_MODELS,
    FutureForecastError,
    future_support_error,
    generate_future_forecast,
    is_future_forecast_supported,
)
from ml_engine.models.hybrid import HybridForecaster, HybridModelError, SUPPORTED_COMPONENTS
from ml_engine.models.trainer import (
    TrainerError,
    expected_configuration_for_request,
)


def test_sarima_configuration_fingerprint_uniqueness() -> None:
    """Verify ARIMA and SARIMA configurations produce distinct, complete fingerprints."""
    arima_cfg = expected_configuration_for_request(model_key="arima", domain="electricity")
    assert arima_cfg == {"order": [1, 1, 1]}
    assert "seasonal_order" not in arima_cfg
    assert "training_window" not in arima_cfg

    sarima_cfg = expected_configuration_for_request(model_key="sarima", domain="electricity")
    assert sarima_cfg == {
        "order": [0, 1, 1],
        "seasonal_order": [0, 1, 1, 24],
        "training_window": 8760,
    }

    fp_arima = config_fingerprint(arima_cfg)
    fp_sarima = config_fingerprint(sarima_cfg)
    assert fp_arima != fp_sarima

    # Verify custom SARIMA config
    custom_cfg = expected_configuration_for_request(
        model_key="sarima",
        domain="electricity",
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        training_window=4380,
    )
    assert custom_cfg == {
        "order": [1, 1, 1],
        "seasonal_order": [1, 1, 1, 12],
        "training_window": 4380,
    }
    assert config_fingerprint(custom_cfg) != fp_sarima


def test_sarima_parameter_validation() -> None:
    """Verify robust validation of order, seasonal_order, and training_window."""
    # Invalid seasonal_order length
    with pytest.raises(ARIMAModelError, match="4-element tuple"):
        ARIMAForecaster(seasonal_order=(0, 1, 1))  # type: ignore[arg-type]

    # Negative seasonal parameters
    with pytest.raises(ARIMAModelError, match="non-negative integer"):
        ARIMAForecaster(seasonal_order=(0, -1, 1, 24))

    # Invalid seasonal period s
    with pytest.raises(ARIMAModelError, match="positive integer >= 1"):
        ARIMAForecaster(seasonal_order=(0, 1, 1, 0))

    # Non-integer seasonal_order
    with pytest.raises(ARIMAModelError, match="non-negative integer"):
        ARIMAForecaster(seasonal_order=(0, 1, 1, 24.5))  # type: ignore[arg-type]

    # Invalid training_window
    with pytest.raises(ARIMAModelError, match="training_window must be a positive integer"):
        ARIMAForecaster(training_window=0)

    with pytest.raises(ARIMAModelError, match="training_window must be a positive integer"):
        ARIMAForecaster(training_window=-100)


def test_sarima_domain_safety_rejection() -> None:
    """Verify SARIMA is strictly rejected for the retail domain across all entrypoints."""
    with pytest.raises(TrainerError, match="Model 'sarima' is only supported for the electricity domain"):
        expected_configuration_for_request(model_key="sarima", domain="retail")

    with pytest.raises(HybridModelError, match="SARIMA model is not supported for the 'retail' domain"):
        HybridForecaster.validate_components(["random_forest", "sarima"], domain="retail")

    assert not is_future_forecast_supported("retail", "sarima")
    assert "retail domain" in future_support_error("retail", "sarima")


def test_sarima_horizon_aware_capability() -> None:
    """Verify SARIMA future forecast is supported up to h=168 and rejected for h>168."""
    assert is_future_forecast_supported("electricity", "sarima", horizon=24)
    assert is_future_forecast_supported("electricity", "sarima", horizon=48)
    assert is_future_forecast_supported("electricity", "sarima", horizon=168)
    assert not is_future_forecast_supported("electricity", "sarima", horizon=169)
    assert not is_future_forecast_supported("electricity", "sarima", horizon=720)
    assert not is_future_forecast_supported("electricity", "sarima", horizon=2160)

    # Error message contains informative guidance
    err_msg = future_support_error("electricity", "sarima", horizon=720)
    assert "restricted to horizon <= 168" in err_msg
    assert "trend_xgboost" in err_msg


def test_sarima_synthetic_fit_persistence_roundtrip(tmp_path) -> None:
    """Verify SARIMA fits on synthetic seasonal data, saves, loads, and produces identical predictions."""
    np.random.seed(42)
    n = 120
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    seasonal_pattern = 100.0 * np.sin(2 * np.pi * np.arange(n) / 24.0)
    series = 5000.0 + seasonal_pattern + np.random.randn(n) * 5.0

    train_df = pd.DataFrame({"timestamp": timestamps[:80], "demand_mw": series[:80]})
    val_df = pd.DataFrame({"timestamp": timestamps[80:100], "demand_mw": series[80:100]})
    test_df = pd.DataFrame({"timestamp": timestamps[100:], "demand_mw": series[100:]})

    forecaster = ARIMAForecaster(
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 24),
        training_window=60,
    )
    forecaster.fit(train_df, target_column="demand_mw")
    val_pred = forecaster.predict(val_df, target_column="demand_mw", history=train_df["demand_mw"])
    assert len(val_pred) == 20
    assert np.isfinite(val_pred).all()

    # Verify persistence
    assert is_persistence_supported("sarima")
    config = {
        "order": [0, 1, 1],
        "seasonal_order": [0, 1, 1, 24],
        "training_window": 60,
    }
    saved = save_model_artifact(
        model=forecaster,
        domain="electricity",
        model_key="sarima",
        target="demand_mw",
        features_id="test_sarima_features",
        feature_names=[],
        configuration=config,
        artifacts_root=tmp_path,
    )
    assert "artifact_dir" in saved
    assert "relative_key" in saved

    loaded = try_load_compatible_artifact(
        domain="electricity",
        model_key="sarima",
        target="demand_mw",
        features_id="test_sarima_features",
        configuration=config,
        artifacts_root=tmp_path,
    )
    assert loaded is not None

    loaded_model = loaded["model"]
    pred_res = predict_with_fitted_model(
        model_key="sarima",
        model=loaded_model,
        domain="electricity",
        target_column="demand_mw",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
    np.testing.assert_allclose(pred_res["val_pred"].values, val_pred.values, rtol=1e-5)


def test_sarima_future_forecast_generation_and_horizon_rejection(tmp_path) -> None:
    """Verify SARIMA future forecast generation for h<=168 and FutureForecastError for h>168."""
    np.random.seed(42)
    n = 100
    timestamps = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    seasonal_pattern = 100.0 * np.sin(2 * np.pi * np.arange(n) / 24.0)
    series = 5000.0 + seasonal_pattern + np.random.randn(n) * 5.0

    history_df = pd.DataFrame({"timestamp": timestamps, "demand_mw": series})
    # Save mock train/validation/test parquet files for load_chronological_history
    fdir = tmp_path / "features"
    fdir.mkdir()
    history_df.iloc[:60].to_parquet(fdir / "train.parquet")
    history_df.iloc[60:80].to_parquet(fdir / "validation.parquet")
    history_df.iloc[80:].to_parquet(fdir / "test.parquet")

    forecaster = ARIMAForecaster(
        order=(0, 1, 1),
        seasonal_order=(0, 1, 1, 24),
        training_window=80,
    )
    forecaster.fit(history_df.iloc[:60], target_column="demand_mw")

    # h=24 valid future forecast
    res_24 = generate_future_forecast(
        model=forecaster,
        model_key="sarima",
        domain="electricity",
        features_dir=fdir,
        horizon=24,
    )
    assert res_24["forecast_type"] == "future"
    assert len(res_24["predictions"]) == 24
    assert res_24["predictions"][0]["timestamp"] > timestamps[-1].isoformat()

    # h=200 exceeds SARIMA horizon cap of 168
    with pytest.raises(FutureForecastError, match="restricted to horizon <= 168"):
        generate_future_forecast(
            model=forecaster,
            model_key="sarima",
            domain="electricity",
            features_dir=fdir,
            horizon=200,
        )


def test_sarima_hybrid_ensemble_compatibility() -> None:
    """Verify SARIMA is accepted as a valid component in HybridForecaster for electricity."""
    assert "sarima" in SUPPORTED_COMPONENTS
    validated = HybridForecaster.validate_components(["random_forest", "sarima"], domain="electricity")
    assert validated == ["random_forest", "sarima"]

    # HybridForecaster instantiation
    hf = HybridForecaster(components=["random_forest", "sarima"], domain="electricity")
    assert hf.components == ["random_forest", "sarima"]


def test_sarima_capability_matrix_registration() -> None:
    """Verify SARIMA is correctly registered in capability matrix."""
    matrix = build_model_capability_matrix()
    sarima_entries = [m for m in matrix if m["key"] == "sarima"]
    assert len(sarima_entries) == 1

    entry = sarima_entries[0]
    assert entry["name"] == "SARIMA"
    assert entry["model_type"] == "statistical"
    assert entry["electricity"] is True
    assert entry["retail"] is False
    assert entry["test_forecast"] is True
    assert entry["future_forecast_supported"] is True
    assert entry["future_forecast_domains"] == ["electricity"]
    assert entry["persistence_supported"] is True
    assert entry["compare_default"] is False
    assert entry["explainability_supported"] is False  # statistical fallback, like ARIMA


def test_sarima_auto_select_horizon_filtering() -> None:
    """Verify Auto Select eligible candidate filtering across short and long horizons."""
    ranking = [
        {"model": "sarima", "rank": 1, "validation_metrics": {"rmse": 1401.19}},
        {"model": "trend_xgboost", "rank": 2, "validation_metrics": {"rmse": 3292.68}},
        {"model": "arima", "rank": 3, "validation_metrics": {"rmse": 3699.80}},
    ]

    # At short horizon (h=24, h=48, h=168), SARIMA is eligible and top-ranked
    for h in (24, 48, 168):
        eligible = [r["model"] for r in ranking if is_future_forecast_supported("electricity", r["model"], horizon=h)]
        assert eligible[0] == "sarima"

    # At long horizon (h=720, h=2160), SARIMA is filtered out, trend_xgboost selected
    for h in (720, 2160):
        eligible = [r["model"] for r in ranking if is_future_forecast_supported("electricity", r["model"], horizon=h)]
        assert "sarima" not in eligible
        assert eligible[0] == "trend_xgboost"

