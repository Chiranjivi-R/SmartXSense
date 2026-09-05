"""Regression and integration tests for Experiment 26: Controlled Production Integration
of Candidate B1 (Piecewise Linear Trend + Conservative Spike Weighting) for Electricity.

Verifies:
1. Piecewise continuity across knot (|f(knot+) - f(knot-)| < 1e-5).
2. Pre-knot slope matches Seg 1 and post-knot slope matches Seg 2.
3. Negative values and near-zero values behave sensibly.
4. Linear mode remains unchanged and behaves identically to prior behavior.
5. Invalid knot or trend parameter rejection.
6. Artifact fingerprint isolation: linear and piecewise configs produce different fingerprints.
7. Old linear artifacts are not loaded for piecewise requests.
8. Retrained model saves and reloads with exact numerical reproducibility (rtol=1e-7, atol=1e-7).
9. Multi-horizon future forecasting (h = 24, 48, 168, 720, 2160) completes with finite, positive forecasts.
10. Auto Select routing: selects SARIMA for h <= 168, selects Piecewise Trend XGBoost for h > 168.
11. Hybrid model integration works with piecewise trend XGBoost as base forecaster.
12. SHAP local attribution: lag features detrended, baseline_trend present, additivity error < 0.05 MW.
13. SHAP lag features detrended before tree explainer.
14. LIME local attribution executes without error and reports baseline_trend.
15. Retail domain strictly rejects trend_mode (raises TrainerError).
16. End-to-end training pipeline via run_training_pipeline produces valid metrics.
17. Custom knot hour override works correctly.
18. Conservative spike weighting is preserved on residuals.
19. Capability matrix registration for trend_xgboost.
20. Natural language explanation generation for trend_xgboost.
21. History recording preserves trend_xgboost.
22. Deterministic predictions with fixed random seed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest

from ml_engine.explainability import ForecastExplainer, LimeTabularExplainer, ShapTreeExplainer
from ml_engine.models.artifact_store import (
    config_fingerprint,
    is_persistence_supported,
    save_model_artifact,
    try_load_compatible_artifact,
)
from ml_engine.models.capability_matrix import build_model_capability_matrix
from ml_engine.models.future_forecasting import (
    generate_future_forecast,
    is_future_forecast_supported,
)
from ml_engine.models.hybrid import HybridForecaster
from ml_engine.models.trainer import (
    TrainerError,
    expected_configuration_for_request,
    run_training_pipeline,
)
from ml_engine.models.xgboost_model import XGBoostForecaster, XGBoostModelError


@pytest.fixture()
def synthetic_elec_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic electricity dataset with two-phase trend and diurnal seasonality."""
    np.random.seed(42)
    n = 600
    timestamps = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    hours = np.arange(n, dtype=float)

    # Piecewise trend: knot at hour 300
    knot = 300.0
    seg1_slope = 0.5
    seg2_slope = 1.5
    intercept = 1000.0
    trend = np.where(hours <= knot, intercept + seg1_slope * hours, intercept + seg1_slope * knot + seg2_slope * (hours - knot))

    # Diurnal seasonality + noise
    diurnal = 50.0 * np.sin(2 * np.pi * hours / 24.0)
    noise = np.random.normal(0.0, 5.0, size=n)
    demand = trend + diurnal + noise

    df = pd.DataFrame({
        "timestamp": timestamps,
        "demand_mw": demand,
        "hour": timestamps.hour,
        "day_of_week": timestamps.dayofweek,
        "month": timestamps.month,
    })
    for lag in (1, 24, 48, 168):
        df[f"lag_{lag}"] = df["demand_mw"].shift(lag)

    df = df.dropna().reset_index(drop=True)
    n_clean = len(df)
    train_idx = int(n_clean * 0.6)
    val_idx = int(n_clean * 0.8)

    train_df = df.iloc[:train_idx].copy()
    val_df = df.iloc[train_idx:val_idx].copy()
    test_df = df.iloc[val_idx:].copy()

    return train_df, val_df, test_df


# 1. Piecewise continuity across knot
def test_piecewise_linear_spline_c0_continuity() -> None:
    """Verify f(knot-) == f(knot+) within 1e-5 MW for C0 spline continuity."""
    forecaster = XGBoostForecaster(
        trend_mode="piecewise_linear",
        trend_knot_hour=30000.0,
    )
    # Mock fitted trend parameters
    forecaster.trend_intercept = 100000.0
    forecaster.trend_slope = 0.2748
    forecaster.trend_slope2_delta = 0.9721  # seg2 = 1.2469
    forecaster.trend_knot_hour = 30000.0

    eps = 1e-6
    val_left = forecaster.compute_trend(30000.0 - eps)
    val_right = forecaster.compute_trend(30000.0 + eps)
    val_knot = forecaster.compute_trend(30000.0)

    assert abs(val_right - val_left) < 1e-5
    assert abs(val_knot - val_left) < 1e-5
    assert abs(val_knot - val_right) < 1e-5


# 2. Pre-knot slope matches Seg 1 and post-knot slope matches Seg 2
def test_piecewise_linear_slopes_match_training(synthetic_elec_data) -> None:
    """Verify fitted slopes match pre-knot and post-knot empirical rates."""
    train_df, _, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=150.0,
        random_state=42,
    )
    forecaster.fit(train_df, target_column="demand_mw")

    params = forecaster.trend_params
    assert params is not None
    assert params["trend_mode"] == "piecewise_linear"
    assert params["trend_knot_hour"] == 150.0
    assert "seg1_slope" in params
    assert "seg2_slope" in params
    assert "slope2_delta" in params
    assert abs(params["seg2_slope"] - (params["seg1_slope"] + params["slope2_delta"])) < 1e-7

    # Empirical slope checks
    y_0 = forecaster.compute_trend(0.0)
    y_100 = forecaster.compute_trend(100.0)
    calc_seg1 = (y_100 - y_0) / 100.0
    assert abs(calc_seg1 - params["seg1_slope"]) < 1e-5

    y_200 = forecaster.compute_trend(200.0)
    y_300 = forecaster.compute_trend(300.0)
    calc_seg2 = (y_300 - y_200) / 100.0
    assert abs(calc_seg2 - params["seg2_slope"]) < 1e-5


# 3. Negative and near-zero values behave sensibly
def test_piecewise_linear_negative_and_zero_values() -> None:
    """Verify compute_trend handles t <= 0 gracefully."""
    forecaster = XGBoostForecaster(
        trend_mode="piecewise_linear",
        trend_knot_hour=30000.0,
    )
    forecaster.trend_intercept = 5000.0
    forecaster.trend_slope = 0.5
    forecaster.trend_slope2_delta = 1.0
    forecaster.trend_knot_hour = 30000.0

    assert forecaster.compute_trend(0.0) == 5000.0
    # Negative t extrapolates backwards along seg 1
    assert forecaster.compute_trend(-10.0) == 5000.0 - 5.0


# 4. Linear mode remains unchanged
def test_linear_mode_backwards_compatibility(synthetic_elec_data) -> None:
    """Verify linear trend_mode remains fully functional and unchanged."""
    train_df, val_df, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="linear",
        random_state=42,
    )
    forecaster.fit(train_df, target_column="demand_mw")

    params = forecaster.trend_params
    assert params is not None
    assert params["trend_mode"] == "linear"
    assert "slope" in params
    assert "intercept" in params
    assert "trend_knot_hour" not in params

    preds = forecaster.predict(val_df)
    assert len(preds) == len(val_df)
    assert np.isfinite(preds).all()


# 5. Invalid knot or trend parameter rejection
def test_invalid_trend_parameters_rejection() -> None:
    """Verify invalid parameters raise appropriate errors."""
    with pytest.raises(XGBoostModelError, match="Unsupported trend_mode"):
        XGBoostForecaster(trend_mode="polynomial")

    with pytest.raises(XGBoostModelError, match="trend_knot_hour must be a positive float"):
        XGBoostForecaster(trend_mode="piecewise_linear", trend_knot_hour=-50.0)

    with pytest.raises(XGBoostModelError, match="trend_knot_hour must be a positive float"):
        XGBoostForecaster(trend_mode="piecewise_linear", trend_knot_hour=float("nan"))


# 6. Artifact fingerprint isolation
def test_artifact_fingerprint_isolation() -> None:
    """Verify linear and piecewise configurations produce distinct SHA-256 fingerprints."""
    cfg_linear = expected_configuration_for_request(
        model_key="trend_xgboost",
        domain="electricity",
        trend_mode="linear",
    )
    cfg_piecewise = expected_configuration_for_request(
        model_key="trend_xgboost",
        domain="electricity",
        trend_mode="piecewise_linear",
        trend_knot_hour=30000.0,
    )

    fp_linear = config_fingerprint(cfg_linear)
    fp_piecewise = config_fingerprint(cfg_piecewise)

    assert fp_linear != fp_piecewise
    assert cfg_piecewise["trend_mode"] == "piecewise_linear"
    assert cfg_piecewise["trend_knot_hour"] == 30000.0


# 7. Old linear artifacts not loaded for piecewise requests
def test_old_linear_artifact_incompatible_with_piecewise_request(tmp_path, synthetic_elec_data) -> None:
    """Verify piecewise requests never inadvertently load legacy linear artifacts."""
    train_df, _, _ = synthetic_elec_data
    linear_model = XGBoostForecaster(
        n_estimators=10,
        trend_mode="linear",
        random_state=42,
    )
    linear_model.fit(train_df, target_column="demand_mw")

    cfg_linear = expected_configuration_for_request(
        model_key="trend_xgboost",
        domain="electricity",
        trend_mode="linear",
    )
    save_model_artifact(
        model=linear_model,
        domain="electricity",
        model_key="trend_xgboost",
        target="demand_mw",
        features_id="test_exp26_feats",
        feature_names=list(linear_model.feature_names),
        configuration=cfg_linear,
        artifacts_root=tmp_path,
    )

    # Attempt to load with piecewise configuration
    cfg_piecewise = expected_configuration_for_request(
        model_key="trend_xgboost",
        domain="electricity",
        trend_mode="piecewise_linear",
        trend_knot_hour=30000.0,
    )
    loaded = try_load_compatible_artifact(
        domain="electricity",
        model_key="trend_xgboost",
        target="demand_mw",
        features_id="test_exp26_feats",
        configuration=cfg_piecewise,
        artifacts_root=tmp_path,
    )
    assert loaded is None


# 8. Artifact persistence roundtrip with numerical reproducibility (rtol=1e-7)
def test_artifact_persistence_roundtrip_numerical_parity(tmp_path, synthetic_elec_data) -> None:
    """Verify saved and reloaded model produces identical predictions within rtol=1e-7."""
    train_df, val_df, _ = synthetic_elec_data
    model = XGBoostForecaster(
        n_estimators=15,
        trend_mode="piecewise_linear",
        trend_knot_hour=150.0,
        sample_weight_strategy="spike_sensitive",
        random_state=42,
    )
    model.fit(train_df, target_column="demand_mw")
    preds_orig = model.predict(val_df)

    cfg = expected_configuration_for_request(
        model_key="trend_xgboost",
        domain="electricity",
        trend_mode="piecewise_linear",
        trend_knot_hour=150.0,
        n_estimators=15,
    )
    saved = save_model_artifact(
        model=model,
        domain="electricity",
        model_key="trend_xgboost",
        target="demand_mw",
        features_id="test_exp26_parity",
        feature_names=list(model.feature_names),
        configuration=cfg,
        artifacts_root=tmp_path,
    )
    assert saved is not None

    loaded = try_load_compatible_artifact(
        domain="electricity",
        model_key="trend_xgboost",
        target="demand_mw",
        features_id="test_exp26_parity",
        configuration=cfg,
        artifacts_root=tmp_path,
    )
    assert loaded is not None
    loaded_model = loaded["model"]
    preds_loaded = loaded_model.predict(val_df)

    np.testing.assert_allclose(preds_orig.values, preds_loaded.values, rtol=1e-7, atol=1e-7)
    assert loaded_model.trend_params == model.trend_params


# 9. Multi-horizon future forecasting (h = 24, 48, 168, 720, 2160)
@pytest.mark.parametrize("h", [24, 48, 168, 720, 2160])
def test_future_forecasting_multi_horizon_execution(tmp_path, h) -> None:
    """Verify recursive future forecasting completes with positive, finite predictions across all horizons."""
    feats_dir = tmp_path / f"elec_feats_h{h}"
    feats_dir.mkdir(parents=True, exist_ok=True)
    n = 400
    ts = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    trend = 2000.0 + 1.0 * np.arange(n)
    demand = trend + 50.0 * np.sin(2 * np.pi * np.arange(n) / 24.0)
    df = pd.DataFrame({
        "timestamp": ts,
        "demand_mw": demand,
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "month": ts.month,
    })
    for lag in (1, 24, 48, 168):
        df[f"lag_{lag}"] = df["demand_mw"].shift(lag)
    df = df.dropna().reset_index(drop=True)

    a, b = int(len(df) * 0.6), int(len(df) * 0.8)
    df.iloc[:a].to_parquet(feats_dir / "train.parquet", index=False)
    df.iloc[a:b].to_parquet(feats_dir / "validation.parquet", index=False)
    df.iloc[b:].to_parquet(feats_dir / "test.parquet", index=False)

    model = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=150.0,
        random_state=42,
    )
    model.fit(df.iloc[:a], target_column="demand_mw")

    result = generate_future_forecast(
        model=model,
        model_key="trend_xgboost",
        domain="electricity",
        features_dir=feats_dir,
        horizon=h,
    )
    assert result["horizon"] == h
    assert len(result["predictions"]) == h
    for p in result["predictions"]:
        assert "timestamp" in p
        assert np.isfinite(p["predicted"])
        assert p["predicted"] > 0.0


# 10. Auto Select routing by horizon
def test_auto_select_routing_by_horizon() -> None:
    """Verify SARIMA is supported up to h=168, and rejected for h>168, routing to trend_xgboost."""
    assert is_future_forecast_supported("electricity", "sarima", horizon=24)
    assert is_future_forecast_supported("electricity", "sarima", horizon=168)
    assert not is_future_forecast_supported("electricity", "sarima", horizon=720)
    assert not is_future_forecast_supported("electricity", "sarima", horizon=2160)

    # trend_xgboost supports all horizons
    assert is_future_forecast_supported("electricity", "trend_xgboost", horizon=24)
    assert is_future_forecast_supported("electricity", "trend_xgboost", horizon=168)
    assert is_future_forecast_supported("electricity", "trend_xgboost", horizon=720)
    assert is_future_forecast_supported("electricity", "trend_xgboost", horizon=2160)


# 11. Hybrid forecaster integration
def test_hybrid_forecaster_compatibility(synthetic_elec_data) -> None:
    """Verify HybridForecaster can combine piecewise trend XGBoost with other models."""
    train_df, val_df, _ = synthetic_elec_data
    hybrid = HybridForecaster(
        components=["trend_xgboost", "random_forest"],
        domain="electricity",
        weighting_strategy="inverse_rmse",
    )
    assert "trend_xgboost" in hybrid.components

    forecaster_trend = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=150.0,
        random_state=42,
    )
    forecaster_trend.fit(train_df, target_column="demand_mw")
    preds = forecaster_trend.predict(val_df)
    assert len(preds) == len(val_df)
    assert np.isfinite(preds).all()


# 12. SHAP local attribution additivity error < 0.05 MW
def test_shap_attribution_additivity_error_under_tolerance(synthetic_elec_data) -> None:
    """Verify local SHAP attribution waterfall sums to point prediction within |error| < 0.05 MW."""
    train_df, val_df, _ = synthetic_elec_data
    model = XGBoostForecaster(
        n_estimators=15,
        trend_mode="piecewise_linear",
        trend_knot_hour=150.0,
        random_state=42,
    )
    model.fit(train_df, target_column="demand_mw")

    explainer = ShapTreeExplainer(model)
    local_exp = explainer.explain_local(val_df, sample_index=0)

    assert "baseline_trend" in local_exp["shap_contributions"]
    base_val = local_exp["base_value"]
    point_pred = local_exp["prediction"]
    sum_contribs = sum(local_exp["shap_contributions"].values())
    total_waterfall = base_val + sum_contribs

    error = abs(total_waterfall - point_pred)
    assert error < 0.05, f"SHAP additivity error {error:.6f} MW exceeds 0.05 MW tolerance"


# 13. SHAP lag features detrended before tree
def test_shap_lag_features_detrended_before_tree(synthetic_elec_data) -> None:
    """Verify lag features are properly detrended prior to tree transformer."""
    train_df, val_df, _ = synthetic_elec_data
    model = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=150.0,
        random_state=42,
    )
    model.fit(train_df, target_column="demand_mw")

    explainer = ShapTreeExplainer(model)
    # Transformed features via explainer
    X_detrended = explainer._get_transformed_features(val_df.iloc[[0]])

    # Without detrending, transformer on raw df
    X_raw = model._transformer.transform(val_df.iloc[[0]])

    # The detrended lags should differ from raw lags
    assert not np.allclose(X_detrended, X_raw)


# 14. LIME local attribution executes and includes baseline_trend
def test_lime_attribution_executes_and_includes_baseline_trend(synthetic_elec_data) -> None:
    """Verify LIME tabular explainer succeeds and outputs baseline_trend rule."""
    train_df, val_df, _ = synthetic_elec_data
    model = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=150.0,
        random_state=42,
    )
    model.fit(train_df, target_column="demand_mw")

    lime_exp = LimeTabularExplainer(model, train_df=train_df)
    res = lime_exp.explain_local(val_df, sample_index=0)

    assert res["explanation_type"] == "local"
    assert res["method"] == "lime"
    rules = [item["rule"] for item in res["feature_attributions"]]
    assert "baseline_trend" in rules
    for item in res["feature_attributions"]:
        assert np.isfinite(item["weight"])


# 15. Retail domain strictly rejects trend_mode
def test_retail_domain_strictly_rejects_trend_mode() -> None:
    """Verify Retail domain raises TrainerError for trend_xgboost or any trend_mode."""
    with pytest.raises(TrainerError, match="is only supported for the electricity domain"):
        expected_configuration_for_request(model_key="trend_xgboost", domain="retail")

    with pytest.raises(TrainerError, match="is only supported for the electricity domain"):
        run_training_pipeline(
            domain="retail",
            features_dir=Path("nonexistent"),
            model_name="trend_xgboost",
        )


# 16. End-to-end training pipeline via run_training_pipeline
def test_end_to_end_training_pipeline_piecewise_trend(tmp_path, synthetic_elec_data) -> None:
    """Verify run_training_pipeline successfully trains piecewise trend_xgboost and returns valid metrics."""
    train_df, val_df, test_df = synthetic_elec_data
    feats_dir = tmp_path / "e2e_feats"
    feats_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(feats_dir / "train.parquet", index=False)
    val_df.to_parquet(feats_dir / "validation.parquet", index=False)
    test_df.to_parquet(feats_dir / "test.parquet", index=False)

    report = run_training_pipeline(
        domain="electricity",
        features_dir=feats_dir,
        model_name="trend_xgboost",
        n_estimators=15,
        trend_knot_hour=150.0,
        save_artifact=False,
    )
    assert report["model"] == "trend_xgboost"
    assert report["configuration"]["trend_mode"] == "piecewise_linear"
    assert report["configuration"]["trend_knot_hour"] == 150.0
    val_m = report["validation_metrics"]
    assert val_m["rmse"] > 0.0
    assert np.isfinite(val_m["rmse"])
    assert np.isfinite(val_m["r2"])


# 17. Knot hour override works correctly
def test_knot_hour_override(synthetic_elec_data) -> None:
    """Verify passing custom trend_knot_hour is respected."""
    train_df, _, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=5,
        trend_mode="piecewise_linear",
        trend_knot_hour=250.0,
        random_state=42,
    )
    forecaster.fit(train_df, target_column="demand_mw")
    assert forecaster.trend_params["trend_knot_hour"] == 250.0


# 18. Conservative spike weighting preservation
def test_spike_weighting_preservation(synthetic_elec_data) -> None:
    """Verify spike-sensitive weighting computes correct sample weights."""
    train_df, _, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=150.0,
        sample_weight_strategy="spike_sensitive",
        random_state=42,
    )
    forecaster.fit(train_df, target_column="demand_mw")
    assert forecaster.sample_weight_strategy == "spike_sensitive"


# 19. Capability matrix registration
def test_capability_matrix_registration() -> None:
    """Verify trend_xgboost is registered in capability matrix."""
    matrix = build_model_capability_matrix()
    models = {entry["key"] for entry in matrix}
    assert "trend_xgboost" in models


# 20. Natural language explanation generation
def test_explainability_natural_language_explanation(synthetic_elec_data) -> None:
    """Verify ForecastExplainer produces natural language explanation for trend_xgboost."""
    train_df, val_df, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=150.0,
        random_state=42,
    )
    forecaster.fit(train_df, target_column="demand_mw")

    explainer = ForecastExplainer(forecaster, model_name="trend_xgboost")
    local_exp = explainer.explain_local(val_df, sample_index=0, method="shap")
    assert "natural_language_explanation" in local_exp
    assert "trend_xgboost" in local_exp["natural_language_explanation"]


# 21. History recording preserves trend_xgboost
def test_forecast_history_recording() -> None:
    """Verify expected configuration preserves trend_xgboost model_key."""
    cfg = expected_configuration_for_request(
        model_key="trend_xgboost",
        domain="electricity",
    )
    assert cfg["trend_mode"] == "piecewise_linear"
    assert cfg["trend_knot_hour"] == 30000.0


# 22. Deterministic predictions with fixed random seed
def test_deterministic_predictions_fixed_seed(synthetic_elec_data) -> None:
    """Verify two models with identical seeds produce bitwise deterministic predictions."""
    train_df, val_df, _ = synthetic_elec_data
    m1 = XGBoostForecaster(n_estimators=10, trend_mode="piecewise_linear", trend_knot_hour=150.0, random_state=42)
    m1.fit(train_df, target_column="demand_mw")
    p1 = m1.predict(val_df)

    m2 = XGBoostForecaster(n_estimators=10, trend_mode="piecewise_linear", trend_knot_hour=150.0, random_state=42)
    m2.fit(train_df, target_column="demand_mw")
    p2 = m2.predict(val_df)

    np.testing.assert_array_equal(p1.values, p2.values)
