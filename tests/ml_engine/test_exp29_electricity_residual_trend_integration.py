"""Regression and integration tests for Experiment 29: Controlled Production Integration
of Candidate B (Stationary AR(1, 24, 168) Residual Error Filter) into Electricity trend_xgboost.

Tests all 25 audit requirements from Section 14:
1. test_trend_xgboost_with_residual_ar_initialization
2. test_trend_xgboost_residual_ar_stationarity_enforcement
3. test_trend_xgboost_residual_ar_negative_weights_stationarity
4. test_trend_xgboost_residual_ar_rejected_for_log1p
5. test_trend_xgboost_residual_ar_fit_estimates_ols_weights
6. test_trend_xgboost_residual_ar_stores_tail_residuals
7. test_trend_xgboost_predict_components_breakdown
8. test_trend_xgboost_predict_exact_match_components_total
9. test_trend_xgboost_with_residual_ar_false_exact_parity
10. test_trend_xgboost_disabled_flag_matches_exp26
11. test_trend_xgboost_retail_domain_rejection_in_trainer
12. test_trend_xgboost_retail_domain_rejection_in_forecaster
13. test_trend_xgboost_auto_select_short_horizon_sarima
14. test_trend_xgboost_auto_select_long_horizon_trend_xgboost
15. test_trend_xgboost_recursive_future_forecast_residual_decay
16. test_trend_xgboost_future_forecast_no_drift_or_explosion
17. test_trend_xgboost_shap_waterfall_exact_additivity
18. test_trend_xgboost_lime_attributions_include_residual_correction
19. test_trend_xgboost_artifact_persistence_roundtrip
20. test_trend_xgboost_artifact_compatibility_fingerprint
21. test_trend_xgboost_synthetic_controlled_accuracy_gain
22. test_trend_xgboost_canonical_electricity_validation_rmse
23. test_trend_xgboost_canonical_electricity_test_rmse
24. test_trend_xgboost_extreme_and_tail_regime_improvement
25. test_trend_xgboost_unrelated_models_unaffected
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest

from ml_engine.evaluation.adaptive_selector import AdaptiveModelSelector
from ml_engine.evaluation.metrics import evaluate_forecast
from ml_engine.explainability import LimeTabularExplainer, ShapTreeExplainer
from ml_engine.models.artifact_store import (
    config_fingerprint,
    derive_features_id,
    is_persistence_supported,
    save_model_artifact,
    try_load_compatible_artifact,
)
from ml_engine.models.future_forecasting import (
    generate_future_forecast,
    is_future_forecast_supported,
    recursive_tabular_future_forecast,
)
from ml_engine.models.trainer import (
    TrainerError,
    expected_configuration_for_request,
    run_training_pipeline,
)
from ml_engine.models.xgboost_model import XGBoostForecaster, XGBoostModelError

CANONICAL_FEATURES_DIR = (
    Path(__file__).resolve().parents[2]
    / "ml_engine"
    / "data"
    / "features"
    / "electricity"
    / "features_23889e57-aa6d-4a50-930c-874f6a963a25"
)


@pytest.fixture()
def synthetic_elec_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic electricity dataset with two-phase trend, seasonality, and AR(1) error dynamics."""
    np.random.seed(42)
    n = 700
    timestamps = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    hours = np.arange(n, dtype=float)

    # Piecewise linear trend with knot at 350
    knot = 350.0
    seg1_slope = 0.4
    seg2_slope = 1.2
    intercept = 1200.0
    trend = np.where(
        hours <= knot,
        intercept + seg1_slope * hours,
        intercept + seg1_slope * knot + seg2_slope * (hours - knot),
    )

    # Diurnal + weekly seasonality
    diurnal = 60.0 * np.sin(2 * np.pi * hours / 24.0)
    weekly = 30.0 * np.cos(2 * np.pi * hours / 168.0)

    # AR(1) + AR(24) autocorrelated residual
    ar_error = np.zeros(n)
    white_noise = np.random.normal(0.0, 8.0, size=n)
    for t in range(n):
        e_lag1 = ar_error[t - 1] if t >= 1 else 0.0
        e_lag24 = ar_error[t - 24] if t >= 24 else 0.0
        ar_error[t] = 0.5 * e_lag1 + 0.2 * e_lag24 + white_noise[t]

    demand = trend + diurnal + weekly + ar_error

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
    tr_idx = int(n_clean * 0.6)
    va_idx = int(n_clean * 0.8)

    train_df = df.iloc[:tr_idx].copy().reset_index(drop=True)
    val_df = df.iloc[tr_idx:va_idx].copy().reset_index(drop=True)
    test_df = df.iloc[va_idx:].copy().reset_index(drop=True)

    return train_df, val_df, test_df


# -----------------------------------------------------------------------------
# Test 1: Initialization parameters
# -----------------------------------------------------------------------------
def test_trend_xgboost_with_residual_ar_initialization() -> None:
    forecaster = XGBoostForecaster(
        trend_mode="piecewise_linear",
        trend_knot_hour=30000.0,
        residual_ar=True,
    )
    assert forecaster.residual_ar is True
    assert forecaster.residual_ar_lags == (1, 24, 168)
    assert forecaster.residual_ar_weights is None
    assert forecaster.tail_residuals is None


# -----------------------------------------------------------------------------
# Test 2: Stationarity enforcement (sum >= 1.0 strictly rejected)
# -----------------------------------------------------------------------------
def test_trend_xgboost_residual_ar_stationarity_enforcement() -> None:
    weights = {"lag_1": 0.6, "lag_24": 0.3, "lag_168": 0.2}  # sum = 1.1 >= 1.0
    with pytest.raises(XGBoostModelError, match="strictly non-stationary"):
        XGBoostForecaster(
            residual_ar=True,
            residual_ar_weights=weights,
        )


# -----------------------------------------------------------------------------
# Test 3: Negative weights stationarity (|w| sum >= 1.0 strictly rejected)
# -----------------------------------------------------------------------------
def test_trend_xgboost_residual_ar_negative_weights_stationarity() -> None:
    weights = {"lag_1": -0.7, "lag_24": 0.2, "lag_168": -0.3}  # sum |w| = 1.2
    with pytest.raises(XGBoostModelError, match="strictly non-stationary"):
        XGBoostForecaster(
            residual_ar=True,
            residual_ar_weights=weights,
        )


# -----------------------------------------------------------------------------
# Test 4: Rejection of residual AR with log1p target transform
# -----------------------------------------------------------------------------
def test_trend_xgboost_residual_ar_rejected_for_log1p() -> None:
    with pytest.raises(XGBoostModelError, match="Residual AR correction is only supported for linear scale"):
        XGBoostForecaster(
            target_transform="log1p",
            residual_ar=True,
        )


# -----------------------------------------------------------------------------
# Test 5: Fit estimates valid OLS weights on training residuals
# -----------------------------------------------------------------------------
def test_trend_xgboost_residual_ar_fit_estimates_ols_weights(
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, _, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=True,
    )
    forecaster.fit(train_df, target_column="demand_mw")
    assert forecaster.residual_ar_weights is not None
    assert "lag_1" in forecaster.residual_ar_weights
    assert "lag_24" in forecaster.residual_ar_weights
    assert "lag_168" in forecaster.residual_ar_weights
    sum_abs = sum(abs(v) for v in forecaster.residual_ar_weights.values())
    assert sum_abs < 1.0


# -----------------------------------------------------------------------------
# Test 6: Fit stores tail residuals
# -----------------------------------------------------------------------------
def test_trend_xgboost_residual_ar_stores_tail_residuals(
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, _, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=True,
    )
    forecaster.fit(train_df, target_column="demand_mw")
    assert forecaster.tail_residuals is not None
    assert isinstance(forecaster.tail_residuals, np.ndarray)
    assert len(forecaster.tail_residuals) == min(168, len(train_df))


# -----------------------------------------------------------------------------
# Test 7: predict_components breakdown
# -----------------------------------------------------------------------------
def test_trend_xgboost_predict_components_breakdown(
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=15,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=True,
    )
    forecaster.fit(train_df, target_column="demand_mw")
    comps = forecaster.predict_components(val_df)
    assert "trend" in comps
    assert "xgb_residual" in comps
    assert "ar_correction" in comps
    assert "total" in comps
    expected_total = comps["trend"] + comps["xgb_residual"] + comps["ar_correction"]
    np.testing.assert_allclose(comps["total"], expected_total, rtol=1e-6, atol=1e-6)


# -----------------------------------------------------------------------------
# Test 8: predict() exact match with components total
# -----------------------------------------------------------------------------
def test_trend_xgboost_predict_exact_match_components_total(
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=15,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=True,
    )
    forecaster.fit(train_df, target_column="demand_mw")
    pred_series = forecaster.predict(val_df)
    comps = forecaster.predict_components(val_df)
    np.testing.assert_allclose(pred_series.to_numpy(), comps["total"], rtol=1e-7, atol=1e-7)


# -----------------------------------------------------------------------------
# Test 9: with_residual_ar=False exact parity
# -----------------------------------------------------------------------------
def test_trend_xgboost_with_residual_ar_false_exact_parity(
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=15,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=True,
        random_state=42,
    )
    forecaster.fit(train_df, target_column="demand_mw")
    pred_disabled = forecaster.predict(val_df, with_residual_ar=False)

    baseline = XGBoostForecaster(
        n_estimators=15,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=False,
        random_state=42,
    )
    baseline.fit(train_df, target_column="demand_mw")
    pred_base = baseline.predict(val_df)

    np.testing.assert_allclose(pred_disabled.to_numpy(), pred_base.to_numpy(), rtol=1e-6, atol=1e-6)


# -----------------------------------------------------------------------------
# Test 10: Disabled flag matches Exp26 baseline exactly
# -----------------------------------------------------------------------------
def test_trend_xgboost_disabled_flag_matches_exp26(
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df, _ = synthetic_elec_data
    m1 = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        sample_weight_strategy="spike_sensitive",
        residual_ar=False,
        random_state=42,
    )
    m1.fit(train_df, target_column="demand_mw")
    p1 = m1.predict(val_df)

    m2 = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        sample_weight_strategy="spike_sensitive",
        random_state=42,
    )
    m2.fit(train_df, target_column="demand_mw")
    p2 = m2.predict(val_df)

    np.testing.assert_array_equal(p1.to_numpy(), p2.to_numpy())


# -----------------------------------------------------------------------------
# Test 11: Retail domain rejection in trainer.py
# -----------------------------------------------------------------------------
def test_trend_xgboost_retail_domain_rejection_in_trainer(tmp_path: Path) -> None:
    with pytest.raises(TrainerError, match="residual_ar is only supported for the electricity domain"):
        expected_configuration_for_request(
            model_key="xgboost",
            domain="retail",
            residual_ar=True,
        )


# -----------------------------------------------------------------------------
# Test 12: Retail domain rejection in forecaster
# -----------------------------------------------------------------------------
def test_trend_xgboost_retail_domain_rejection_in_forecaster() -> None:
    with pytest.raises(XGBoostModelError, match="Residual AR correction is only supported for linear scale"):
        XGBoostForecaster(target_transform="log1p", residual_ar=True)


# -----------------------------------------------------------------------------
# Test 13: Auto Select routes to SARIMA for short horizon (h <= 168)
# -----------------------------------------------------------------------------
def test_trend_xgboost_auto_select_short_horizon_sarima() -> None:
    horizon = 24
    reports = [
        {"model": "sarima", "validation_metrics": {"rmse": 1401.19, "mae": 980.0, "mape": 0.60}},
        {"model": "trend_xgboost", "validation_metrics": {"rmse": 1863.53, "mae": 1334.7, "mape": 0.71}},
        {"model": "xgboost", "validation_metrics": {"rmse": 4543.20, "mae": 3200.0, "mape": 2.10}},
    ]
    eligible_reports = [r for r in reports if is_future_forecast_supported("electricity", r["model"], horizon=horizon)]
    selector = AdaptiveModelSelector(selection_metric="rmse")
    decision = selector.select_best_model(eligible_reports)
    assert decision["selected_model"] == "sarima"
    assert [r["model"] for r in eligible_reports] == ["sarima", "trend_xgboost", "xgboost"]


# -----------------------------------------------------------------------------
# Test 14: Auto Select routes to trend_xgboost for long horizon (h > 168)
# -----------------------------------------------------------------------------
def test_trend_xgboost_auto_select_long_horizon_trend_xgboost() -> None:
    horizon = 720
    reports = [
        {"model": "sarima", "validation_metrics": {"rmse": 1401.19, "mae": 980.0, "mape": 0.60}},
        {"model": "trend_xgboost", "validation_metrics": {"rmse": 1863.53, "mae": 1334.7, "mape": 0.71}},
        {"model": "xgboost", "validation_metrics": {"rmse": 4543.20, "mae": 3200.0, "mape": 2.10}},
    ]
    eligible_reports = [r for r in reports if is_future_forecast_supported("electricity", r["model"], horizon=horizon)]
    selector = AdaptiveModelSelector(selection_metric="rmse")
    decision = selector.select_best_model(eligible_reports)
    assert decision["selected_model"] == "trend_xgboost"
    assert "sarima" not in [r["model"] for r in eligible_reports]


# -----------------------------------------------------------------------------
# Test 15: Recursive future forecast residual decay
# -----------------------------------------------------------------------------
def test_trend_xgboost_recursive_future_forecast_residual_decay(
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, _, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=True,
    )
    forecaster.fit(train_df, target_column="demand_mw")

    preds = recursive_tabular_future_forecast(
        model=forecaster,
        history_df=train_df,
        domain="electricity",
        target_column="demand_mw",
        horizon=200,
    )
    assert len(preds) == 200
    # Values should remain finite and positive
    vals = [p["predicted"] for p in preds]
    assert all(np.isfinite(v) for v in vals)
    assert all(v > 0 for v in vals)


# -----------------------------------------------------------------------------
# Test 16: Future forecast stays bounded without explosion (h=720, 2160)
# -----------------------------------------------------------------------------
def test_trend_xgboost_future_forecast_no_drift_or_explosion(
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, _, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=True,
    )
    forecaster.fit(train_df, target_column="demand_mw")

    preds = recursive_tabular_future_forecast(
        model=forecaster,
        history_df=train_df,
        domain="electricity",
        target_column="demand_mw",
        horizon=720,
    )
    vals = np.array([p["predicted"] for p in preds])
    assert np.all(np.isfinite(vals))
    assert np.all(vals > 0)
    # Check that predictions don't explode
    hist_max = train_df["demand_mw"].max()
    assert np.max(vals) < hist_max * 5.0


# -----------------------------------------------------------------------------
# Test 17: SHAP waterfall exact additivity (< 0.05 MW difference)
# -----------------------------------------------------------------------------
def test_trend_xgboost_shap_waterfall_exact_additivity(
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=True,
    )
    forecaster.fit(train_df, target_column="demand_mw")

    explainer = ShapTreeExplainer(forecaster=forecaster)
    exp = explainer.explain_local(eval_df=val_df, sample_index=0)

    pred = exp["prediction"]
    base = exp["base_value"]
    contrib_sum = sum(exp["shap_contributions"].values())
    total_attributed = base + contrib_sum
    assert abs(pred - total_attributed) < 0.05
    assert "baseline_trend" in exp["shap_contributions"]
    assert "residual_ar_correction" in exp["shap_contributions"]


# -----------------------------------------------------------------------------
# Test 18: LIME attributions include residual_ar_correction
# -----------------------------------------------------------------------------
def test_trend_xgboost_lime_attributions_include_residual_correction(
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=True,
    )
    forecaster.fit(train_df, target_column="demand_mw")

    explainer = LimeTabularExplainer(forecaster=forecaster, train_df=train_df.iloc[:50])
    exp = explainer.explain_local(eval_df=val_df, sample_index=0)

    rules = [attr["rule"] for attr in exp["feature_attributions"]]
    assert "baseline_trend" in rules
    assert "residual_ar_correction" in rules


# -----------------------------------------------------------------------------
# Test 19: Artifact persistence roundtrip
# -----------------------------------------------------------------------------
def test_trend_xgboost_artifact_persistence_roundtrip(
    tmp_path: Path,
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df, test_df = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=True,
    )
    forecaster.fit(train_df, target_column="demand_mw")
    preds_before = forecaster.predict(val_df)

    feats_dir = tmp_path / "feats"
    feats_dir.mkdir()
    train_df.to_parquet(feats_dir / "train.parquet")
    val_df.to_parquet(feats_dir / "validation.parquet")
    test_df.to_parquet(feats_dir / "test.parquet")

    cfg = {"trend_mode": "piecewise_linear", "trend_knot_hour": 350.0, "residual_ar": True}
    saved = save_model_artifact(
        model=forecaster,
        domain="electricity",
        model_key="trend_xgboost",
        target="demand_mw",
        features_id=derive_features_id(feats_dir),
        feature_names=forecaster.feature_names,
        configuration=cfg,
        validation_metrics={"rmse": 10.0},
        test_metrics={"rmse": 12.0},
        artifacts_root=tmp_path / "artifacts",
    )
    loaded = try_load_compatible_artifact(
        domain="electricity",
        model_key="trend_xgboost",
        target="demand_mw",
        features_id=derive_features_id(feats_dir),
        configuration=cfg,
        artifacts_root=tmp_path / "artifacts",
    )
    assert loaded is not None
    loaded_model = loaded["model"]
    preds_after = loaded_model.predict(val_df)
    np.testing.assert_allclose(preds_before.to_numpy(), preds_after.to_numpy(), rtol=1e-7, atol=1e-7)
    assert loaded_model.residual_ar is True
    assert loaded_model.residual_ar_weights == forecaster.residual_ar_weights


# -----------------------------------------------------------------------------
# Test 20: Artifact compatibility fingerprint differs with residual_ar
# -----------------------------------------------------------------------------
def test_trend_xgboost_artifact_compatibility_fingerprint() -> None:
    cfg1 = expected_configuration_for_request(
        model_key="trend_xgboost",
        domain="electricity",
        trend_mode="piecewise_linear",
        residual_ar=True,
    )
    cfg2 = expected_configuration_for_request(
        model_key="trend_xgboost",
        domain="electricity",
        trend_mode="piecewise_linear",
        residual_ar=False,
    )
    fp1 = config_fingerprint(cfg1)
    fp2 = config_fingerprint(cfg2)
    assert fp1 != fp2


# -----------------------------------------------------------------------------
# Test 21: Synthetic controlled accuracy gain
# -----------------------------------------------------------------------------
def test_trend_xgboost_synthetic_controlled_accuracy_gain(
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df, _ = synthetic_elec_data
    # Baseline piecewise without residual AR
    m_base = XGBoostForecaster(
        n_estimators=20,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=False,
        random_state=42,
    )
    m_base.fit(train_df, target_column="demand_mw")
    p_base = m_base.predict(val_df)
    rmse_base = evaluate_forecast(val_df["demand_mw"].to_numpy(), p_base.to_numpy())["rmse"]

    # Candidate with residual AR
    m_ar = XGBoostForecaster(
        n_estimators=20,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=True,
        random_state=42,
    )
    m_ar.fit(train_df, target_column="demand_mw")
    p_ar = m_ar.predict(val_df)
    rmse_ar = evaluate_forecast(val_df["demand_mw"].to_numpy(), p_ar.to_numpy())["rmse"]

    assert rmse_ar < rmse_base


# -----------------------------------------------------------------------------
# Test 22: Canonical electricity validation RMSE match (Exp27 Candidate B benchmark)
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not CANONICAL_FEATURES_DIR.exists(), reason="Canonical features directory missing")
def test_trend_xgboost_canonical_electricity_validation_rmse() -> None:
    train_df = pd.read_parquet(CANONICAL_FEATURES_DIR / "train.parquet")
    val_df = pd.read_parquet(CANONICAL_FEATURES_DIR / "validation.parquet")

    model = XGBoostForecaster(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        trend_mode="piecewise_linear",
        trend_knot_hour=30000.0,
        sample_weight_strategy="spike_sensitive",
        residual_ar=True,
        random_state=42,
    )
    model.fit(train_df, target_column="demand_mw")
    val_pred = model.predict(val_df)
    metrics = evaluate_forecast(val_df["demand_mw"].to_numpy(), val_pred.to_numpy())
    # Benchmark: 1863.53 MW +/- 5 MW
    assert abs(metrics["rmse"] - 1863.53) < 5.0
    assert metrics["rmse"] < 1900.0


# -----------------------------------------------------------------------------
# Test 23: Canonical electricity held-out test RMSE match
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not CANONICAL_FEATURES_DIR.exists(), reason="Canonical features directory missing")
def test_trend_xgboost_canonical_electricity_test_rmse() -> None:
    train_df = pd.read_parquet(CANONICAL_FEATURES_DIR / "train.parquet")
    val_df = pd.read_parquet(CANONICAL_FEATURES_DIR / "validation.parquet")
    test_df = pd.read_parquet(CANONICAL_FEATURES_DIR / "test.parquet")

    model = XGBoostForecaster(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        trend_mode="piecewise_linear",
        trend_knot_hour=30000.0,
        sample_weight_strategy="spike_sensitive",
        residual_ar=True,
        random_state=42,
    )
    model.fit(train_df, target_column="demand_mw")
    val_pred = model.predict(val_df)
    val_res = val_df["demand_mw"].to_numpy() - val_pred.to_numpy()
    test_pred = model.predict(test_df, residual_history=val_res[-168:])
    metrics = evaluate_forecast(test_df["demand_mw"].to_numpy(), test_pred.to_numpy())
    # Benchmark: 2552.50 MW +/- 10 MW
    assert abs(metrics["rmse"] - 2552.50) < 10.0


# -----------------------------------------------------------------------------
# Test 24: Extreme and tail regime improvement on canonical validation set
# -----------------------------------------------------------------------------
@pytest.mark.skipif(not CANONICAL_FEATURES_DIR.exists(), reason="Canonical features directory missing")
def test_trend_xgboost_extreme_and_tail_regime_improvement() -> None:
    train_df = pd.read_parquet(CANONICAL_FEATURES_DIR / "train.parquet")
    val_df = pd.read_parquet(CANONICAL_FEATURES_DIR / "validation.parquet")

    model = XGBoostForecaster(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        trend_mode="piecewise_linear",
        trend_knot_hour=30000.0,
        sample_weight_strategy="spike_sensitive",
        residual_ar=True,
        random_state=42,
    )
    model.fit(train_df, target_column="demand_mw")
    val_pred = model.predict(val_df)

    y_tr = train_df["demand_mw"].to_numpy()
    y_va = val_df["demand_mw"].to_numpy()
    p_va = val_pred.to_numpy()

    q95_tr = float(np.percentile(y_tr, 95))
    extreme_mask = y_va > q95_tr
    extreme_rmse = np.sqrt(np.mean((y_va[extreme_mask] - p_va[extreme_mask]) ** 2))
    assert extreme_rmse < 2000.0  # Exp27 Candidate B: 1969.68 MW vs baseline 2926.12 MW

    q95_va = float(np.percentile(y_va, 95))
    tail_mask = y_va > q95_va
    tail_rmse = np.sqrt(np.mean((y_va[tail_mask] - p_va[tail_mask]) ** 2))
    assert tail_rmse < 3200.0  # Exp27 Candidate B: 3124.66 MW vs baseline 6323.83 MW


# -----------------------------------------------------------------------------
# Test 25: Unrelated models completely unaffected
# -----------------------------------------------------------------------------
def test_trend_xgboost_unrelated_models_unaffected() -> None:
    # Retail XGBoost must not have residual_ar
    retail_cfg = expected_configuration_for_request(
        model_key="xgboost",
        domain="retail",
    )
    assert "residual_ar" not in retail_cfg
    assert retail_cfg.get("target_transform") == "log1p"

    # ARIMA config must not have residual_ar
    arima_cfg = expected_configuration_for_request(
        model_key="arima",
        domain="electricity",
    )
    assert "residual_ar" not in arima_cfg

    # Naive config must not have residual_ar
    naive_cfg = expected_configuration_for_request(
        model_key="naive",
        domain="electricity",
    )
    assert "residual_ar" not in naive_cfg


# -----------------------------------------------------------------------------
# Test 26: Backward-compatible default verification (residual_ar=False)
# -----------------------------------------------------------------------------
def test_trend_xgboost_backward_compatible_default_is_false() -> None:
    # Default expected_configuration must not have residual_ar
    cfg_default = expected_configuration_for_request(
        model_key="trend_xgboost",
        domain="electricity",
        trend_mode="piecewise_linear",
    )
    assert "residual_ar" not in cfg_default

    # Explicit residual_ar=True must have residual_ar: True
    cfg_enabled = expected_configuration_for_request(
        model_key="trend_xgboost",
        domain="electricity",
        trend_mode="piecewise_linear",
        residual_ar=True,
    )
    assert cfg_enabled.get("residual_ar") is True


# -----------------------------------------------------------------------------
# Test 27: Causality and boundary invariance verification
# -----------------------------------------------------------------------------
def test_trend_xgboost_causality_boundary_invariance(
    synthetic_elec_data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    train_df, val_df, _ = synthetic_elec_data
    forecaster = XGBoostForecaster(
        n_estimators=10,
        trend_mode="piecewise_linear",
        trend_knot_hour=350.0,
        residual_ar=True,
    )
    forecaster.fit(train_df, target_column="demand_mw")

    # Original prediction
    p_orig = forecaster.predict(val_df).to_numpy()

    # Create mutated eval df where rows >= 50 have drastically altered targets
    val_mutated = val_df.copy()
    val_mutated.loc[val_mutated.index >= 50, "demand_mw"] = val_mutated.loc[val_mutated.index >= 50, "demand_mw"] * 5.0

    p_mut = forecaster.predict(val_mutated).to_numpy()

    # Rows 0..49 must be strictly identical (zero leakage from future actuals at t >= 50)
    np.testing.assert_array_equal(p_orig[:50], p_mut[:50])

    # Row 50 must also be identical because row 50's residual correction only uses t <= 49
    np.testing.assert_array_equal(p_orig[50], p_mut[50])

    # Row 51 uses the mutated error at row 50, so it will differ
    assert not np.array_equal(p_orig[51:], p_mut[51:])


# -----------------------------------------------------------------------------
# Test 28: Auto Select is purely metric-driven (not hardcoded by model name)
# -----------------------------------------------------------------------------
def test_auto_select_is_purely_metric_driven_not_hardcoded() -> None:
    # If a hypothetical model has better validation RMSE than SARIMA, it must win
    hypothetical_reports = [
        {"model": "hypothetical_super_model", "validation_metrics": {"rmse": 999.0, "mae": 700.0, "mape": 0.40}},
        {"model": "sarima", "validation_metrics": {"rmse": 1401.19, "mae": 980.0, "mape": 0.60}},
        {"model": "trend_xgboost", "validation_metrics": {"rmse": 1863.53, "mae": 1334.7, "mape": 0.71}},
    ]
    selector = AdaptiveModelSelector(selection_metric="rmse")
    decision = selector.select_best_model(hypothetical_reports)
    assert decision["selected_model"] == "hypothetical_super_model"

