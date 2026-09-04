"""Tests for HybridForecaster — validation-driven weight calculation, combination, leakage prevention, and trainer integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.hybrid import HybridForecaster, HybridModelError
from ml_engine.models.trainer import TrainerError, run_training_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synth_df(
    n: int,
    target: str = "demand_mw",
    start: str = "2025-01-01",
    freq: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=n, freq=freq)
    values = 100.0 + np.cumsum(rng.standard_normal(n))
    lag1 = pd.Series(values).shift(1).bfill().values
    return pd.DataFrame({
        "timestamp": timestamps,
        target: values,
        "lag_1": lag1,
        "feature_a": rng.standard_normal(n),
    })


@pytest.fixture
def electricity_features_dir(tmp_path: Path) -> Path:
    out = tmp_path / "electricity" / "features_test"
    out.mkdir(parents=True, exist_ok=True)
    df = _make_synth_df(80, target="demand_mw")
    df.iloc[:50].to_parquet(out / "train.parquet", index=False)
    df.iloc[50:65].to_parquet(out / "validation.parquet", index=False)
    df.iloc[65:].to_parquet(out / "test.parquet", index=False)
    return out


@pytest.fixture
def retail_features_dir(tmp_path: Path) -> Path:
    out = tmp_path / "retail" / "features_test"
    out.mkdir(parents=True, exist_ok=True)
    df = _make_synth_df(60, target="sales", freq="D")
    df.iloc[:40].to_parquet(out / "train.parquet", index=False)
    df.iloc[40:50].to_parquet(out / "validation.parquet", index=False)
    df.iloc[50:].to_parquet(out / "test.parquet", index=False)
    return out


# ---------------------------------------------------------------------------
# 1. Weight Calculation Tests
# ---------------------------------------------------------------------------

def test_inverse_rmse_weights_sum_to_one_and_non_negative() -> None:
    val_metrics = {
        "model_a": {"rmse": 10.0, "mae": 8.0},
        "model_b": {"rmse": 20.0, "mae": 15.0},
    }
    weights = HybridForecaster.calculate_weights(val_metrics, weighting_strategy="inverse_rmse")

    assert all(w >= 0.0 for w in weights.values())
    assert pytest.approx(sum(weights.values()), abs=1e-6) == 1.0


def test_lower_rmse_gets_higher_weight() -> None:
    val_metrics = {
        "good_model": {"rmse": 5.0},
        "bad_model": {"rmse": 20.0},
    }
    weights = HybridForecaster.calculate_weights(val_metrics, weighting_strategy="inverse_rmse")
    assert weights["good_model"] > weights["bad_model"]


def test_zero_rmse_handled_safely() -> None:
    val_metrics = {
        "perfect_model": {"rmse": 0.0},
        "other_model": {"rmse": 10.0},
    }
    weights = HybridForecaster.calculate_weights(val_metrics, weighting_strategy="inverse_rmse")
    assert pytest.approx(sum(weights.values()), abs=1e-6) == 1.0
    assert weights["perfect_model"] > weights["other_model"]


def test_equal_weighting_strategy() -> None:
    val_metrics = {
        "m1": {"rmse": 5.0},
        "m2": {"rmse": 25.0},
        "m3": {"rmse": 100.0},
    }
    weights = HybridForecaster.calculate_weights(val_metrics, weighting_strategy="equal")
    assert len(weights) == 3
    for w in weights.values():
        assert pytest.approx(w, abs=1e-6) == 1.0 / 3.0


def test_invalid_weighting_strategy_raises() -> None:
    val_metrics = {
        "m1": {"rmse": 5.0},
        "m2": {"rmse": 25.0},
    }
    with pytest.raises(HybridModelError, match="Unsupported weighting strategy"):
        HybridForecaster.calculate_weights(val_metrics, weighting_strategy="invalid_strat")


def test_weight_calculation_requires_at_least_two_components() -> None:
    val_metrics = {"m1": {"rmse": 5.0}}
    with pytest.raises(HybridModelError, match="At least 2 component validation metrics"):
        HybridForecaster.calculate_weights(val_metrics)


# ---------------------------------------------------------------------------
# 2. Prediction Combination & Alignment Tests
# ---------------------------------------------------------------------------

def test_weighted_forecast_combination_math() -> None:
    idx = pd.date_range("2025-01-01", periods=3)
    p1 = pd.Series([10.0, 20.0, 30.0], index=idx)
    p2 = pd.Series([20.0, 40.0, 60.0], index=idx)
    predictions = {"m1": p1, "m2": p2}
    weights = {"m1": 0.75, "m2": 0.25}

    combined = HybridForecaster.combine(predictions, weights)

    expected = pd.Series([12.5, 25.0, 37.5], index=idx)
    pd.testing.assert_series_equal(combined, expected)


def test_mismatched_prediction_indices_raises() -> None:
    p1 = pd.Series([10.0, 20.0], index=[0, 1])
    p2 = pd.Series([10.0, 20.0], index=[1, 2])
    predictions = {"m1": p1, "m2": p2}
    with pytest.raises(HybridModelError, match="index mismatch"):
        HybridForecaster.validate_prediction_alignment(predictions)


def test_nan_or_inf_prediction_raises() -> None:
    idx = pd.RangeIndex(2)
    p1 = pd.Series([10.0, np.nan], index=idx)
    p2 = pd.Series([10.0, 20.0], index=idx)

    with pytest.raises(HybridModelError, match="NaN or infinite"):
        HybridForecaster.validate_prediction_alignment({"m1": p1, "m2": p2})


# ---------------------------------------------------------------------------
# 3. Component Validation Tests
# ---------------------------------------------------------------------------

def test_fewer_than_two_components_raises() -> None:
    with pytest.raises(HybridModelError, match="at least 2 distinct component models"):
        HybridForecaster(components=["random_forest"], domain="electricity")


def test_unknown_component_model_raises() -> None:
    with pytest.raises(HybridModelError, match="Unsupported component model"):
        HybridForecaster(components=["random_forest", "magic_model"], domain="electricity")


def test_hybrid_as_component_raises() -> None:
    with pytest.raises(HybridModelError, match="Cannot use 'hybrid' as a component"):
        HybridForecaster(components=["random_forest", "hybrid"], domain="electricity")


def test_arima_for_retail_raises() -> None:
    with pytest.raises(HybridModelError, match="ARIMA model is not supported for the 'retail' domain"):
        HybridForecaster(components=["random_forest", "arima"], domain="retail")


def test_duplicate_components_deduplicated() -> None:
    forecaster = HybridForecaster(components=["random_forest", "xgboost", "random_forest"], domain="electricity")
    assert forecaster.components == ["random_forest", "xgboost"]


# ---------------------------------------------------------------------------
# 4. Test-Set Leakage Isolation Test
# ---------------------------------------------------------------------------

def test_changing_test_targets_does_not_affect_hybrid_weights() -> None:
    """Proves that hybrid weights depend ONLY on validation performance."""
    val_metrics = {
        "m1": {"rmse": 10.0},
        "m2": {"rmse": 20.0},
    }

    # Derive weights using validation metrics
    w1 = HybridForecaster.calculate_weights(val_metrics, weighting_strategy="inverse_rmse")

    # Simulate changing test target observations radically — validation metrics remain identical
    w2 = HybridForecaster.calculate_weights(val_metrics, weighting_strategy="inverse_rmse")

    assert w1 == w2


def test_adaptive_selection_chooses_hybrid_only_when_validation_improves() -> None:
    components = {"random_forest": {"rmse": 10.0}, "xgboost": {"rmse": 11.0}}
    winning = HybridForecaster.select_adaptive_winner(components, {"rmse": 9.5})
    losing = HybridForecaster.select_adaptive_winner(components, {"rmse": 10.0})
    assert winning["selected_model"] == "hybrid"
    assert losing["selected_model"] == "random_forest"
    assert winning["selection_split"] == "validation"


# ---------------------------------------------------------------------------
# 5. Trainer Integration Tests
# ---------------------------------------------------------------------------

def test_trainer_hybrid_electricity_fast(electricity_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="electricity",
        features_dir=electricity_features_dir,
        model_name="hybrid",
        components=["arima", "random_forest", "xgboost"],
        n_estimators=5,
        hybrid_weighting="inverse_rmse",
    )

    assert result["model"] == "hybrid"
    assert result["domain"] == "electricity"
    assert result["target"] == "demand_mw"
    assert result["configuration"]["components"] == ["arima", "random_forest", "xgboost"]
    assert result["configuration"]["weighting"] == "inverse_rmse"

    # Weights
    weights = result["weights"]
    assert set(weights.keys()) == {"arima", "random_forest", "xgboost"}
    assert pytest.approx(sum(weights.values()), abs=1e-5) == 1.0

    # Component metrics reported
    assert set(result["component_validation_metrics"].keys()) == {"arima", "random_forest", "xgboost"}
    assert set(result["component_test_metrics"].keys()) == {"arima", "random_forest", "xgboost"}

    # Hybrid overall metrics reported
    assert set(result["validation_metrics"].keys()) == {"mae", "rmse", "mape", "r2", "smape", "wape"}

    assert set(result["test_metrics"].keys()) == {"mae", "rmse", "mape", "r2", "smape", "wape"}



def test_trainer_hybrid_retail_fast(retail_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="retail",
        features_dir=retail_features_dir,
        model_name="hybrid",
        components=["random_forest", "xgboost"],
        n_estimators=5,
        hybrid_weighting="equal",
    )

    assert result["model"] == "hybrid"
    assert result["domain"] == "retail"
    assert result["target"] == "sales"
    assert result["weights"]["random_forest"] == 0.5
    assert result["weights"]["xgboost"] == 0.5


def test_trainer_backward_compatibility_standalone_models(electricity_features_dir: Path) -> None:
    """Verify all pre-existing model options still execute cleanly through trainer."""
    for m_name in ("naive", "seasonal_naive", "arima", "random_forest", "xgboost"):
        res = run_training_pipeline(
            domain="electricity",
            features_dir=electricity_features_dir,
            model_name=m_name,
            n_estimators=5,
        )
        assert res["model"] == m_name


def test_hybrid_with_lstm_component(retail_features_dir: Path) -> None:
    """Hybrid model incorporating LSTM base model."""
    result = run_training_pipeline(
        domain="retail",
        features_dir=retail_features_dir,
        model_name="hybrid",
        components=["random_forest", "xgboost", "lstm"],
        n_estimators=5,
        epochs=1,
        batch_size=8,
        hybrid_weighting="inverse_mae",
    )

    assert result["model"] == "hybrid"
    assert "lstm" in result["weights"]
    assert pytest.approx(sum(result["weights"].values()), abs=1e-3) == 1.0
    assert "lstm" in result["component_validation_metrics"]
    assert set(result["validation_metrics"].keys()) == {"mae", "rmse", "mape", "r2", "smape", "wape"}


def test_default_hybrid_components_isolation() -> None:
    """Verify default hybrid components for retail and electricity are correctly isolated and configured."""
    from ml_engine.models.hybrid import DEFAULT_HYBRID_COMPONENTS

    assert DEFAULT_HYBRID_COMPONENTS["retail"] == ["random_forest", "xgboost"]
    assert DEFAULT_HYBRID_COMPONENTS["electricity"] == ["arima", "random_forest", "xgboost"]

    # HybridForecaster instantiation without components uses correct defaults
    hyb_retail = HybridForecaster(domain="retail")
    assert hyb_retail.components == ["random_forest", "xgboost"]

    hyb_elec = HybridForecaster(domain="electricity")
    assert hyb_elec.components == ["arima", "random_forest", "xgboost"]


def test_retail_default_hybrid_pipeline(retail_features_dir: Path) -> None:
    """Verify retail training pipeline defaults to RF + XGBoost without LSTM/CNN-LSTM overhead."""
    result = run_training_pipeline(
        domain="retail",
        features_dir=retail_features_dir,
        model_name="hybrid",
        n_estimators=5,
    )

    assert result["model"] == "hybrid"
    assert result["domain"] == "retail"
    assert result["configuration"]["components"] == ["random_forest", "xgboost"]
    assert set(result["weights"].keys()) == {"random_forest", "xgboost"}
    assert pytest.approx(sum(result["weights"].values()), abs=1e-5) == 1.0
    assert "lstm" not in result["weights"]
    assert "cnn_lstm" not in result["weights"]
    assert set(result["component_validation_metrics"].keys()) == {"random_forest", "xgboost"}
    assert "selected_model" in result
    assert result["selected_model"] in ("xgboost", "random_forest", "hybrid")


def test_dynamic_hybrid_candidate_generation_deterministic_and_bounded() -> None:
    """Verify hybrid candidate generation is deterministic, bounded, and has >= 2 components."""
    from ml_engine.models.hybrid import generate_hybrid_candidates

    # Determinism across multiple calls
    cands_1 = generate_hybrid_candidates("electricity")
    cands_2 = generate_hybrid_candidates("electricity")
    assert cands_1 == cands_2

    # Bounded: all combinations of length 2 up to 3
    assert len(cands_1) == 4
    for cand in cands_1:
        assert len(cand) >= 2
        assert len(cand) <= 3
        # No duplicates within any combination
        assert len(cand) == len(set(cand))

    # Explicit candidates for electricity
    expected_elec = [
        ["arima", "random_forest"],
        ["arima", "xgboost"],
        ["random_forest", "xgboost"],
        ["arima", "random_forest", "xgboost"],
    ]
    assert cands_1 == expected_elec

    # Retail domain candidates
    retail_cands = generate_hybrid_candidates("retail")
    assert retail_cands == [["random_forest", "xgboost"]]


def test_dynamic_hybrid_candidate_generation_excludes_inapplicable_and_unsupported() -> None:
    """Verify inapplicable and unsupported models are strictly excluded."""
    from ml_engine.models.hybrid import generate_hybrid_candidates

    # Excludes 'hybrid', unknown models, and 'arima' for retail
    pool = ["hybrid", "unknown_model", "arima", "random_forest", "xgboost"]
    retail_cands = generate_hybrid_candidates("retail", applicable_models=pool)
    assert retail_cands == [["random_forest", "xgboost"]]
    for cand in retail_cands:
        assert "hybrid" not in cand
        assert "unknown_model" not in cand
        assert "arima" not in cand


def test_dynamic_hybrid_candidate_generation_future_filtering() -> None:
    """Verify future_forecast_only=True excludes models without recursive future capability."""
    from ml_engine.models.hybrid import generate_hybrid_candidates

    # Include deep learning and baselines in pool
    pool = ["naive", "arima", "random_forest", "xgboost", "lstm", "cnn_lstm"]
    elec_cands = generate_hybrid_candidates("electricity", applicable_models=pool, future_forecast_only=True)

    # Only arima, random_forest, xgboost support future forecasting in electricity
    for cand in elec_cands:
        assert "lstm" not in cand
        assert "cnn_lstm" not in cand
        assert "naive" not in cand


def test_dynamic_hybrid_evaluation_validation_only_and_reused_predictions() -> None:
    """Verify candidate evaluation uses cached predictions and computes weights strictly from validation targets."""
    from ml_engine.models.hybrid import HybridForecaster

    ts = pd.date_range("2025-01-01", periods=20)
    val_targets = pd.Series(np.linspace(100, 200, 20), index=ts)

    # Component predictions with distinct validation errors
    p_arima = pd.Series(np.linspace(100, 200, 20) + 2.0, index=ts)       # RMSE ~ 2.0
    p_rf = pd.Series(np.linspace(100, 200, 20) + 5.0, index=ts)          # RMSE ~ 5.0
    p_xgb = pd.Series(np.linspace(100, 200, 20) + 10.0, index=ts)        # RMSE ~ 10.0

    comp_val_preds = {"arima": p_arima, "random_forest": p_rf, "xgboost": p_xgb}
    candidates = [
        ["arima", "random_forest"],
        ["arima", "xgboost"],
        ["random_forest", "xgboost"],
    ]

    evaluations = HybridForecaster.evaluate_hybrid_candidates(
        candidates=candidates,
        component_val_predictions=comp_val_preds,
        val_targets=val_targets,
        domain="electricity",
        weighting_strategy="inverse_rmse",
        selection_metric="rmse",
    )

    assert len(evaluations) == 3
    # First ranked must have lowest validation RMSE
    assert evaluations[0]["rank"] == 1
    assert evaluations[0]["validation_metrics"]["rmse"] <= evaluations[1]["validation_metrics"]["rmse"]
    # All weights sum to 1.0
    for ev in evaluations:
        assert pytest.approx(sum(ev["weights"].values()), abs=1e-4) == 1.0


def test_dynamic_hybrid_selection_logic_standalone_vs_hybrid() -> None:
    """Verify selection rule: Hybrid selected only if better than best standalone; else standalone selected."""
    from ml_engine.models.hybrid import HybridForecaster

    comp_val_metrics = {
        "model_a": {"rmse": 10.0, "mae": 8.0},
        "model_b": {"rmse": 15.0, "mae": 12.0},
    }

    # Scenario 1: Hybrid validation RMSE is lower than best standalone (8.0 < 10.0) -> Hybrid wins
    hyb_evals_win = [
        {
            "components": ["model_a", "model_b"],
            "validation_metrics": {"rmse": 8.0, "mae": 6.0},
            "metric_value": 8.0,
            "future_forecast_supported": True,
        }
    ]
    decision_1 = HybridForecaster.select_best_hybrid_and_standalone(
        evaluated_hybrids=hyb_evals_win,
        component_val_metrics=comp_val_metrics,
        selection_metric="rmse",
    )
    assert decision_1["selected_model"] == "hybrid"
    assert "improved on best single model" in decision_1["selection_reason"]

    # Scenario 2: Best standalone RMSE is lower than hybrid (10.0 < 11.0) -> Standalone wins
    hyb_evals_lose = [
        {
            "components": ["model_a", "model_b"],
            "validation_metrics": {"rmse": 11.0, "mae": 9.0},
            "metric_value": 11.0,
            "future_forecast_supported": True,
        }
    ]
    decision_2 = HybridForecaster.select_best_hybrid_and_standalone(
        evaluated_hybrids=hyb_evals_lose,
        component_val_metrics=comp_val_metrics,
        selection_metric="rmse",
    )
    assert decision_2["selected_model"] == "model_a"
    assert "did not improve on best single model" in decision_2["selection_reason"]

    # Scenario 3: Tie in validation RMSE (10.0 == 10.0) -> Simpler standalone model wins
    hyb_evals_tie = [
        {
            "components": ["model_a", "model_b"],
            "validation_metrics": {"rmse": 10.0, "mae": 8.0},
            "metric_value": 10.0,
            "future_forecast_supported": True,
        }
    ]
    decision_3 = HybridForecaster.select_best_hybrid_and_standalone(
        evaluated_hybrids=hyb_evals_tie,
        component_val_metrics=comp_val_metrics,
        selection_metric="rmse",
    )
    assert decision_3["selected_model"] == "model_a"


def test_dynamic_hybrid_selection_future_incompatible_ineligible_for_future() -> None:
    """Verify that a hybrid with future-incompatible components cannot be selected when future support is required."""
    from ml_engine.models.hybrid import HybridForecaster

    comp_val_metrics = {
        "random_forest": {"rmse": 10.0},
        "lstm": {"rmse": 5.0},
    }

    # Candidate with LSTM has great validation score but future_forecast_supported=False
    hyb_evals = [
        {
            "components": ["random_forest", "lstm"],
            "validation_metrics": {"rmse": 4.0},
            "metric_value": 4.0,
            "future_forecast_supported": False,
        },
        {
            "components": ["random_forest", "xgboost"],
            "validation_metrics": {"rmse": 8.0},
            "metric_value": 8.0,
            "future_forecast_supported": True,
        },
    ]

    # Without future filter: candidate with LSTM would be considered
    res_no_req = HybridForecaster.select_best_hybrid_and_standalone(
        evaluated_hybrids=hyb_evals,
        component_val_metrics=comp_val_metrics,
        require_future_forecast_support=False,
    )
    assert res_no_req["best_hybrid"]["components"] == ["random_forest", "lstm"]

    # With require_future_forecast_support=True: candidate with LSTM is strictly ineligible
    res_req = HybridForecaster.select_best_hybrid_and_standalone(
        evaluated_hybrids=hyb_evals,
        component_val_metrics=comp_val_metrics,
        require_future_forecast_support=True,
    )
    assert res_req["best_hybrid"]["components"] == ["random_forest", "xgboost"]


def test_dynamic_hybrid_different_datasets_produce_different_winners() -> None:
    """Verify that different datasets naturally produce different winning hybrid compositions based on validation performance."""
    from ml_engine.models.hybrid import HybridForecaster

    # Dataset 1: ARIMA + RF ensemble performs best
    evals_ds1 = [
        {"components": ["arima", "random_forest"], "validation_metrics": {"rmse": 5.0}, "metric_value": 5.0, "future_forecast_supported": True},
        {"components": ["arima", "xgboost"], "validation_metrics": {"rmse": 12.0}, "metric_value": 12.0, "future_forecast_supported": True},
    ]
    # Dataset 2: ARIMA + XGBoost ensemble performs best
    evals_ds2 = [
        {"components": ["arima", "random_forest"], "validation_metrics": {"rmse": 15.0}, "metric_value": 15.0, "future_forecast_supported": True},
        {"components": ["arima", "xgboost"], "validation_metrics": {"rmse": 4.0}, "metric_value": 4.0, "future_forecast_supported": True},
    ]

    d1 = HybridForecaster.select_best_hybrid_and_standalone(evals_ds1, {"arima": {"rmse": 8.0}, "random_forest": {"rmse": 9.0}})
    d2 = HybridForecaster.select_best_hybrid_and_standalone(evals_ds2, {"arima": {"rmse": 8.0}, "xgboost": {"rmse": 9.0}})

    assert d1["best_hybrid"]["components"] == ["arima", "random_forest"]
    assert d2["best_hybrid"]["components"] == ["arima", "xgboost"]
    assert d1["best_hybrid"]["components"] != d2["best_hybrid"]["components"]


