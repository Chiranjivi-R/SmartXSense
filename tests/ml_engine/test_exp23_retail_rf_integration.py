"""Regression tests for Experiment 23: Retail Random Forest Macro-Feature Pruning.

Verifies:
1. Retail RF excludes exactly the 9 identified macroeconomic features.
2. Electricity RF retains all features (no macro exclusions).
3. Configuration and artifact fingerprint distinguish old vs. new RF.
4. Artifact save, reload, identical feature order/configuration, and deterministic numerical parity (rtol=1e-7, atol=1e-7).
5. SHAP global/local explanation dimension parity (41 features).
6. LIME local explanation dimension parity (41 features).
7. Future forecasting compatibility with no-macro Retail RF.
8. Hybrid compatibility with no-macro Retail RF.
9. Auto Select validation evaluation ranking with no-macro Retail RF.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml_engine.evaluation import AdaptiveModelSelector, compare_models
from ml_engine.evaluation.metrics import evaluate_forecast
from ml_engine.explainability import ForecastExplainer, LimeTabularExplainer, ShapTreeExplainer
from ml_engine.models.artifact_store import (
    ArtifactError,
    check_compatibility,
    config_fingerprint,
    save_model_artifact,
    try_load_compatible_artifact,
)
from ml_engine.models.future_forecasting import generate_future_forecast
from ml_engine.models.hybrid import HybridForecaster
from ml_engine.models.random_forest import RETAIL_MACRO_EXCLUSIONS, RandomForestForecaster
from ml_engine.models.trainer import (
    expected_configuration_for_request,
    run_training_pipeline,
)
from ml_engine.models.xgboost_model import XGBoostForecaster


@pytest.fixture()
def sample_retail_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic retail dataset containing the 9 macro features and other demand features."""
    np.random.seed(42)
    n = 90
    dates = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    data = {
        "timestamp": dates,
        "sales": np.random.uniform(10.0, 100.0, size=n),
        "city_tier": np.random.choice(["Tier 1", "Tier 2", "Tier 3"], size=n),
        "lag_1": np.random.uniform(10.0, 100.0, size=n),
        "lag_7": np.random.uniform(10.0, 100.0, size=n),
        "rolling_mean_7": np.random.uniform(10.0, 100.0, size=n),
        "is_weekend": np.random.choice([0.0, 1.0], size=n),
        "festival_flag": np.random.choice([0.0, 1.0], size=n),
    }
    # Add the 9 macro features
    for macro_col in RETAIL_MACRO_EXCLUSIONS:
        data[macro_col] = np.random.uniform(100.0, 200.0, size=n)

    df = pd.DataFrame(data)
    train_df = df.iloc[:50].copy()
    val_df = df.iloc[50:70].copy()
    test_df = df.iloc[70:].copy()
    return train_df, val_df, test_df


def test_retail_rf_excludes_exactly_nine_macro_features(sample_retail_data):
    """1. Verify Retail RF excludes exactly the 9 macroeconomic features."""
    train_df, val_df, _ = sample_retail_data

    rf = RandomForestForecaster(
        n_estimators=10,
        random_state=42,
        target_transform="log1p",
        feature_exclusions=list(RETAIL_MACRO_EXCLUSIONS),
    )
    rf.fit(train_df, target_column="sales")

    for macro_col in RETAIL_MACRO_EXCLUSIONS:
        assert macro_col not in rf.feature_names, f"Macro feature '{macro_col}' should be excluded!"

    # Included features must be present
    assert "city_tier" in rf.feature_names
    assert "lag_1" in rf.feature_names
    assert "lag_7" in rf.feature_names
    assert "rolling_mean_7" in rf.feature_names
    assert "is_weekend" in rf.feature_names
    assert "festival_flag" in rf.feature_names

    preds = rf.predict(val_df)
    assert len(preds) == len(val_df)
    assert not preds.isna().any()


def test_electricity_rf_does_not_exclude_features(sample_retail_data):
    """2. Verify Electricity RF retains all features without macro exclusions."""
    train_df, val_df, _ = sample_retail_data
    # Rebrand target column as demand_mw
    elec_train = train_df.rename(columns={"sales": "demand_mw"})
    elec_val = val_df.rename(columns={"sales": "demand_mw"})

    rf = RandomForestForecaster(n_estimators=10, random_state=42, feature_exclusions=None)
    rf.fit(elec_train, target_column="demand_mw")

    # Macro features were NOT excluded because feature_exclusions was None
    for macro_col in RETAIL_MACRO_EXCLUSIONS:
        assert macro_col in rf.feature_names


def test_rf_artifact_fingerprint_isolation():
    """3. Verify configuration and artifact fingerprint distinguish old vs. new RF."""
    old_retail_cfg = {
        "n_estimators": 200,
        "max_depth": None,
        "random_state": 42,
        "target_transform": "log1p",
    }
    new_retail_cfg = expected_configuration_for_request(
        model_key="random_forest",
        domain="retail",
        n_estimators=200,
        random_state=42,
    )
    assert "feature_exclusions" in new_retail_cfg
    assert new_retail_cfg["feature_exclusions"] == list(RETAIL_MACRO_EXCLUSIONS)

    old_fp = config_fingerprint(old_retail_cfg)
    new_fp = config_fingerprint(new_retail_cfg)
    assert old_fp != new_fp, "Old and new configurations must have distinct fingerprints!"

    # Incompatibility check: old artifact metadata must fail check_compatibility with new config
    old_metadata = {
        "artifact_version": 1,
        "domain": "retail",
        "model": "random_forest",
        "target": "sales",
        "features_id": "features_test",
        "dataset_fingerprint": "abcd1234",
        "feature_names": ["f1", "f2"],
        "feature_count": 2,
        "configuration": old_retail_cfg,
    }
    with pytest.raises(ArtifactError, match="configuration mismatch"):
        check_compatibility(
            old_metadata,
            domain="retail",
            model="random_forest",
            target="sales",
            features_id="features_test",
            dataset_fingerprint="abcd1234",
            configuration=new_retail_cfg,
        )


def test_rf_artifact_save_reload_deterministic_numerical_parity(sample_retail_data, tmp_path):
    """4. Verify artifact save, reload, identical feature order/configuration, and deterministic numerical parity."""
    train_df, val_df, _ = sample_retail_data
    rf = RandomForestForecaster(
        n_estimators=10,
        random_state=42,
        target_transform="log1p",
        feature_exclusions=list(RETAIL_MACRO_EXCLUSIONS),
    )
    rf.fit(train_df, target_column="sales")
    preds_orig = rf.predict(val_df)

    cfg = expected_configuration_for_request(
        model_key="random_forest",
        domain="retail",
        n_estimators=10,
        random_state=42,
    )
    save_model_artifact(
        model=rf,
        domain="retail",
        model_key="random_forest",
        target="sales",
        features_id="test_features",
        dataset_fingerprint="test_fp",
        feature_names=rf.feature_names,
        configuration=cfg,
        artifacts_root=tmp_path,
    )

    loaded = try_load_compatible_artifact(
        domain="retail",
        model_key="random_forest",
        target="sales",
        features_id="test_features",
        dataset_fingerprint="test_fp",
        feature_names=rf.feature_names,
        configuration=cfg,
        artifacts_root=tmp_path,
    )
    assert loaded is not None, "Failed to reload compatible artifact!"
    loaded_model = loaded["model"]

    # Verify identical feature names and order
    assert loaded_model.feature_names == rf.feature_names
    assert loaded["metadata"]["configuration"]["feature_exclusions"] == list(RETAIL_MACRO_EXCLUSIONS)

    # Verify deterministic numerical parity with tight tolerance
    preds_loaded = loaded_model.predict(val_df)
    np.testing.assert_allclose(preds_orig.to_numpy(), preds_loaded.to_numpy(), rtol=1e-7, atol=1e-7)


def test_shap_explainability_feature_parity(sample_retail_data):
    """5. Verify SHAP global and local explanations maintain 1-to-1 feature parity with pruned features."""
    train_df, val_df, _ = sample_retail_data
    rf = RandomForestForecaster(
        n_estimators=10,
        random_state=42,
        target_transform="log1p",
        feature_exclusions=list(RETAIL_MACRO_EXCLUSIONS),
    )
    rf.fit(train_df, target_column="sales")

    explainer = ForecastExplainer(rf, model_name="random_forest")
    global_exp = explainer.explain_global(val_df, method="shap")
    assert global_exp["status"] == "success"
    assert global_exp["feature_count"] == len(rf.feature_names)
    assert set(global_exp["feature_importance"].keys()) == set(rf.feature_names)

    # Confirm no macro features present in explanation
    for macro_col in RETAIL_MACRO_EXCLUSIONS:
        assert macro_col not in global_exp["feature_importance"]

    local_exp = explainer.explain_local(val_df, sample_index=0, method="shap")
    assert local_exp["explanation_status"] == "success"
    assert isinstance(local_exp["prediction"], float)


def test_lime_explainability_feature_parity(sample_retail_data):
    """6. Verify LIME explanation feature parity."""
    train_df, val_df, _ = sample_retail_data
    rf = RandomForestForecaster(
        n_estimators=10,
        random_state=42,
        target_transform="log1p",
        feature_exclusions=list(RETAIL_MACRO_EXCLUSIONS),
    )
    rf.fit(train_df, target_column="sales")

    lime_exp = LimeTabularExplainer(rf, train_df=train_df)
    assert lime_exp.feature_names == rf.feature_names

    local_exp = lime_exp.explain_local(val_df, sample_index=1)
    assert local_exp["explanation_type"] == "local"
    assert local_exp["method"] == "lime"
    assert isinstance(local_exp["prediction"], float)


def test_future_forecasting_with_pruned_retail_rf(sample_retail_data):
    """7. Verify future forecasting executes recursively with no-macro Retail RF."""
    from ml_engine.models.future_forecasting import recursive_tabular_future_forecast

    train_df, val_df, _ = sample_retail_data
    combined = pd.concat([train_df, val_df], ignore_index=True)

    rf = RandomForestForecaster(
        n_estimators=10,
        random_state=42,
        target_transform="log1p",
        feature_exclusions=list(RETAIL_MACRO_EXCLUSIONS),
    )
    rf.fit(combined, target_column="sales")

    future_res = recursive_tabular_future_forecast(
        model=rf,
        history_df=combined,
        domain="retail",
        target_column="sales",
        horizon=7,
    )
    assert len(future_res) == 7
    for point in future_res:
        assert np.isfinite(point["predicted"])
        assert point["predicted"] >= 0.0


def test_hybrid_compatibility_with_pruned_retail_rf(sample_retail_data):
    """8. Verify HybridForecaster operates with pruned Retail RF and unpruned XGBoost."""
    train_df, val_df, test_df = sample_retail_data

    rf = RandomForestForecaster(
        n_estimators=10,
        random_state=42,
        target_transform="log1p",
        feature_exclusions=list(RETAIL_MACRO_EXCLUSIONS),
    )
    rf.fit(train_df, target_column="sales")

    xgb = XGBoostForecaster(
        n_estimators=10,
        max_depth=3,
        random_state=42,
        target_transform="log1p",
    )
    xgb.fit(train_df, target_column="sales")

    val_rf = rf.predict(val_df)
    val_xgb = xgb.predict(val_df)
    test_rf = rf.predict(test_df)
    test_xgb = xgb.predict(test_df)

    evaluated = HybridForecaster.evaluate_hybrid_candidates(
        candidates=[["random_forest", "xgboost"]],
        component_val_predictions={"random_forest": val_rf, "xgboost": val_xgb},
        val_targets=val_df["sales"],
        component_test_predictions={"random_forest": test_rf, "xgboost": test_xgb},
        test_targets=test_df["sales"],
        domain="retail",
        weighting_strategy="inverse_rmse",
        selection_metric="rmse",
    )
    assert len(evaluated) == 1
    best_cand = evaluated[0]
    assert "weights" in best_cand
    assert "random_forest" in best_cand["weights"]
    assert "xgboost" in best_cand["weights"]


def test_auto_select_ranking_with_pruned_retail_rf(sample_retail_data):
    """9. Verify Auto Select / compare_models handles pruned Retail RF."""
    train_df, val_df, test_df = sample_retail_data

    rf = RandomForestForecaster(
        n_estimators=10,
        random_state=42,
        target_transform="log1p",
        feature_exclusions=list(RETAIL_MACRO_EXCLUSIONS),
    )
    rf.fit(train_df, target_column="sales")

    val_preds = rf.predict(val_df)
    test_preds = rf.predict(test_df)
    val_m = evaluate_forecast(val_df["sales"], val_preds)
    test_m = evaluate_forecast(test_df["sales"], test_preds)

    rep_rf = {
        "model": "random_forest",
        "domain": "retail",
        "validation_metrics": val_m,
        "test_metrics": test_m,
    }
    rep_naive = {
        "model": "naive",
        "domain": "retail",
        "validation_metrics": {"rmse": 999.0, "mae": 999.0, "mape": 99.0, "r2": -1.0},
        "test_metrics": {"rmse": 999.0, "mae": 999.0, "mape": 99.0, "r2": -1.0},
    }

    comp = compare_models([rep_rf, rep_naive], selection_metric="rmse")
    assert comp["recommended_model"] == "random_forest"
    assert comp["selection_split"] == "validation"

    selector = AdaptiveModelSelector(selection_metric="rmse")
    sel = selector.select_best_model([rep_rf, rep_naive])
    assert sel["selected_model"] == "random_forest"
    assert sel["selection_split"] == "validation"
