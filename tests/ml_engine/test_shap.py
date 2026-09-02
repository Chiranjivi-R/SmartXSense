"""Unit tests for SHAP explainability layer on Random Forest and XGBoost forecasters."""

import numpy as np
import pandas as pd
import pytest

from ml_engine.explainability import ExplainabilityError, ForecastExplainer, ShapTreeExplainer
from ml_engine.models.arima import ARIMAForecaster
from ml_engine.models.random_forest import RandomForestForecaster
from ml_engine.models.xgboost_model import XGBoostForecaster


@pytest.fixture()
def synthetic_data():
    np.random.seed(42)
    n = 60
    dates = pd.date_range("2026-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "demand_mw": np.sin(np.linspace(0, 10, n)) * 100 + 500 + np.random.randn(n) * 10,
        "feature_lag1": np.roll(np.sin(np.linspace(0, 10, n)), 1),
        "feature_rolling": np.random.rand(n) * 50,
        "category_code": np.random.choice(["cat_a", "cat_b", "cat_c"], size=n),
    })
    train_df = df.iloc[:40].copy()
    val_df = df.iloc[40:].copy()
    return train_df, val_df


def test_shap_random_forest_global_and_local(synthetic_data):
    train_df, val_df = synthetic_data
    rf = RandomForestForecaster(n_estimators=10, random_state=42)
    rf.fit(train_df, target_column="demand_mw")

    explainer = ForecastExplainer(rf, model_name="random_forest")

    # Global SHAP
    global_exp = explainer.explain_global(val_df, method="shap")
    assert global_exp["explanation_type"] == "global"
    assert global_exp["method"] == "shap"
    assert global_exp["model"] == "random_forest"
    assert len(global_exp["feature_importance"]) == len(rf.feature_names)

    # Verify feature dimension alignment
    assert len(global_exp["feature_importance"]) == rf._transformer.transform(val_df).shape[1]

    # Local SHAP
    local_exp = explainer.explain_local(val_df, sample_index=2, method="shap")
    assert local_exp["explanation_type"] == "local"
    assert local_exp["method"] == "shap"
    assert local_exp["sample_index"] == 2
    assert isinstance(local_exp["prediction"], float)
    assert isinstance(local_exp["base_value"], float)
    assert isinstance(local_exp["top_positive_contributors"], list)
    assert isinstance(local_exp["top_negative_contributors"], list)


def test_shap_xgboost_global_and_local(synthetic_data):
    train_df, val_df = synthetic_data
    xgb = XGBoostForecaster(n_estimators=10, random_state=42)
    xgb.fit(train_df, target_column="demand_mw")

    explainer = ForecastExplainer(xgb, model_name="xgboost")

    global_exp = explainer.explain_global(val_df, method="shap")
    assert global_exp["model"] == "xgboost"
    assert len(global_exp["feature_importance"]) == len(xgb.feature_names)

    local_exp = explainer.explain_local(val_df, sample_index=0, method="shap")
    assert local_exp["model"] == "xgboost"
    assert len(local_exp["shap_contributions"]) == len(xgb.feature_names)


def test_shap_preprocessing_consistency_no_refit(synthetic_data):
    train_df, val_df = synthetic_data
    rf = RandomForestForecaster(n_estimators=10, random_state=42)
    rf.fit(train_df, target_column="demand_mw")

    transformer_fitted = rf._transformer._is_fitted
    categories_learned = list(rf._transformer.categorical_columns)

    shap_tree = ShapTreeExplainer(rf)
    _ = shap_tree.explain_global(val_df)

    # Ensure transformer was NOT refitted
    assert rf._transformer._is_fitted == transformer_fitted
    assert list(rf._transformer.categorical_columns) == categories_learned


def test_shap_unsupported_model_rejection():
    with pytest.raises(ExplainabilityError, match="Explainability is supported for"):
        ForecastExplainer(None, model_name="arima")


def test_forecast_explainer_non_empty_feature_importance(synthetic_data):
    train_df, val_df = synthetic_data
    rf = RandomForestForecaster(n_estimators=10, random_state=42)
    rf.fit(train_df, target_column="demand_mw")

    explainer = ForecastExplainer(rf, model_name="random_forest")
    exp = explainer.explain_global(val_df)

    assert exp["status"] == "success"
    assert exp["model"] == "random_forest"
    assert "feature_importance" in exp
    assert len(exp["feature_importance"]) > 0
    assert "features" in exp
    assert len(exp["features"]) > 0
    assert isinstance(exp["features"], list)
    assert "feature" in exp["features"][0]
    assert "importance" in exp["features"][0]
