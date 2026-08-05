"""Unit tests for LIME tabular explainer."""

import numpy as np
import pandas as pd
import pytest

from ml_engine.explainability import ForecastExplainer, LimeTabularExplainer
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
    })
    train_df = df.iloc[:40].copy()
    val_df = df.iloc[40:].copy()
    return train_df, val_df


def test_lime_random_forest_local_explanation(synthetic_data):
    train_df, val_df = synthetic_data
    rf = RandomForestForecaster(n_estimators=10, random_state=42)
    rf.fit(train_df, target_column="demand_mw")

    explainer = ForecastExplainer(rf, model_name="random_forest")
    local_exp = explainer.explain_local(val_df, sample_index=1, method="lime", train_df=train_df)

    assert local_exp["explanation_type"] == "local"
    assert local_exp["method"] == "lime"
    assert local_exp["model"] == "random_forest"
    assert isinstance(local_exp["prediction"], float)
    assert isinstance(local_exp["intercept"], float)
    assert isinstance(local_exp["feature_attributions"], list)
    assert len(local_exp["feature_attributions"]) > 0


def test_lime_xgboost_local_explanation(synthetic_data):
    train_df, val_df = synthetic_data
    xgb = XGBoostForecaster(n_estimators=10, random_state=42)
    xgb.fit(train_df, target_column="demand_mw")

    lime_exp = LimeTabularExplainer(xgb, train_df=train_df)
    res = lime_exp.explain_local(val_df, sample_index=0)

    assert res["explanation_type"] == "local"
    assert res["method"] == "lime"
    assert isinstance(res["prediction"], float)
    assert len(res["feature_attributions"]) == len(xgb.feature_names)
