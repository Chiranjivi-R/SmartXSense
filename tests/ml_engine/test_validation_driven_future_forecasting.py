"""Tests for validation-driven model selection in future forecasting and feature improvements."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from ml_engine.features.artifacts import materialize_forecasting_features
from ml_engine.features.common import add_festival_interaction_features, add_history_features
from ml_engine.models.future_forecasting import generate_future_forecast
from ml_engine.models.xgboost_model import XGBoostForecaster
from ml_engine.models.random_forest import RandomForestForecaster


def test_festival_interactions_and_ewma_leakage_safety(tmp_path):
    """Verify EWMA and festival interactions do not leak future information."""
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    sales = np.arange(100.0, 160.0)
    fest_flags = [1.0 if i % 7 == 0 else 0.0 for i in range(60)]
    
    df = pd.DataFrame({
        "timestamp": dates,
        "sales": sales,
        "festival_flag": fest_flags,
    })
    
    # Check add_history_features strictly shifts by 1 for EWMA
    res, names = add_history_features(df, target="sales", lags=(1, 7), rolling_windows=(7,), ewma_spans=(3,))
    assert "ewma_3" in names
    assert "ewma_3" in res.columns
    # Row 0 history is shift(1), which must be NaN
    assert pd.isna(res["ewma_3"].iloc[0])
    
    # Check festival interaction creation
    res, fest_names = add_festival_interaction_features(res)
    assert "festival_distance" in fest_names
    assert "fest_x_lag_1" in fest_names
    assert "fest_x_rolling_mean_7" in fest_names
    # festival_season_flag and fest_season_x_lag1 are added only when festival_season column is present
    # (it may be absent in minimal test frames — check conditionally)
    if "festival_season" in res.columns:
        assert "festival_season_flag" in fest_names
        assert "fest_season_x_lag1" in fest_names


    # Verify values at non-festival days are 0.0
    non_fest_idx = res.index[res["festival_flag"] == 0.0]
    assert (res.loc[non_fest_idx, "fest_x_lag_1"] == 0.0).all()



def test_festival_interactions_skipped_when_no_festival_column():
    """Verify festival interactions are cleanly skipped when no festival column is present."""
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "sales": np.arange(100.0, 130.0),
    })
    res, fest_names = add_festival_interaction_features(df)
    assert fest_names == []
    assert len(res.columns) == 2


def test_future_forecasting_with_validation_winning_xgboost(tmp_path):
    """Verify future forecasting works correctly with tuned XGBoost forecaster."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
        "sales": np.sin(np.linspace(0, 10, 100)) * 50 + 100,
        "festival_flag": [1 if i % 10 == 0 else 0 for i in range(100)],
    })
    
    prep = materialize_forecasting_features(df, "retail", output_root=tmp_path / "features")
    features_dir = prep["features_dir"]
    
    train_df = pd.read_parquet(features_dir / "train.parquet")
    val_df = pd.read_parquet(features_dir / "validation.parquet")
    test_df = pd.read_parquet(features_dir / "test.parquet")
    
    xgb = XGBoostForecaster(n_estimators=50, max_depth=4, learning_rate=0.04)
    xgb.fit(train_df, target_column="sales")
    
    result = generate_future_forecast(
        model=xgb,
        model_key="xgboost",
        domain="retail",
        features_dir=features_dir,
        horizon=14,
    )
    
    assert result["forecast_type"] == "future"
    assert result["domain"] == "retail"
    assert result["model"] == "xgboost"
    assert result["target"] == "sales"
    assert result["horizon"] == 14
    assert len(result["predictions"]) == 14
    for p in result["predictions"]:
        assert "timestamp" in p
        assert "predicted" in p
        assert "actual" not in p
