"""Tests for Random Forest Regressor supervised forecasting model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.random_forest import RandomForestForecaster, RandomForestModelError


def test_random_forest_fit_predict_and_determinism() -> None:
    np.random.seed(42)
    timestamps = pd.date_range("2025-01-01", periods=60, freq="h")
    target = 100.0 + np.cumsum(np.random.randn(60))
    feature1 = target + np.random.randn(60) * 2.0
    feature2 = np.random.randn(60)

    train_df = pd.DataFrame({
        "timestamp": timestamps[:40],
        "demand_mw": target[:40],
        "lag_1": feature1[:40],
        "rolling_mean_24": feature2[:40],
    })

    eval_df = pd.DataFrame({
        "timestamp": timestamps[40:],
        "demand_mw": target[40:],
        "lag_1": feature1[40:],
        "rolling_mean_24": feature2[40:],
    })

    model1 = RandomForestForecaster(n_estimators=10, random_state=42)
    model1.fit(train_df, target_column="demand_mw")
    preds1 = model1.predict(eval_df)

    assert len(preds1) == len(eval_df)
    assert not preds1.isna().any()

    # Verify determinism
    model2 = RandomForestForecaster(n_estimators=10, random_state=42)
    model2.fit(train_df, target_column="demand_mw")
    preds2 = model2.predict(eval_df)

    pd.testing.assert_series_equal(preds1, preds2)


def test_random_forest_feature_exclusions() -> None:
    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10),
        "sales": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "revenue": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],  # Leakage column
        "lag_1": [9, 19, 29, 39, 49, 59, 69, 79, 89, 99],
    })

    model = RandomForestForecaster(n_estimators=5, random_state=42)
    model.fit(train_df, target_column="sales")

    assert "sales" not in model.feature_names
    assert "timestamp" not in model.feature_names
    assert "revenue" not in model.feature_names
    assert model.feature_names == ["lag_1"]


def test_random_forest_feature_importances() -> None:
    np.random.seed(42)
    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=50),
        "sales": np.arange(50, dtype=float),
        "strong_feature": np.arange(50, dtype=float) + np.random.randn(50) * 0.01,
        "noise_feature": np.random.randn(50),
    })

    model = RandomForestForecaster(n_estimators=20, random_state=42)
    model.fit(train_df, target_column="sales")

    importances = model.get_feature_importances()
    assert set(importances.keys()) == {"strong_feature", "noise_feature"}
    assert importances["strong_feature"] > importances["noise_feature"]


def test_random_forest_unfitted_raises() -> None:
    model = RandomForestForecaster()
    with pytest.raises(RandomForestModelError, match="must be fitted"):
        model.predict(pd.DataFrame({"lag_1": [1, 2]}))

    with pytest.raises(RandomForestModelError, match="must be fitted"):
        model.get_feature_importances()


def test_random_forest_log1p_target_transform() -> None:
    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=30),
        "sales": [10.0, 15.0, 25.0, 50.0, 80.0] * 6,
        "lag_1": [8.0, 12.0, 20.0, 45.0, 70.0] * 6,
    })
    eval_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-02-01", periods=10),
        "sales": [12.0, 18.0] * 5,
        "lag_1": [10.0, 16.0] * 5,
    })

    model = RandomForestForecaster(n_estimators=10, random_state=42, target_transform="log1p")
    model.fit(train_df, target_column="sales")
    assert model.target_transform == "log1p"

    preds = model.predict(eval_df)
    assert len(preds) == len(eval_df)
    assert (preds >= 0.0).all()
    # Predictions should be in the physical range of sales (~10 to 80), not in log space (~2 to 4)
    assert preds.mean() > 5.0


def test_random_forest_unsupported_target_transform_raises() -> None:
    with pytest.raises(RandomForestModelError, match="Unsupported target_transform"):
        RandomForestForecaster(target_transform="invalid_transform")


def test_random_forest_negative_target_with_log1p_raises() -> None:
    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10),
        "sales": [-5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
        "lag_1": [5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0],
    })
    model = RandomForestForecaster(n_estimators=5, random_state=42, target_transform="log1p")
    with pytest.raises(RandomForestModelError, match="Cannot apply log1p transform"):
        model.fit(train_df, target_column="sales")

