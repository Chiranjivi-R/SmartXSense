"""Tests for XGBoost Regressor supervised forecasting model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.xgboost_model import XGBoostForecaster, XGBoostModelError


def test_xgboost_fit_predict_and_determinism() -> None:
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

    model1 = XGBoostForecaster(n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42)
    model1.fit(train_df, target_column="demand_mw")
    preds1 = model1.predict(eval_df)

    assert len(preds1) == len(eval_df)
    assert not preds1.isna().any()

    # Determinism
    model2 = XGBoostForecaster(n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42)
    model2.fit(train_df, target_column="demand_mw")
    preds2 = model2.predict(eval_df)

    pd.testing.assert_series_equal(preds1, preds2)


def test_xgboost_configurable_parameters() -> None:
    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=20),
        "demand_mw": np.arange(20, dtype=float),
        "lag_1": np.arange(20, dtype=float) - 1,
    })

    model = XGBoostForecaster(
        n_estimators=15,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=123,
    )
    model.fit(train_df, target_column="demand_mw")

    assert model.n_estimators == 15
    assert model.max_depth == 4
    assert model.learning_rate == 0.08
    assert model.subsample == 0.9
    assert model.colsample_bytree == 0.9
    assert model.random_state == 123


def test_xgboost_feature_importances() -> None:
    np.random.seed(42)
    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=50),
        "demand_mw": np.arange(50, dtype=float),
        "strong_feature": np.arange(50, dtype=float) + np.random.randn(50) * 0.01,
        "noise_feature": np.random.randn(50),
    })

    model = XGBoostForecaster(n_estimators=20, max_depth=3, random_state=42)
    model.fit(train_df, target_column="demand_mw")

    importances = model.get_feature_importances()
    assert set(importances.keys()) == {"strong_feature", "noise_feature"}
    assert importances["strong_feature"] > importances["noise_feature"]


def test_xgboost_unfitted_raises() -> None:
    model = XGBoostForecaster()
    with pytest.raises(XGBoostModelError, match="must be fitted"):
        model.predict(pd.DataFrame({"lag_1": [1, 2]}))

    with pytest.raises(XGBoostModelError, match="must be fitted"):
        model.get_feature_importances()


def test_xgboost_log1p_target_transform() -> None:
    np.random.seed(42)
    timestamps = pd.date_range("2025-01-01", periods=60, freq="D")
    sales = np.exp(np.random.uniform(2.0, 5.0, size=60))  # positive sales
    lag_1 = pd.Series(sales).shift(1).bfill().values

    train_df = pd.DataFrame({
        "timestamp": timestamps[:40],
        "sales": sales[:40],
        "lag_1": lag_1[:40],
    })
    eval_df = pd.DataFrame({
        "timestamp": timestamps[40:],
        "sales": sales[40:],
        "lag_1": lag_1[40:],
    })

    model = XGBoostForecaster(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        target_transform="log1p",
    )
    model.fit(train_df, target_column="sales")
    preds = model.predict(eval_df)

    assert len(preds) == len(eval_df)
    assert not preds.isna().any()
    assert (preds >= 0.0).all()

    # Determinism
    model2 = XGBoostForecaster(
        n_estimators=30,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        target_transform="log1p",
    )
    model2.fit(train_df, target_column="sales")
    preds2 = model2.predict(eval_df)
    pd.testing.assert_series_equal(preds, preds2)

    # Invalid transform check
    with pytest.raises(XGBoostModelError, match="Unsupported target_transform"):
        XGBoostForecaster(target_transform="invalid_transform")

    # Negative target check
    neg_train = train_df.copy()
    neg_train.loc[0, "sales"] = -5.0
    bad_model = XGBoostForecaster(target_transform="log1p")
    with pytest.raises(XGBoostModelError, match="Cannot apply log1p transform"):
        bad_model.fit(neg_train, target_column="sales")
