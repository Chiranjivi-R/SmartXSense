"""Tests for tabular preprocessing and end-to-end ML trainer pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.tabular_preprocessing import PreprocessingError, TabularFeatureTransformer
from ml_engine.models.trainer import TrainerError, run_training_pipeline


def test_categorical_encoding_train_only_and_unseen_categories() -> None:
    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5),
        "sales": [10.0, 20.0, 30.0, 40.0, 50.0],
        "category": ["A", "B", "A", "C", "B"],
        "lag_1": [5.0, 10.0, 20.0, 30.0, 40.0],
    })

    eval_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-06", periods=3),
        "sales": [60.0, 70.0, 80.0],
        "category": ["A", "D", "B"],  # "D" is unseen in training!
        "lag_1": [50.0, 60.0, 70.0],
    })

    transformer = TabularFeatureTransformer(target_column="sales")
    transformer.fit(train_df)

    X_train = transformer.transform(train_df)
    X_eval = transformer.transform(eval_df)

    assert X_train.shape == (5, 2)  # category, lag_1
    assert X_eval.shape == (3, 2)

    # Index 1 of eval is "D" -> must be encoded as -1 (unknown value)
    # Finding column index for category
    cat_idx = transformer.feature_names.index("category")
    assert X_eval[1, cat_idx] == -1.0


def test_missing_lag_imputation_preserves_rows() -> None:
    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=5),
        "sales": [10.0, 20.0, 30.0, 40.0, 50.0],
        "lag_1": [np.nan, 10.0, 20.0, 30.0, 40.0],  # First row missing lag
    })

    transformer = TabularFeatureTransformer(target_column="sales")
    transformer.fit(train_df)

    X_train = transformer.transform(train_df)
    assert X_train.shape[0] == 5
    assert not np.isnan(X_train).any()


def test_target_and_revenue_leakage_exclusion() -> None:
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=3),
        "sales": [100.0, 200.0, 300.0],
        "revenue": [1000.0, 2000.0, 3000.0],  # Leakage column
        "price": [10.0, 10.0, 10.0],
        "lag_1": [90.0, 190.0, 290.0],
    })

    transformer = TabularFeatureTransformer(target_column="sales")
    transformer.fit(df)

    assert "sales" not in transformer.feature_names
    assert "timestamp" not in transformer.feature_names
    assert "revenue" not in transformer.feature_names
    assert set(transformer.feature_names) == {"price", "lag_1"}


@pytest.fixture
def dummy_retail_features_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "retail" / "features_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = pd.date_range("2025-01-01", periods=60, freq="D")
    values = 50.0 + np.cumsum(np.random.randn(60))
    categories = np.random.choice(["Electronics", "Clothing", "Home"], size=60)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "sales": values,
        "category": categories,
        "lag_1": pd.Series(values).shift(1).bfill(),
        "lag_7": pd.Series(values).shift(7).bfill(),
    })

    train = df.iloc[:40].reset_index(drop=True)
    val = df.iloc[40:50].reset_index(drop=True)
    test = df.iloc[50:].reset_index(drop=True)

    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "validation.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)

    return output_dir


@pytest.fixture
def dummy_electricity_features_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "electricity" / "features_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = pd.date_range("2025-01-01", periods=80, freq="h")
    values = 100.0 + np.cumsum(np.random.randn(80))
    df = pd.DataFrame({
        "timestamp": timestamps,
        "demand_mw": values,
        "hour": timestamps.hour,
        "lag_1": pd.Series(values).shift(1).bfill(),
        "lag_24": pd.Series(values).shift(24).bfill(),
    })

    train = df.iloc[:50].reset_index(drop=True)
    val = df.iloc[50:65].reset_index(drop=True)
    test = df.iloc[65:].reset_index(drop=True)

    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "validation.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)

    return output_dir


def test_retail_random_forest_pipeline(dummy_retail_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="retail",
        features_dir=dummy_retail_features_dir,
        model_name="random_forest",
        n_estimators=10,
        random_state=42,
    )

    assert result["domain"] == "retail"
    assert result["target"] == "sales"
    assert result["model"] == "random_forest"
    assert "feature_importances" in result
    assert result["feature_count"] > 0
    assert set(result["validation_metrics"].keys()) == {"mae", "rmse", "mape", "r2", "smape", "wape"}

    assert set(result["test_metrics"].keys()) == {"mae", "rmse", "mape", "r2", "smape", "wape"}



def test_retail_xgboost_pipeline(dummy_retail_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="retail",
        features_dir=dummy_retail_features_dir,
        model_name="xgboost",
        n_estimators=10,
        random_state=42,
    )

    assert result["domain"] == "retail"
    assert result["target"] == "sales"
    assert result["model"] == "xgboost"
    assert "feature_importances" in result


def test_electricity_random_forest_pipeline(dummy_electricity_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="electricity",
        features_dir=dummy_electricity_features_dir,
        model_name="random_forest",
        n_estimators=10,
        random_state=42,
    )

    assert result["domain"] == "electricity"
    assert result["target"] == "demand_mw"
    assert result["model"] == "random_forest"


def test_electricity_xgboost_pipeline(dummy_electricity_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="electricity",
        features_dir=dummy_electricity_features_dir,
        model_name="xgboost",
        n_estimators=10,
        random_state=42,
    )

    assert result["domain"] == "electricity"
    assert result["target"] == "demand_mw"
    assert result["model"] == "xgboost"


def test_backward_compatibility_baselines(dummy_electricity_features_dir: Path) -> None:
    naive_res = run_training_pipeline(
        domain="electricity",
        features_dir=dummy_electricity_features_dir,
        model_name="naive",
    )
    assert naive_res["model"] == "naive"

    arima_res = run_training_pipeline(
        domain="electricity",
        features_dir=dummy_electricity_features_dir,
        model_name="arima",
    )
    assert arima_res["model"] == "arima"


def test_all_missing_training_feature_exclusion() -> None:
    import warnings
    from ml_engine.models.random_forest import RandomForestForecaster
    from ml_engine.models.xgboost_model import XGBoostForecaster

    train_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=10),
        "sales": np.arange(10, dtype=float),
        "valid_feature": np.arange(10, dtype=float) * 2.0,
        "all_missing_feature": [np.nan] * 10,
    })

    eval_df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-11", periods=5),
        "sales": np.arange(10, 15, dtype=float),
        "valid_feature": np.arange(10, 15, dtype=float) * 2.0,
        "all_missing_feature": [1.0, 2.0, 3.0, 4.0, 5.0],  # Value in eval, but missing in train!
    })

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        transformer = TabularFeatureTransformer(target_column="sales")
        transformer.fit(train_df)
        X_train = transformer.transform(train_df)
        X_eval = transformer.transform(eval_df)

    # Prove no sklearn all-missing imputation warning occurred
    imputer_warnings = [
        w for w in recorded_warnings if "Skipping features without any observed values" in str(w.message)
    ]
    assert len(imputer_warnings) == 0

    # Prove all_missing_feature is excluded
    assert "all_missing_feature" not in transformer.feature_names
    assert transformer.feature_names == ["valid_feature"]
    assert X_train.shape == (10, 1)
    assert X_eval.shape == (5, 1)

    # Prove Random Forest alignment
    rf = RandomForestForecaster(n_estimators=5, random_state=42)
    rf.fit(train_df, target_column="sales")
    rf_preds = rf.predict(eval_df)
    assert len(rf_preds) == len(eval_df)
    assert rf.feature_names == ["valid_feature"]
    assert list(rf.get_feature_importances().keys()) == ["valid_feature"]

    # Prove XGBoost alignment
    xgb = XGBoostForecaster(n_estimators=5, random_state=42)
    xgb.fit(train_df, target_column="sales")
    xgb_preds = xgb.predict(eval_df)
    assert len(xgb_preds) == len(eval_df)
    assert xgb.feature_names == ["valid_feature"]
    assert list(xgb.get_feature_importances().keys()) == ["valid_feature"]
