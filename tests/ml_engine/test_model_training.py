"""Tests for model training and evaluation orchestration pipeline (trainer.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.trainer import TrainerError, run_training_pipeline


@pytest.fixture
def dummy_electricity_features_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "electricity" / "features_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = pd.date_range("2025-01-01", periods=100, freq="h")
    values = 100.0 + np.cumsum(np.random.randn(100))
    df = pd.DataFrame({"timestamp": timestamps, "demand_mw": values})

    train = df.iloc[:70].reset_index(drop=True)
    val = df.iloc[70:85].reset_index(drop=True)
    test = df.iloc[85:].reset_index(drop=True)

    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "validation.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)

    return output_dir


@pytest.fixture
def dummy_retail_features_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "retail" / "features_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamps = pd.date_range("2025-01-01", periods=60, freq="D")
    values = 50.0 + np.cumsum(np.random.randn(60))
    df = pd.DataFrame({"timestamp": timestamps, "sales": values})

    train = df.iloc[:40].reset_index(drop=True)
    val = df.iloc[40:50].reset_index(drop=True)
    test = df.iloc[50:].reset_index(drop=True)

    train.to_parquet(output_dir / "train.parquet", index=False)
    val.to_parquet(output_dir / "validation.parquet", index=False)
    test.to_parquet(output_dir / "test.parquet", index=False)

    return output_dir


def test_electricity_naive_pipeline(dummy_electricity_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="electricity",
        features_dir=dummy_electricity_features_dir,
        model_name="naive",
    )

    assert result["domain"] == "electricity"
    assert result["target"] == "demand_mw"
    assert result["model"] == "naive"
    assert "validation_metrics" in result
    assert "test_metrics" in result
    assert set(result["validation_metrics"].keys()) == {"mae", "rmse", "mape", "r2", "smape", "wape"}

    assert set(result["test_metrics"].keys()) == {"mae", "rmse", "mape", "r2", "smape", "wape"}



def test_electricity_arima_pipeline(dummy_electricity_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="electricity",
        features_dir=dummy_electricity_features_dir,
        model_name="arima",
        order=(1, 1, 1),
    )

    assert result["domain"] == "electricity"
    assert result["target"] == "demand_mw"
    assert result["model"] == "arima"
    assert result["configuration"]["order"] == [1, 1, 1]
    assert result["validation_metrics"]["mae"] >= 0.0
    assert result["test_metrics"]["mae"] >= 0.0


def test_retail_naive_pipeline(dummy_retail_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="retail",
        features_dir=dummy_retail_features_dir,
        model_name="naive",
    )

    assert result["domain"] == "retail"
    assert result["target"] == "sales"
    assert result["model"] == "naive"


def test_retail_arima_unsupported_error(dummy_retail_features_dir: Path) -> None:
    with pytest.raises(TrainerError, match="supported only for the electricity domain"):
        run_training_pipeline(
            domain="retail",
            features_dir=dummy_retail_features_dir,
            model_name="arima",
        )


def test_invalid_domain_raises(dummy_electricity_features_dir: Path) -> None:
    with pytest.raises(TrainerError, match="Unsupported domain"):
        run_training_pipeline(
            domain="unknown_domain",
            features_dir=dummy_electricity_features_dir,
            model_name="naive",
        )


def test_save_artifact_option(dummy_electricity_features_dir: Path, tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    result = run_training_pipeline(
        domain="electricity",
        features_dir=dummy_electricity_features_dir,
        model_name="naive",
        save_artifact=True,
        artifacts_root=artifacts_root,
    )

    assert "artifact_path" in result
    assert Path(result["artifact_path"]).exists()
