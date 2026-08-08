"""Tests for materializing forecasting feature splits from processed frames."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_engine.features.artifacts import materialize_forecasting_features
from ml_engine.features.common import FeatureEngineeringError
from ml_engine.splitting.time_series import TimeSeriesSplitError


def test_materialize_retail_feature_splits(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="D", tz="UTC"),
            "sales": [40 + i % 9 for i in range(50)],
            "price": [10.0] * 50,
            "city": ["Delhi"] * 50,
        }
    )
    result = materialize_forecasting_features(frame, "retail", output_root=tmp_path / "features")
    features_dir = Path(result["features_dir"])
    assert (features_dir / "train.parquet").exists()
    assert (features_dir / "validation.parquet").exists()
    assert (features_dir / "test.parquet").exists()
    assert (features_dir / "metadata.json").exists()
    assert result["train_rows"] > 0
    assert result["validation_rows"] > 0
    assert result["test_rows"] > 0
    assert "lag_1" in result["feature_names"]


def test_materialize_rejects_too_small_frame(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"),
            "sales": [10, 12],
        }
    )
    with pytest.raises((TimeSeriesSplitError, FeatureEngineeringError, ValueError)):
        materialize_forecasting_features(frame, "retail", output_root=tmp_path / "features")
