"""Regression tests for LSTM feature schema validation and input dimension isolation.

Verifies:
1. Matching LSTM feature dimensions reuses fitted artifacts correctly.
2. Changed feature schema / input dimension rejects stale cached artifacts and retrains cleanly.
3. Stale / incompatible LSTM artifact rejection in check_compatibility and try_load_compatible_artifact.
4. Model comparison / benchmark evaluation after switching between datasets with different feature counts.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from ml_engine.models.artifact_store import (
    ArtifactError,
    check_compatibility,
    save_model_artifact,
    try_load_compatible_artifact,
)
from ml_engine.models.lstm import LSTMForecaster, LSTMModelError
from ml_engine.models.tabular_preprocessing import TabularFeatureTransformer
from ml_engine.models.trainer import (
    evaluate_fitted_model_on_splits,
    run_training_pipeline,
)


@pytest.fixture()
def dataset_20_features(tmp_path):
    """Create feature split directory with 20 input features."""
    fdir = tmp_path / "features_ds20"
    fdir.mkdir()

    dates = pd.date_range("2026-01-01", periods=60)
    data = {"timestamp": dates, "sales": np.linspace(100, 200, 60)}
    for i in range(20):
        data[f"feat_{i:02d}"] = np.random.randn(60)

    df = pd.DataFrame(data)
    train_df = df.iloc[:40]
    val_df = df.iloc[40:50]
    test_df = df.iloc[50:]

    train_df.to_parquet(fdir / "train.parquet")
    val_df.to_parquet(fdir / "validation.parquet")
    test_df.to_parquet(fdir / "test.parquet")
    return fdir


@pytest.fixture()
def dataset_19_features(tmp_path):
    """Create feature split directory with 19 input features (one feature dropped)."""
    fdir = tmp_path / "features_ds19"
    fdir.mkdir()

    dates = pd.date_range("2026-01-01", periods=60)
    data = {"timestamp": dates, "sales": np.linspace(100, 200, 60)}
    for i in range(19):
        data[f"feat_{i:02d}"] = np.random.randn(60)

    df = pd.DataFrame(data)
    train_df = df.iloc[:40]
    val_df = df.iloc[40:50]
    test_df = df.iloc[50:]

    train_df.to_parquet(fdir / "train.parquet")
    val_df.to_parquet(fdir / "validation.parquet")
    test_df.to_parquet(fdir / "test.parquet")
    return fdir


def test_matching_lstm_feature_dimensions(dataset_20_features, tmp_path):
    """1. Matching feature dimensions load and evaluate cleanly."""
    artifacts_root = tmp_path / "artifacts"

    res1 = run_training_pipeline(
        domain="retail",
        features_dir=dataset_20_features,
        model_name="lstm",
        epochs=1,
        save_artifact=True,
        artifacts_root=artifacts_root,
    )
    assert res1["feature_count"] == 20
    assert res1["artifact_persistence"]["saved"] is True

    # Second run should reuse the artifact
    res2 = run_training_pipeline(
        domain="retail",
        features_dir=dataset_20_features,
        model_name="lstm",
        epochs=1,
        save_artifact=True,
        artifacts_root=artifacts_root,
    )
    assert res2["feature_count"] == 20
    assert res2["artifact_persistence"]["reused"] is True


def test_changed_feature_schema_rejects_artifact_and_retrains(
    dataset_20_features, dataset_19_features, tmp_path
):
    """2. Changed feature schema / dimension rejects 20-feature artifact and trains 19-feature model cleanly."""
    artifacts_root = tmp_path / "artifacts"

    # Step 1: Train and save LSTM artifact with 20 features
    res20 = run_training_pipeline(
        domain="retail",
        features_dir=dataset_20_features,
        model_name="lstm",
        epochs=1,
        save_artifact=True,
        artifacts_root=artifacts_root,
    )
    assert res20["feature_count"] == 20

    # Step 2: Try to run pipeline on dataset with 19 features using the same artifact root
    # Should NOT crash with 20-vs-19 dimension mismatch error; should retrain cleanly
    res19 = run_training_pipeline(
        domain="retail",
        features_dir=dataset_19_features,
        model_name="lstm",
        epochs=1,
        save_artifact=True,
        artifacts_root=artifacts_root,
    )
    assert res19["feature_count"] == 19
    assert res19["artifact_persistence"]["reused"] is False


def test_check_compatibility_feature_schema_mismatch():
    """3. check_compatibility raises ArtifactError when feature_names do not match."""
    metadata = {
        "artifact_version": 1,
        "domain": "retail",
        "model": "lstm",
        "target": "sales",
        "features_id": "feat_dir",
        "dataset_fingerprint": "fp123",
        "feature_names": ["f1", "f2", "f3"],
        "feature_count": 3,
    }

    # Same names -> passes
    check_compatibility(
        metadata,
        domain="retail",
        model="lstm",
        target="sales",
        features_id="feat_dir",
        dataset_fingerprint="fp123",
        feature_names=["f1", "f2", "f3"],
    )

    # Different count -> raises ArtifactError
    with pytest.raises(ArtifactError, match="Artifact feature schema mismatch"):
        check_compatibility(
            metadata,
            domain="retail",
            model="lstm",
            target="sales",
            features_id="feat_dir",
            dataset_fingerprint="fp123",
            feature_names=["f1", "f2"],
        )


def test_lstm_predict_raises_on_dimension_mismatch(dataset_20_features, dataset_19_features):
    """4. LSTMForecaster.predict raises explicit LSTMModelError if dimension mismatch occurs."""
    train_df = pd.read_parquet(dataset_20_features / "train.parquet")
    eval_df = pd.read_parquet(dataset_19_features / "validation.parquet")

    model = LSTMForecaster(epochs=1)
    model.fit(train_df, "sales")

    with pytest.raises(LSTMModelError, match="Sequence preprocessing failed|dimension mismatch"):
        model.predict(eval_df, "sales")



def test_benchmark_evaluation_after_dataset_switch(
    dataset_20_features, dataset_19_features, tmp_path
):
    from app import create_app
    from app.services.ml_service import compare_service


    artifacts_root = tmp_path / "artifacts"

    class TestCfg:
        TESTING = True
        SECRET_KEY = "test"
        JWT_SECRET_KEY = "jwt-secret-at-least-32-bytes-long"
        SQLALCHEMY_DATABASE_URI = "sqlite://"
        SQLALCHEMY_TRACK_MODIFICATIONS = False
        CORS_ORIGINS = ["http://localhost:5173"]
        ML_ARTIFACTS_ROOT = str(artifacts_root)
        UPLOAD_FOLDER = str(tmp_path / "uploads")
        PROCESSED_DATA_FOLDER = str(tmp_path / "processed")
        FEATURES_DATA_FOLDER = str(tmp_path / "features")

    app = create_app(TestCfg)
    with app.app_context():
        # Benchmark on 20-feature dataset
        res20 = compare_service({
            "domain": "retail",
            "features_dir": str(dataset_20_features),
            "models": ["random_forest", "lstm"],
        })
        assert res20["recommended_model"]

        # Benchmark on 19-feature dataset (must not raise 400 dimension mismatch error)
        res19 = compare_service({
            "domain": "retail",
            "features_dir": str(dataset_19_features),
            "models": ["random_forest", "lstm"],
        })
        assert res19["recommended_model"]
