"""Tests for SmartXSense model artifact persistence (Phase A)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.artifact_store import (
    ARTIFACT_VERSION,
    ArtifactError,
    build_metadata,
    check_compatibility,
    config_fingerprint,
    derive_features_id,
    is_persistence_supported,
    load_model_artifact,
    save_model_artifact,
    try_load_compatible_artifact,
)
from ml_engine.models.random_forest import RandomForestForecaster
from ml_engine.models.trainer import (
    evaluate_fitted_model_on_splits,
    expected_configuration_for_request,
    run_training_pipeline,
)


def _write_splits(features_dir: Path, *, n: int = 80, domain: str = "retail") -> Path:
    features_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    sales = 40 + np.sin(np.linspace(0, 8, n)) * 10 + rng.normal(0, 1.5, n)
    demand = 500 + np.sin(np.linspace(0, 8, n)) * 40 + rng.normal(0, 5, n)
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sales": sales,
            "demand_mw": demand,
            "lag_1": np.roll(sales if domain == "retail" else demand, 1),
            "lag_7": np.roll(sales if domain == "retail" else demand, 7),
            "rolling_mean_7": pd.Series(sales if domain == "retail" else demand).rolling(7, min_periods=1).mean(),
            "day_of_week": timestamps.dayofweek,
            "is_weekend": (timestamps.dayofweek >= 5).astype(int),
        }
    )
    frame.iloc[0, frame.columns.get_loc("lag_1")] = frame.iloc[1]["lag_1"]
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
    frame.iloc[:train_end].to_parquet(features_dir / "train.parquet", index=False)
    frame.iloc[train_end:val_end].to_parquet(features_dir / "validation.parquet", index=False)
    frame.iloc[val_end:].to_parquet(features_dir / "test.parquet", index=False)
    return features_dir


@pytest.fixture()
def retail_features_dir(tmp_path: Path) -> Path:
    return _write_splits(tmp_path / "features_retail_fixture_aaa", domain="retail")


@pytest.fixture()
def artifacts_root(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts" / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_metadata_creation_contains_required_fields() -> None:
    meta = build_metadata(
        domain="retail",
        model="random_forest",
        target="sales",
        feature_names=["lag_1", "lag_7"],
        configuration={"n_estimators": 10, "max_depth": None, "random_state": 42},
        features_id="features_abc",
        validation_metrics={"rmse": 1.2},
        test_metrics={"rmse": 2.3},
        serialization_format="pickle",
    )
    assert meta["artifact_version"] == ARTIFACT_VERSION
    assert meta["domain"] == "retail"
    assert meta["model"] == "random_forest"
    assert meta["target"] == "sales"
    assert meta["feature_names"] == ["lag_1", "lag_7"]
    assert meta["feature_count"] == 2
    assert meta["features_id"] == "features_abc"
    assert "training_timestamp" in meta
    assert meta["validation_metrics"]["rmse"] == 1.2
    assert meta["test_metrics"]["rmse"] == 2.3
    assert meta["configuration"]["n_estimators"] == 10


def test_derive_features_id_uses_basename_only(tmp_path: Path) -> None:
    path = tmp_path / "features_1c0c92ff-264e-4398-b83c-e0e082b956b4"
    path.mkdir()
    assert derive_features_id(path) == "features_1c0c92ff-264e-4398-b83c-e0e082b956b4"
    assert derive_features_id(str(path)) == "features_1c0c92ff-264e-4398-b83c-e0e082b956b4"


def test_hybrid_persistence_unsupported() -> None:
    assert not is_persistence_supported("hybrid")
    assert is_persistence_supported("random_forest")


def test_save_and_load_random_forest(retail_features_dir: Path, artifacts_root: Path) -> None:
    config = expected_configuration_for_request(
        model_key="random_forest",
        domain="retail",
        n_estimators=25,
        random_state=42,
    )
    result = run_training_pipeline(
        domain="retail",
        features_dir=retail_features_dir,
        model_name="random_forest",
        n_estimators=25,
        save_artifact=True,
        artifacts_root=artifacts_root,
    )
    assert result["artifact_persistence"]["saved"] is True
    assert Path(result["artifact_path"]).exists()
    assert (Path(result["artifact_path"]) / "metadata.json").is_file()

    features_id = derive_features_id(retail_features_dir)
    loaded = load_model_artifact(
        domain="retail",
        model_key="random_forest",
        target="sales",
        features_id=features_id,
        configuration=config,
        artifacts_root=artifacts_root,
    )
    assert isinstance(loaded["model"], RandomForestForecaster)
    assert loaded["metadata"]["artifact_version"] == ARTIFACT_VERSION


def test_compatible_artifact_detection(retail_features_dir: Path, artifacts_root: Path) -> None:
    config = expected_configuration_for_request(model_key="random_forest", domain="retail", n_estimators=20)
    run_training_pipeline(
        domain="retail",
        features_dir=retail_features_dir,
        model_name="random_forest",
        n_estimators=20,
        save_artifact=True,
        artifacts_root=artifacts_root,
    )
    loaded = try_load_compatible_artifact(
        domain="retail",
        model_key="random_forest",
        target="sales",
        features_id=derive_features_id(retail_features_dir),
        configuration=config,
        artifacts_root=artifacts_root,
    )
    assert loaded is not None


def test_wrong_domain_rejection(retail_features_dir: Path, artifacts_root: Path) -> None:
    config = expected_configuration_for_request(model_key="random_forest", domain="retail", n_estimators=15)
    run_training_pipeline(
        domain="retail",
        features_dir=retail_features_dir,
        model_name="random_forest",
        n_estimators=15,
        save_artifact=True,
        artifacts_root=artifacts_root,
    )
    # Different domain resolves to a different controlled directory → missing artifact.
    assert (
        try_load_compatible_artifact(
            domain="electricity",
            model_key="random_forest",
            target="sales",
            features_id=derive_features_id(retail_features_dir),
            configuration=config,
            artifacts_root=artifacts_root,
        )
        is None
    )
    meta = build_metadata(
        domain="retail",
        model="random_forest",
        target="sales",
        feature_names=["lag_1"],
        configuration=config,
        features_id=derive_features_id(retail_features_dir),
        serialization_format="pickle",
    )
    with pytest.raises(ArtifactError, match="domain"):
        check_compatibility(
            meta,
            domain="electricity",
            model="random_forest",
            target="sales",
            features_id=derive_features_id(retail_features_dir),
            configuration=config,
        )


def test_wrong_model_rejection(retail_features_dir: Path, artifacts_root: Path) -> None:
    config = expected_configuration_for_request(model_key="random_forest", domain="retail", n_estimators=15)
    run_training_pipeline(
        domain="retail",
        features_dir=retail_features_dir,
        model_name="random_forest",
        n_estimators=15,
        save_artifact=True,
        artifacts_root=artifacts_root,
    )
    # Same features_id/config fingerprint folder won't exist for xgboost with different config,
    # so missing directory → ArtifactError. Also verify explicit metadata model check.
    meta = build_metadata(
        domain="retail",
        model="random_forest",
        target="sales",
        feature_names=["lag_1"],
        configuration=config,
        features_id=derive_features_id(retail_features_dir),
        serialization_format="pickle",
    )
    with pytest.raises(ArtifactError, match="model"):
        check_compatibility(
            meta,
            domain="retail",
            model="xgboost",
            target="sales",
            features_id=derive_features_id(retail_features_dir),
            configuration=config,
        )


def test_wrong_target_rejection() -> None:
    meta = build_metadata(
        domain="retail",
        model="random_forest",
        target="sales",
        feature_names=["lag_1"],
        configuration={"n_estimators": 10},
        features_id="features_x",
        serialization_format="pickle",
    )
    with pytest.raises(ArtifactError, match="target"):
        check_compatibility(
            meta,
            domain="retail",
            model="random_forest",
            target="demand_mw",
            features_id="features_x",
            configuration={"n_estimators": 10},
        )


def test_feature_mismatch_rejection() -> None:
    meta = build_metadata(
        domain="retail",
        model="random_forest",
        target="sales",
        feature_names=["lag_1", "lag_7"],
        configuration={"n_estimators": 10},
        features_id="features_x",
        serialization_format="pickle",
    )
    with pytest.raises(ArtifactError, match="feature names"):
        check_compatibility(
            meta,
            domain="retail",
            model="random_forest",
            target="sales",
            features_id="features_x",
            feature_names=["lag_1", "other"],
            configuration={"n_estimators": 10},
        )


def test_wrong_artifact_version_rejection() -> None:
    meta = build_metadata(
        domain="retail",
        model="random_forest",
        target="sales",
        feature_names=["lag_1"],
        configuration={"n_estimators": 10},
        features_id="features_x",
        serialization_format="pickle",
    )
    meta["artifact_version"] = 999
    with pytest.raises(ArtifactError, match="version"):
        check_compatibility(
            meta,
            domain="retail",
            model="random_forest",
            target="sales",
            features_id="features_x",
            configuration={"n_estimators": 10},
        )


def test_missing_artifact_returns_none(artifacts_root: Path) -> None:
    loaded = try_load_compatible_artifact(
        domain="retail",
        model_key="random_forest",
        target="sales",
        features_id="features_missing",
        configuration={"n_estimators": 10, "max_depth": None, "random_state": 42},
        artifacts_root=artifacts_root,
    )
    assert loaded is None


def test_corrupted_artifact_fallback(retail_features_dir: Path, artifacts_root: Path) -> None:
    config = expected_configuration_for_request(model_key="random_forest", domain="retail", n_estimators=12)
    result = run_training_pipeline(
        domain="retail",
        features_dir=retail_features_dir,
        model_name="random_forest",
        n_estimators=12,
        save_artifact=True,
        artifacts_root=artifacts_root,
    )
    pickle_path = Path(result["artifact_path"]) / "model.pkl"
    pickle_path.write_bytes(b"not-a-valid-pickle")

    loaded = try_load_compatible_artifact(
        domain="retail",
        model_key="random_forest",
        target="sales",
        features_id=derive_features_id(retail_features_dir),
        configuration=config,
        artifacts_root=artifacts_root,
    )
    assert loaded is None


def test_prediction_consistency_train_vs_load(retail_features_dir: Path, artifacts_root: Path) -> None:
    config = expected_configuration_for_request(model_key="random_forest", domain="retail", n_estimators=30)
    trained = run_training_pipeline(
        domain="retail",
        features_dir=retail_features_dir,
        model_name="random_forest",
        n_estimators=30,
        save_artifact=True,
        artifacts_root=artifacts_root,
        prediction_limit=50,
    )
    loaded = load_model_artifact(
        domain="retail",
        model_key="random_forest",
        target="sales",
        features_id=derive_features_id(retail_features_dir),
        configuration=config,
        artifacts_root=artifacts_root,
    )
    reused = evaluate_fitted_model_on_splits(
        domain="retail",
        model_key="random_forest",
        model=loaded["model"],
        features_dir=retail_features_dir,
        configuration=config,
        prediction_limit=50,
        feature_names=loaded["metadata"]["feature_names"],
    )

    assert reused["test_metrics"]["rmse"] == pytest.approx(trained["test_metrics"]["rmse"], rel=1e-9, abs=1e-9)
    assert reused["test_metrics"]["mae"] == pytest.approx(trained["test_metrics"]["mae"], rel=1e-9, abs=1e-9)
    assert len(reused["predictions"]) == len(trained["predictions"])
    for left, right in zip(reused["predictions"], trained["predictions"]):
        assert left["timestamp"] == right["timestamp"]
        assert left["predicted"] == pytest.approx(right["predicted"], rel=1e-9, abs=1e-9)
        assert left["actual"] == pytest.approx(right["actual"], rel=1e-9, abs=1e-9)


def test_config_fingerprint_stable() -> None:
    a = {"n_estimators": 200, "max_depth": None, "random_state": 42}
    b = {"random_state": 42, "max_depth": None, "n_estimators": 200}
    assert config_fingerprint(a) == config_fingerprint(b)
