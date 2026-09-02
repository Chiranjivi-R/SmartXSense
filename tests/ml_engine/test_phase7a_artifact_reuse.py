"""Regression and functional tests for Phase 7A model artifact registry, dataset fingerprinting, and reuse."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.artifact_store import (
    compute_dataset_fingerprint,
    derive_features_id,
    load_model_artifact,
    save_model_artifact,
    try_load_compatible_artifact,
)
from ml_engine.models.trainer import (
    expected_configuration_for_request,
    run_training_pipeline,
)


def _write_split_data(features_dir: Path, *, n: int = 80, domain: str = "retail", seed: int = 42) -> Path:
    features_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
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
def sample_retail_dir(tmp_path: Path) -> Path:
    return _write_split_data(tmp_path / "features_retail_7a_a", domain="retail", seed=42)


@pytest.fixture()
def sample_retail_dir_b(tmp_path: Path) -> Path:
    return _write_split_data(tmp_path / "features_retail_7a_b", domain="retail", seed=99)


@pytest.fixture()
def artifacts_dir(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts" / "models"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_A_first_forecast_trains_and_saves_artifact(sample_retail_dir: Path, artifacts_dir: Path) -> None:
    res = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    assert res["artifact_persistence"]["saved"] is True
    assert res["artifact_persistence"]["reused"] is False
    assert Path(res["artifact_path"]).exists()


def test_B_second_forecast_reuses_artifact(sample_retail_dir: Path, artifacts_dir: Path) -> None:
    res1 = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    assert res1["artifact_persistence"]["reused"] is False

    res2 = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    assert res2["artifact_persistence"]["reused"] is True
    assert res2["test_metrics"]["rmse"] == pytest.approx(res1["test_metrics"]["rmse"], rel=1e-9)


def test_C_force_retrain_bypasses_artifact(sample_retail_dir: Path, artifacts_dir: Path) -> None:
    run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    res_forced = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        force_retrain=True,
        artifacts_root=artifacts_dir,
    )
    assert res_forced["artifact_persistence"]["reused"] is False


def test_D_modified_dataset_fingerprint_invalidates_artifact(sample_retail_dir: Path, artifacts_dir: Path) -> None:
    run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    # Modify train.parquet in place
    df = pd.read_parquet(sample_retail_dir / "train.parquet")
    df["sales"] = df["sales"] * 10
    df.to_parquet(sample_retail_dir / "train.parquet", index=False)

    res_modified = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    assert res_modified["artifact_persistence"]["reused"] is False


def test_E_different_dataset_does_not_reuse_artifact(sample_retail_dir: Path, sample_retail_dir_b: Path, artifacts_dir: Path) -> None:
    run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    res_b = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir_b,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    assert res_b["artifact_persistence"]["reused"] is False


def test_F_different_model_configuration_does_not_reuse(sample_retail_dir: Path, artifacts_dir: Path) -> None:
    run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    res_diff_config = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=25,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    assert res_diff_config["artifact_persistence"]["reused"] is False


def test_G_corrupt_artifact_triggers_safe_retraining(sample_retail_dir: Path, artifacts_dir: Path) -> None:
    res = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    pkl_file = Path(res["artifact_path"]) / "model.pkl"
    pkl_file.write_bytes(b"corrupt-bytes-data")

    res_recovery = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    assert res_recovery["artifact_persistence"]["reused"] is False
    assert "test_metrics" in res_recovery


def test_H_existing_api_response_remains_valid(sample_retail_dir: Path, artifacts_dir: Path) -> None:
    res = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    required_keys = {
        "model", "domain", "target", "configuration", "validation_metrics",
        "test_metrics", "predictions", "prediction_count", "total_test_predictions"
    }
    assert required_keys.issubset(res.keys())


def test_I_hybrid_reuse_works(sample_retail_dir: Path, artifacts_dir: Path) -> None:
    res1 = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="hybrid",
        components=["random_forest", "xgboost"],
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    assert res1["model"] == "hybrid"

    # Verify component artifact reuse for hybrid
    rf_artifact = try_load_compatible_artifact(
        domain="retail",
        model_key="random_forest",
        target="sales",
        features_id=derive_features_id(sample_retail_dir),
        dataset_fingerprint=compute_dataset_fingerprint(sample_retail_dir),
        configuration=expected_configuration_for_request(model_key="random_forest", domain="retail", n_estimators=10),
        artifacts_root=artifacts_dir,
    )
    assert rf_artifact is not None

    res2 = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="hybrid",
        components=["random_forest", "xgboost"],
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    assert res2["model"] == "hybrid"
    assert res2["validation_metrics"]["rmse"] == pytest.approx(res1["validation_metrics"]["rmse"], rel=1e-9)


def test_J_K_auto_select_validation_only_selection(sample_retail_dir: Path) -> None:
    from ml_engine.evaluation import AdaptiveModelSelector, compare_models
    rep1 = run_training_pipeline(domain="retail", features_dir=sample_retail_dir, model_name="random_forest", n_estimators=10)
    rep2 = run_training_pipeline(domain="retail", features_dir=sample_retail_dir, model_name="xgboost", n_estimators=10)

    comp = compare_models([rep1, rep2], selection_metric="rmse")
    assert comp["selection_split"] == "validation"
    assert comp["recommended_model"] in {"random_forest", "xgboost"}

    selector = AdaptiveModelSelector(selection_metric="rmse")
    sel = selector.select_best_model([rep1, rep2])
    assert sel["selection_split"] == "validation"


def test_L_benchmark_cache_saving_and_loading(sample_retail_dir: Path, artifacts_dir: Path) -> None:
    from ml_engine.models.artifact_store import (
        benchmark_cache_path,
        save_benchmark_cache,
        try_load_benchmark_cache,
    )
    fid = derive_features_id(sample_retail_dir)
    fp = compute_dataset_fingerprint(sample_retail_dir)

    # Initial check: missing cache returns None
    loaded = try_load_benchmark_cache(
        domain="retail",
        features_id=fid,
        dataset_fingerprint=fp,
        candidate_models=["random_forest", "xgboost"],
        selection_metric="rmse",
        artifacts_root=artifacts_dir,
    )
    assert loaded is None

    # Save cache
    save_benchmark_cache(
        domain="retail",
        features_id=fid,
        dataset_fingerprint=fp,
        candidate_models=["random_forest", "xgboost"],
        selection_metric="rmse",
        ranking=[{"rank": 1, "model": "random_forest"}],
        recommended_model="random_forest",
        adaptive_selection={"selected_model": "random_forest"},
        artifacts_root=artifacts_dir,
    )

    # Subsequent check: hits cache
    loaded = try_load_benchmark_cache(
        domain="retail",
        features_id=fid,
        dataset_fingerprint=fp,
        candidate_models=["random_forest", "xgboost"],
        selection_metric="rmse",
        artifacts_root=artifacts_dir,
    )
    assert loaded is not None
    assert loaded["recommended_model"] == "random_forest"

    # Dataset fingerprint change -> misses cache
    loaded_diff_fp = try_load_benchmark_cache(
        domain="retail",
        features_id=fid,
        dataset_fingerprint="invalid_fp_123",
        candidate_models=["random_forest", "xgboost"],
        selection_metric="rmse",
        artifacts_root=artifacts_dir,
    )
    assert loaded_diff_fp is None

    # Corrupt cache file -> gracefully returns None
    key = loaded["cached_at"]  # get path
    cpath = benchmark_cache_path(
        domain="retail",
        key=list(artifacts_dir.glob("retail/benchmarks/benchmark_*.json"))[0].stem.replace("benchmark_", ""),
        artifacts_root=artifacts_dir,
    )
    cpath.write_text("corrupt json data", encoding="utf-8")

    loaded_corrupt = try_load_benchmark_cache(
        domain="retail",
        features_id=fid,
        dataset_fingerprint=fp,
        candidate_models=["random_forest", "xgboost"],
        selection_metric="rmse",
        artifacts_root=artifacts_dir,
    )
    assert loaded_corrupt is None


def test_different_dataset_contents_produce_different_artifact_keys_and_no_reuse(
    sample_retail_dir: Path, sample_retail_dir_b: Path, artifacts_dir: Path
) -> None:
    """Requirement 12 B & C: Dataset A vs Dataset B isolation.
    Different dataset contents must produce different content hashes/fingerprints and MUST NOT reuse artifacts.
    """
    res_a = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    assert res_a["artifact_persistence"]["saved"] is True
    assert res_a["artifact_persistence"]["reused"] is False

    # Train on dataset B with identical model configuration
    res_b = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir_b,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )

    # Dataset B MUST NOT reuse artifact from dataset A
    assert res_b["artifact_persistence"]["reused"] is False
    assert res_a["artifact_key"] != res_b["artifact_key"]

    # Re-running dataset A MUST reuse artifact A
    res_a_repeat = run_training_pipeline(
        domain="retail",
        features_dir=sample_retail_dir,
        model_name="random_forest",
        n_estimators=10,
        save_artifact=True,
        artifacts_root=artifacts_dir,
    )
    assert res_a_repeat["artifact_persistence"]["reused"] is True
    assert res_a_repeat["artifact_key"] == res_a["artifact_key"]
