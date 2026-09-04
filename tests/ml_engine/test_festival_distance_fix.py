"""Regression tests for festival_distance correctness hardening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_engine.features.artifacts import materialize_forecasting_features
from ml_engine.features.common import add_festival_interaction_features
from ml_engine.models.artifact_store import compute_dataset_fingerprint, try_load_compatible_artifact
from ml_engine.models.tabular_preprocessing import TabularFeatureTransformer


def test_festival_distance_signed_continuous_semantics() -> None:
    """Verify signed continuous distance semantics (-3..+3) and absence of 99.0 sentinel."""
    dates = pd.date_range("2026-01-01", periods=15, freq="D", tz="UTC")
    # Festival date on 2026-01-08 (index 7)
    fest_flags = [0.0] * 15
    fest_flags[7] = 1.0

    df = pd.DataFrame({
        "timestamp": dates,
        "sales": [100.0] * 15,
        "festival_flag": fest_flags,
    })

    res, names = add_festival_interaction_features(df)
    assert "festival_distance" in names
    distances = res["festival_distance"].tolist()

    # 1. No 99.0 sentinel present in the series
    assert 99.0 not in distances

    # 2. Date immediately before festival (2026-01-07, index 6) has distance -1.0
    assert distances[6] == -1.0
    assert distances[5] == -2.0
    assert distances[4] == -3.0

    # 3. Festival date (2026-01-08, index 7) has distance 0.0
    assert distances[7] == 0.0

    # 4. Date immediately after festival (2026-01-09, index 8) has distance +1.0
    assert distances[8] == 1.0
    assert distances[9] == 2.0
    assert distances[10] == 3.0

    # 5. Non-festival dates outside the +-3 window receive bounded saturation (-4.0 / +4.0)
    assert distances[0] == -4.0  # 7 days before festival -> capped at -4.0
    assert distances[14] == 4.0  # 6 days after festival -> capped at +4.0


def test_festival_distance_scaling_and_preprocessing_safety() -> None:
    """Verify festival_distance is finite, bounded, and scales safely via TabularFeatureTransformer."""
    dates = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    fest_flags = [1.0 if i in (5, 20) else 0.0 for i in range(30)]

    df = pd.DataFrame({
        "timestamp": dates,
        "sales": np.arange(100.0, 130.0),
        "festival_flag": fest_flags,
        "lag_1": np.arange(99.0, 129.0),
    })

    res, _ = add_festival_interaction_features(df)
    transformer = TabularFeatureTransformer(target_column="sales")
    transformer.fit(res)
    X_trans = transformer.transform(res)

    assert not np.isnan(X_trans).any()
    assert not np.isinf(X_trans).any()


def test_existing_festival_interaction_features_unimpaired() -> None:
    """Verify existing interaction features (fest_x_lag_1, fest_x_rolling_mean_7) continue to behave correctly."""
    dates = pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC")
    fest_flags = [1.0 if i == 10 else 0.0 for i in range(20)]

    df = pd.DataFrame({
        "timestamp": dates,
        "sales": [100.0] * 20,
        "festival_flag": fest_flags,
        "lag_1": [50.0] * 20,
        "rolling_mean_7": [60.0] * 20,
    })

    res, names = add_festival_interaction_features(df)

    assert "fest_x_lag_1" in names
    assert "fest_x_rolling_mean_7" in names
    assert res.loc[10, "fest_x_lag_1"] == 50.0
    assert res.loc[9, "fest_x_lag_1"] == 0.0
    assert res.loc[10, "fest_x_rolling_mean_7"] == 60.0
    assert res.loc[9, "fest_x_rolling_mean_7"] == 0.0


def test_no_target_leakage_in_festival_distance() -> None:
    """Verify festival_distance depends strictly on timestamp and calendar festival flag, never target values."""
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    fest_flags = [0.0] * 10
    fest_flags[5] = 1.0

    df1 = pd.DataFrame({"timestamp": dates, "sales": [10.0] * 10, "festival_flag": fest_flags})
    df2 = pd.DataFrame({"timestamp": dates, "sales": [9999.0] * 10, "festival_flag": fest_flags})

    res1, _ = add_festival_interaction_features(df1)
    res2, _ = add_festival_interaction_features(df2)

    pd.testing.assert_series_equal(res1["festival_distance"], res2["festival_distance"])


def test_feature_regeneration_changes_dataset_fingerprint(tmp_path: Path) -> None:
    """Verify that dataset fingerprint changes when festival_distance values change from legacy 99.0 sentinel to corrected representation."""
    dates = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "sales": np.arange(100.0, 140.0),
        "festival_flag": [1.0 if i == 10 else 0.0 for i in range(40)],
    })

    prep = materialize_forecasting_features(df, "retail", output_root=tmp_path / "corrected_root")
    corrected_dir = prep["features_dir"]
    fp_corrected = compute_dataset_fingerprint(corrected_dir)

    # Build a legacy split directory with matching folder structure but carrying legacy 99.0 sentinels
    legacy_dir = tmp_path / "legacy_root" / corrected_dir.name
    legacy_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("train.parquet", "validation.parquet", "test.parquet"):
        split_df = pd.read_parquet(corrected_dir / fname).copy()
        if "festival_distance" in split_df.columns:
            mask = split_df["festival_distance"].abs() == 4.0
            split_df.loc[mask, "festival_distance"] = 99.0
        split_df.to_parquet(legacy_dir / fname, index=False)

    fp_legacy = compute_dataset_fingerprint(legacy_dir)

    # Prove that dataset fingerprints genuinely differ between corrected and legacy sentinel feature outputs
    assert fp_corrected != fp_legacy
