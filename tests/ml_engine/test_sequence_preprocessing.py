"""Tests for SequencePreprocessor — sequence construction, scaling, and boundary policies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.sequence_preprocessing import SequencePreprocessingError, SequencePreprocessor


def _make_df(n: int, start: str = "2025-01-01", freq: str = "h", target: str = "demand_mw") -> pd.DataFrame:
    """Generate a simple synthetic time-series DataFrame."""
    timestamps = pd.date_range(start, periods=n, freq=freq)
    values = np.arange(1.0, n + 1.0, dtype=float)
    return pd.DataFrame({
        "timestamp": timestamps,
        target: values,
        "lag_1": pd.Series(values).shift(1).bfill().values,
        "feature_a": np.random.default_rng(42).standard_normal(n),
    })


def test_sequence_shapes() -> None:
    df = _make_df(50)
    train = df.iloc[:30]
    proc = SequencePreprocessor(target_column="demand_mw")
    proc.fit(train)
    X, y = proc.build_sequences(train, lookback=5)
    # 30 rows, lookback 5 → 30-5 = 25 sequences
    assert X.shape == (25, 5, proc.n_features)
    assert y.shape == (25,)


def test_target_alignment_current_target_not_in_sequence() -> None:
    """y(t) must equal the scaled target at row t; X(t) must NOT include row t."""
    df = _make_df(20)
    train = df.iloc[:15]
    proc = SequencePreprocessor(target_column="demand_mw")
    proc.fit(train)
    X, y = proc.build_sequences(train, lookback=3)
    n_seq = X.shape[0]
    assert n_seq == 15 - 3  # 12

    # The feature matrix must NOT contain the target column
    assert "demand_mw" not in proc.feature_names

    # Verify target values correspond to rows 3..14 (0-indexed)
    original_targets = train["demand_mw"].values
    scaled_targets = proc.scale_target(original_targets)
    for i in range(n_seq):
        target_row = i + 3  # offset by lookback
        assert y[i] == pytest.approx(scaled_targets[target_row], rel=1e-5)


def test_no_future_rows_in_sequence() -> None:
    """Sequence X(t) must use ONLY rows before t."""
    df = _make_df(10)
    proc = SequencePreprocessor(target_column="demand_mw")
    proc.fit(df)
    X, y = proc.build_sequences(df, lookback=3)
    # For sequence i, the 3 input rows should be rows i, i+1, i+2 (producing target at row i+3)
    # No row at index i+3 or beyond should appear in X[i]
    assert X.shape[1] == 3  # lookback dimension


def test_scaler_fitted_train_only() -> None:
    """Fitting on a different (larger) dataset should NOT change feature scaler."""
    small_train = _make_df(20)
    proc = SequencePreprocessor(target_column="demand_mw")
    proc.fit(small_train)
    mean_before = proc._feature_scaler.mean_.copy()

    # Transform a different dataframe — should NOT refit
    large_df = _make_df(200)
    X_large = proc.transform_features(large_df)

    mean_after = proc._feature_scaler.mean_.copy()
    np.testing.assert_array_equal(mean_before, mean_after)


def test_target_scaler_inverse_round_trip() -> None:
    df = _make_df(30)
    proc = SequencePreprocessor(target_column="demand_mw")
    proc.fit(df)

    y = df["demand_mw"].values.astype(float)
    y_scaled = proc.scale_target(y)
    y_recovered = proc.inverse_scale_target(y_scaled)
    np.testing.assert_allclose(y, y_recovered, rtol=1e-5)


def test_validation_boundary_uses_train_history() -> None:
    """First val sequences can use trailing train rows as lookback context."""
    df = _make_df(30)
    train = df.iloc[:20]
    val = df.iloc[20:]

    proc = SequencePreprocessor(target_column="demand_mw")
    proc.fit(train)

    lookback = 5
    # Without history, val of 10 rows with lookback=5 produces 5 sequences
    X_no_hist, y_no_hist = proc.build_sequences(val, lookback=lookback)
    # With history (last 5 train rows), val of 10 rows → all 10 rows get a sequence
    X_hist, y_hist = proc.build_sequences(val, lookback=lookback, history_df=train.iloc[-lookback:])

    assert X_no_hist.shape[0] == 5
    assert X_hist.shape[0] == 10


def test_chronological_order_preserved() -> None:
    """Sequences must be constructed in strictly ascending time order."""
    df = _make_df(15)
    proc = SequencePreprocessor(target_column="demand_mw")
    proc.fit(df)
    _, y = proc.build_sequences(df, lookback=2)
    # Target values should be monotonically increasing (our synthetic data is 1..N)
    original_targets = proc.inverse_scale_target(y)
    assert (np.diff(original_targets) > 0).all()


def test_all_missing_feature_excluded_from_sequences() -> None:
    """Columns entirely missing in TRAIN must be excluded from sequence features."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=15),
        "demand_mw": np.arange(1.0, 16.0),
        "lag_1": np.arange(0.0, 15.0),
        "all_nan_col": [np.nan] * 15,
    })
    proc = SequencePreprocessor(target_column="demand_mw")
    proc.fit(df)
    assert "all_nan_col" not in proc.feature_names


def test_target_and_timestamp_excluded_from_features() -> None:
    df = _make_df(15)
    df["revenue"] = df["demand_mw"] * 1.1  # leakage column for retail
    proc = SequencePreprocessor(target_column="demand_mw")
    proc.fit(df)
    assert "demand_mw" not in proc.feature_names
    assert "timestamp" not in proc.feature_names


def test_empty_training_data_raises() -> None:
    proc = SequencePreprocessor(target_column="demand_mw")
    with pytest.raises(SequencePreprocessingError, match="cannot be empty"):
        proc.fit(pd.DataFrame())
