"""Tests for CNNLSTMForecaster — architecture, configuration, training, leakage, and trainer integration.

Design principles:
- Tiny synthetic data to keep tests fast (1–2 epochs, small networks)
- No duplication of SequencePreprocessor tests (those live in test_sequence_preprocessing.py)
- Verify CNN-LSTM-specific concerns: Conv1d shape/safety, kernel validation, n_training_sequences
- Verify backward compatibility of all existing models
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.cnn_lstm import CNNLSTMForecaster, CNNLSTMModelError
from ml_engine.models.trainer import TrainerError, run_training_pipeline


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_synth_df(
    n: int,
    target: str = "demand_mw",
    start: str = "2025-01-01",
    freq: str = "h",
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=n, freq=freq)
    values = 100.0 + np.cumsum(rng.standard_normal(n))
    lag1 = pd.Series(values).shift(1).bfill().values
    return pd.DataFrame({
        "timestamp": timestamps,
        target: values,
        "lag_1": lag1,
        "feature_a": rng.standard_normal(n),
    })


@pytest.fixture
def tiny_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _make_synth_df(60)
    train = df.iloc[:40].reset_index(drop=True)
    val = df.iloc[40:50].reset_index(drop=True)
    test = df.iloc[50:].reset_index(drop=True)
    return train, val, test


@pytest.fixture
def electricity_features_dir(tmp_path: Path) -> Path:
    out = tmp_path / "electricity" / "features_test"
    out.mkdir(parents=True, exist_ok=True)
    df = _make_synth_df(80, target="demand_mw")
    df.iloc[:50].to_parquet(out / "train.parquet", index=False)
    df.iloc[50:65].to_parquet(out / "validation.parquet", index=False)
    df.iloc[65:].to_parquet(out / "test.parquet", index=False)
    return out


@pytest.fixture
def retail_features_dir(tmp_path: Path) -> Path:
    out = tmp_path / "retail" / "features_test"
    out.mkdir(parents=True, exist_ok=True)
    df = _make_synth_df(60, target="sales", freq="D")
    df.iloc[:40].to_parquet(out / "train.parquet", index=False)
    df.iloc[40:50].to_parquet(out / "validation.parquet", index=False)
    df.iloc[50:].to_parquet(out / "test.parquet", index=False)
    return out


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def test_kernel_size_larger_than_lookback_raises() -> None:
    """kernel_size > lookback must fail immediately at construction."""
    with pytest.raises(CNNLSTMModelError, match="kernel_size"):
        CNNLSTMForecaster(lookback=5, kernel_size=6)


def test_kernel_size_zero_raises() -> None:
    with pytest.raises(CNNLSTMModelError, match="kernel_size"):
        CNNLSTMForecaster(lookback=5, kernel_size=0)


def test_kernel_size_equals_lookback_is_valid() -> None:
    """kernel_size == lookback is valid; temporal dim after conv = 1."""
    model = CNNLSTMForecaster(lookback=5, kernel_size=5, cnn_filters=4, lstm_units=4, epochs=1)
    assert model.kernel_size == 5


def test_cnn_filters_configurable(tiny_splits: tuple) -> None:
    train, _, _ = tiny_splits
    model = CNNLSTMForecaster(lookback=3, cnn_filters=8, kernel_size=2, lstm_units=4, dropout=0.0, epochs=1, batch_size=8)
    model.fit(train, target_column="demand_mw")
    assert model.cnn_filters == 8


def test_lstm_units_configurable(tiny_splits: tuple) -> None:
    train, _, _ = tiny_splits
    model = CNNLSTMForecaster(lookback=3, cnn_filters=4, kernel_size=2, lstm_units=16, dropout=0.0, epochs=1, batch_size=8)
    model.fit(train, target_column="demand_mw")
    assert model.lstm_units == 16


# ---------------------------------------------------------------------------
# Model shape / architecture
# ---------------------------------------------------------------------------

def test_model_builds_and_predicts(tiny_splits: tuple) -> None:
    train, val, _ = tiny_splits
    model = CNNLSTMForecaster(lookback=5, cnn_filters=8, kernel_size=3, lstm_units=8, dropout=0.0, epochs=1, batch_size=16, random_seed=42)
    model.fit(train, target_column="demand_mw", val_df=val)
    preds = model.predict(val, target_column="demand_mw", history_df=train)
    assert len(preds) == len(val)
    assert not preds.isna().any()


def test_prediction_count_val_equals_val_rows(tiny_splits: tuple) -> None:
    """With sufficient history, validation prediction count == val rows."""
    train, val, test = tiny_splits
    model = CNNLSTMForecaster(lookback=5, cnn_filters=4, kernel_size=2, lstm_units=4, dropout=0.0, epochs=1, batch_size=8, random_seed=42)
    model.fit(train, target_column="demand_mw", val_df=val)

    val_preds = model.predict(val, target_column="demand_mw", history_df=train)
    test_preds = model.predict(test, target_column="demand_mw", history_df=pd.concat([train, val], ignore_index=True))

    assert len(val_preds) == len(val)
    assert len(test_preds) == len(test)


def test_predictions_in_original_units(tiny_splits: tuple) -> None:
    """Inverse-scaled predictions must be in original target units (~100), not near 0 (scaled)."""
    train, val, _ = tiny_splits
    model = CNNLSTMForecaster(lookback=3, cnn_filters=4, kernel_size=2, lstm_units=4, dropout=0.0, epochs=1, batch_size=8, random_seed=42)
    model.fit(train, target_column="demand_mw")
    preds = model.predict(val, target_column="demand_mw", history_df=train)
    # Scaled values cluster around 0±2; original values cluster around 100
    assert preds.abs().mean() > 1.0


# ---------------------------------------------------------------------------
# n_training_sequences — regression guard (Phase 5F bug fix preserved)
# ---------------------------------------------------------------------------

def test_n_training_sequences_equals_train_minus_lookback() -> None:
    """n_training_sequences must equal len(X_train) = train_rows - lookback.

    Will FAIL if someone assigns n_training_sequences = len(val_pred).
    """
    train = _make_synth_df(20)
    model = CNNLSTMForecaster(lookback=5, cnn_filters=4, kernel_size=2, lstm_units=4, dropout=0.0, epochs=1, batch_size=8, random_seed=42)
    model.fit(train, target_column="demand_mw")

    expected = 20 - 5  # 15
    assert model.n_training_sequences == expected, (
        f"Expected n_training_sequences=15, got {model.n_training_sequences}. "
        "Must be len(X_train) from fit(), not inferred from val_pred."
    )


def test_n_training_sequences_not_confused_with_val(tiny_splits: tuple) -> None:
    """n_training_sequences != n_validation rows when they differ."""
    train, val, _ = tiny_splits
    # train=40 rows, val=10 rows, lookback=5 => training_sequences=35, val pred count=10
    model = CNNLSTMForecaster(lookback=5, cnn_filters=4, kernel_size=2, lstm_units=4, dropout=0.0, epochs=1, batch_size=8, random_seed=42)
    model.fit(train, target_column="demand_mw", val_df=val)
    val_preds = model.predict(val, target_column="demand_mw", history_df=train)

    assert model.n_training_sequences == 35  # 40 - 5
    assert len(val_preds) == 10
    assert model.n_training_sequences != len(val_preds)


# ---------------------------------------------------------------------------
# epochs_trained
# ---------------------------------------------------------------------------

def test_epochs_trained_populated(tiny_splits: tuple) -> None:
    train, val, _ = tiny_splits
    model = CNNLSTMForecaster(lookback=3, cnn_filters=4, kernel_size=2, lstm_units=4, dropout=0.0, epochs=2, batch_size=8, random_seed=42)
    model.fit(train, target_column="demand_mw", val_df=val)
    assert model.epochs_trained >= 1


# ---------------------------------------------------------------------------
# Leakage / Pipeline guarantees
# ---------------------------------------------------------------------------

def test_feature_names_exclude_target_and_timestamp(tiny_splits: tuple) -> None:
    """target column and timestamp must never appear in the feature matrix."""
    train, _, _ = tiny_splits
    model = CNNLSTMForecaster(lookback=3, cnn_filters=4, kernel_size=2, lstm_units=4, dropout=0.0, epochs=1, batch_size=8, random_seed=42)
    model.fit(train, target_column="demand_mw")
    assert "demand_mw" not in model.feature_names
    assert "timestamp" not in model.feature_names


def test_all_missing_feature_excluded(tmp_path: Path) -> None:
    """Columns entirely NaN in TRAIN must be excluded from features (Phase 5E rule preserved)."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=20),
        "demand_mw": np.arange(1.0, 21.0),
        "lag_1": np.arange(0.0, 20.0),
        "all_nan_col": [np.nan] * 20,
    })
    model = CNNLSTMForecaster(lookback=3, cnn_filters=4, kernel_size=2, lstm_units=4, dropout=0.0, epochs=1, batch_size=4, random_seed=42)
    model.fit(df, target_column="demand_mw")
    assert "all_nan_col" not in model.feature_names


def test_unfitted_model_predict_raises() -> None:
    model = CNNLSTMForecaster()
    val = _make_synth_df(5)
    with pytest.raises(CNNLSTMModelError, match="must be fitted"):
        model.predict(val, target_column="demand_mw")


def test_empty_training_raises() -> None:
    model = CNNLSTMForecaster()
    with pytest.raises(CNNLSTMModelError, match="empty"):
        model.fit(pd.DataFrame(), target_column="demand_mw")


def test_missing_target_column_raises() -> None:
    df = _make_synth_df(20)
    model = CNNLSTMForecaster(lookback=3, cnn_filters=4, kernel_size=2, lstm_units=4, epochs=1)
    with pytest.raises(CNNLSTMModelError, match="missing from training"):
        model.fit(df, target_column="nonexistent")


# ---------------------------------------------------------------------------
# Trainer integration
# ---------------------------------------------------------------------------

def test_trainer_cnn_lstm_electricity(electricity_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="electricity",
        features_dir=electricity_features_dir,
        model_name="cnn_lstm",
        lookback=5,
        cnn_filters=8,
        kernel_size=3,
        lstm_units=8,
        dropout=0.0,
        epochs=1,
        batch_size=16,
        lstm_learning_rate=0.001,
        random_state=42,
    )
    assert result["model"] == "cnn_lstm"
    assert result["domain"] == "electricity"
    assert result["target"] == "demand_mw"
    cfg = result["configuration"]
    assert cfg["lookback"] == 5
    assert cfg["cnn_filters"] == 8
    assert cfg["kernel_size"] == 3
    assert result["epochs_trained"] >= 1
    assert set(result["validation_metrics"].keys()) == {"mae", "rmse", "mape", "r2"}
    assert set(result["test_metrics"].keys()) == {"mae", "rmse", "mape", "r2"}


def test_trainer_cnn_lstm_retail(retail_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="retail",
        features_dir=retail_features_dir,
        model_name="cnn_lstm",
        lookback=3,
        cnn_filters=4,
        kernel_size=2,
        lstm_units=4,
        dropout=0.0,
        epochs=1,
        batch_size=8,
        lstm_learning_rate=0.001,
        random_state=42,
    )
    assert result["model"] == "cnn_lstm"
    assert result["target"] == "sales"
    assert result["configuration"]["lookback"] == 3


def test_trainer_reports_correct_training_sequences(electricity_features_dir: Path) -> None:
    """Trainer JSON must report n_training_sequences = train_rows - lookback.

    Fixtures: train=50 rows, lookback=5 => 45 training sequences
              val=15 rows (with full lookback history => 15 val sequences)
    FAILS if training_sequences is set to len(val_pred)=15 instead of 45.
    """
    result = run_training_pipeline(
        domain="electricity",
        features_dir=electricity_features_dir,
        model_name="cnn_lstm",
        lookback=5,
        cnn_filters=4,
        kernel_size=3,
        lstm_units=4,
        dropout=0.0,
        epochs=1,
        batch_size=8,
        lstm_learning_rate=0.001,
        random_state=42,
    )
    train_rows = result["training_rows"]            # 50
    lookback = result["configuration"]["lookback"]  # 5
    val_sequences = result["validation_sequences"]  # 15

    expected = train_rows - lookback   # 45
    actual = result["training_sequences"]

    assert actual == expected, (
        f"training_sequences should be {expected} "
        f"(train_rows={train_rows} - lookback={lookback}), got {actual}. "
        f"validation_sequences={val_sequences} — must not equal training_sequences."
    )
    assert actual != val_sequences, (
        "training_sequences must not equal validation_sequences"
    )


def test_trainer_cnn_lstm_kernel_too_large_raises(electricity_features_dir: Path) -> None:
    """Trainer must propagate CNNLSTMModelError for invalid kernel_size."""
    with pytest.raises(CNNLSTMModelError, match="kernel_size"):
        run_training_pipeline(
            domain="electricity",
            features_dir=electricity_features_dir,
            model_name="cnn_lstm",
            lookback=5,
            kernel_size=10,  # > lookback
            epochs=1,
        )


# ---------------------------------------------------------------------------
# Backward compatibility: existing models must remain unaffected
# ---------------------------------------------------------------------------

def test_backward_compat_all_existing_models_still_work(electricity_features_dir: Path) -> None:
    for model_name in ("naive", "arima", "random_forest", "xgboost"):
        result = run_training_pipeline(
            domain="electricity",
            features_dir=electricity_features_dir,
            model_name=model_name,
            n_estimators=5,
        )
        assert result["model"] == model_name, f"Expected model={model_name}, got {result['model']}"


def test_backward_compat_lstm_still_works(electricity_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="electricity",
        features_dir=electricity_features_dir,
        model_name="lstm",
        lookback=5,
        lstm_units=4,
        dropout=0.0,
        epochs=1,
        batch_size=8,
        lstm_learning_rate=0.001,
        random_state=42,
    )
    assert result["model"] == "lstm"
    assert result["epochs_trained"] >= 1
