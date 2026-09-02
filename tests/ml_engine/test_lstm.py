"""Tests for LSTMForecaster model — build, fit, predict, and trainer integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_engine.models.lstm import LSTMForecaster, LSTMModelError
from ml_engine.models.trainer import TrainerError, run_training_pipeline


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


def test_lstm_model_builds_and_predicts(tiny_splits: tuple) -> None:
    train, val, test = tiny_splits
    model = LSTMForecaster(lookback=5, lstm_units=8, dropout=0.0, epochs=1, batch_size=16, random_seed=42)
    model.fit(train, target_column="demand_mw", val_df=val)
    val_preds = model.predict(val, target_column="demand_mw", history_df=train)
    assert len(val_preds) == len(val)
    assert not val_preds.isna().any()


def test_lstm_prediction_count_matches_eval(tiny_splits: tuple) -> None:
    train, val, test = tiny_splits
    model = LSTMForecaster(lookback=5, lstm_units=8, dropout=0.0, epochs=1, batch_size=16, random_seed=42)
    model.fit(train, target_column="demand_mw", val_df=val)

    val_preds = model.predict(val, target_column="demand_mw", history_df=train)
    test_preds = model.predict(test, target_column="demand_mw", history_df=pd.concat([train, val], ignore_index=True))

    assert len(val_preds) == len(val)
    assert len(test_preds) == len(test)


def test_lstm_predictions_in_original_units(tiny_splits: tuple) -> None:
    """Predictions must be inverse-transformed and in original target units."""
    train, val, _ = tiny_splits
    model = LSTMForecaster(lookback=3, lstm_units=4, dropout=0.0, epochs=1, batch_size=8, random_seed=42)
    model.fit(train, target_column="demand_mw")
    preds = model.predict(val, target_column="demand_mw", history_df=train)

    # Original target is around 100 ± some drift. Scaled target would be around 0±1.
    # If predictions were NOT inverse-transformed, values would be near 0.
    # So we check they are not all near 0.
    assert preds.abs().mean() > 1.0


def test_lstm_determinism(tiny_splits: tuple) -> None:
    train, val, _ = tiny_splits
    m1 = LSTMForecaster(lookback=3, lstm_units=4, dropout=0.0, epochs=2, batch_size=8, random_seed=42)
    m2 = LSTMForecaster(lookback=3, lstm_units=4, dropout=0.0, epochs=2, batch_size=8, random_seed=42)
    m1.fit(train, target_column="demand_mw")
    m2.fit(train, target_column="demand_mw")
    p1 = m1.predict(val, target_column="demand_mw", history_df=train)
    p2 = m2.predict(val, target_column="demand_mw", history_df=train)
    pd.testing.assert_series_equal(p1, p2)


def test_lstm_feature_names_exclude_target_and_timestamp(tiny_splits: tuple) -> None:
    train, _, _ = tiny_splits
    model = LSTMForecaster(lookback=3, lstm_units=4, dropout=0.0, epochs=1, batch_size=8, random_seed=42)
    model.fit(train, target_column="demand_mw")
    assert "demand_mw" not in model.feature_names
    assert "timestamp" not in model.feature_names


def test_lstm_unfitted_raises() -> None:
    model = LSTMForecaster()
    val = _make_synth_df(5)
    with pytest.raises(LSTMModelError, match="must be fitted"):
        model.predict(val, target_column="demand_mw")


def test_lstm_empty_training_raises() -> None:
    model = LSTMForecaster()
    with pytest.raises(LSTMModelError, match="empty"):
        model.fit(pd.DataFrame(), target_column="demand_mw")


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


def test_trainer_lstm_electricity(electricity_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="electricity",
        features_dir=electricity_features_dir,
        model_name="lstm",
        lookback=5,
        lstm_units=8,
        dropout=0.0,
        epochs=1,
        batch_size=16,
        lstm_learning_rate=0.001,
        random_state=42,
    )
    assert result["model"] == "lstm"
    assert result["domain"] == "electricity"
    assert result["target"] == "demand_mw"
    assert result["configuration"]["lookback"] == 5
    assert result["epochs_trained"] >= 1
    assert set(result["validation_metrics"].keys()) == {"mae", "rmse", "mape", "r2", "smape", "wape"}

    assert set(result["test_metrics"].keys()) == {"mae", "rmse", "mape", "r2", "smape", "wape"}



def test_trainer_lstm_retail(retail_features_dir: Path) -> None:
    result = run_training_pipeline(
        domain="retail",
        features_dir=retail_features_dir,
        model_name="lstm",
        lookback=3,
        lstm_units=4,
        dropout=0.0,
        epochs=1,
        batch_size=8,
        lstm_learning_rate=0.001,
        random_state=42,
    )
    assert result["model"] == "lstm"
    assert result["target"] == "sales"


def test_trainer_backward_compat_old_models_still_work(electricity_features_dir: Path) -> None:
    for model_name in ("naive", "arima", "random_forest"):
        result = run_training_pipeline(
            domain="electricity",
            features_dir=electricity_features_dir,
            model_name=model_name,
            n_estimators=5,
        )
        assert result["model"] == model_name


# ---------------------------------------------------------------------------
# Regression: training_sequences must be train_rows - lookback, never len(val_pred)
# ---------------------------------------------------------------------------

def test_n_training_sequences_attribute_set_from_fit() -> None:
    """LSTMForecaster.n_training_sequences must equal len(X_train) from fit.

    With train rows=20 and lookback=5, there are 15 valid sequences
    (indices 5..19 produce targets; no history prepended).
    This test will FAIL if someone later reassigns n_training_sequences = len(val_pred).
    """
    train = _make_synth_df(20)
    model = LSTMForecaster(lookback=5, lstm_units=4, dropout=0.0, epochs=1, batch_size=8, random_seed=42)
    model.fit(train, target_column="demand_mw")

    expected = 20 - 5  # = 15
    assert model.n_training_sequences == expected, (
        f"Expected n_training_sequences={expected}, got {model.n_training_sequences}. "
        "n_training_sequences must be len(X_train) from fit(), not inferred from val_pred."
    )


def test_trainer_reports_correct_training_sequences(electricity_features_dir: Path) -> None:
    """Trainer JSON report must carry the true training sequence count.

    Fixtures: train=50 rows, lookback=5 → training_sequences = 45
              val=15 rows (with full lookback history → 15 val sequences)

    The test FAILS if training_sequences is accidentally set to len(val_pred)=15
    instead of the correct 45.
    """
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
    train_rows = result["training_rows"]       # 50
    lookback = result["configuration"]["lookback"]  # 5
    val_sequences = result["validation_sequences"]  # 15

    expected_training_sequences = train_rows - lookback  # 45
    actual_training_sequences = result["training_sequences"]

    assert actual_training_sequences == expected_training_sequences, (
        f"training_sequences should be {expected_training_sequences} "
        f"(train_rows={train_rows} - lookback={lookback}), "
        f"but got {actual_training_sequences}. "
        f"validation_sequences={val_sequences} — must NOT equal training_sequences."
    )
    # Confirm val sequences are not confused with training sequences
    assert actual_training_sequences != val_sequences, (
        "training_sequences must not equal validation_sequences "
        "(they reflect different partition sizes)"
    )


def test_lstm_dynamic_feature_dimensions() -> None:
    """LSTM input size must dynamically adapt to any number of feature columns."""
    rng = np.random.default_rng(42)
    n_rows = 50
    timestamps = pd.date_range("2026-01-01", periods=n_rows, freq="h")

    # Test with 15 features (e.g. got 15, expected 15, not hardcoded 20)
    feats_15 = {f"feat_{i}": rng.standard_normal(n_rows) for i in range(15)}
    df15 = pd.DataFrame({"timestamp": timestamps, "sales": rng.standard_normal(n_rows), **feats_15})
    m15 = LSTMForecaster(lookback=5, lstm_units=8, epochs=1, batch_size=8, random_seed=42)
    m15.fit(df15.iloc[:35], target_column="sales", val_df=df15.iloc[35:45])
    preds15 = m15.predict(df15.iloc[45:], target_column="sales", history_df=df15.iloc[:45])
    assert len(preds15) == 5

    # Test with 25 features
    feats_25 = {f"feat_{i}": rng.standard_normal(n_rows) for i in range(25)}
    df25 = pd.DataFrame({"timestamp": timestamps, "sales": rng.standard_normal(n_rows), **feats_25})
    m25 = LSTMForecaster(lookback=5, lstm_units=8, epochs=1, batch_size=8, random_seed=42)
    m25.fit(df25.iloc[:35], target_column="sales", val_df=df25.iloc[35:45])
    preds25 = m25.predict(df25.iloc[45:], target_column="sales", history_df=df25.iloc[:45])
    assert len(preds25) == 5
