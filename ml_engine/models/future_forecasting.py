"""Future (out-of-sample) demand forecasting for SmartXSense.

Distinction from held-out TEST evaluation:
  - TEST evaluation returns timestamp + actual + predicted
  - FUTURE forecasting returns timestamp + predicted only (no actuals)

Exogenous feature policy (Retail tabular models)
------------------------------------------------
Electricity feature matrices contain only deterministic calendar fields and
target-history lag/rolling features. No exogenous carry-forward is required.

Retail matrices also contain slowly-changing economic, weather, pricing, and
entity descriptors that are not knowable for future dates unless supplied by
a user. Phase B does not accept arbitrary future exogenous payloads yet.

Documented policy for Retail Random Forest / XGBoost short-horizon forecasts:
  * Deterministic calendar fields are recomputed from each future timestamp.
  * Target-derived lag/rolling features are computed recursively from
    historical actual targets and previously generated future predictions.
  * Remaining exogenous / entity columns use last-observation-carried-forward
    (LOCF) from the final historical feature row.

LOCF is intentional, tested, and disclosed in API metadata under
``exogenous_policy``. It is NOT used for Electricity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_engine.models.arima import ARIMAForecaster
from ml_engine.models.trainer import DOMAIN_TARGET_MAP
from ml_engine.preprocessing.calendar_features import add_region_calendar_features


class FutureForecastError(ValueError):
    """Raised when future forecasting cannot be executed safely."""


ARTIFACT_VERSION_NOTE = 1
# 90 days of hourly electricity demand.  Tabular recursion remains bounded so
# a request cannot accidentally turn into an unbounded browser/API workload.
MAX_HORIZON = 2160

DOMAIN_FREQUENCY = {
    "retail": "D",
    "electricity": "h",
}

DOMAIN_LAGS = {
    "retail": (1, 7, 14, 28),
    "electricity": (1, 24, 48, 168),
}

DOMAIN_ROLLING_WINDOWS = {
    "retail": (7, 14, 28),
    "electricity": (24, 168),
}

DETERMINISTIC_CALENDAR_FEATURES = {
    "retail": (
        "year",
        "month",
        "day",
        "day_of_week",
        "quarter",
        "is_weekend",
        "week_of_month",
    ),
    "electricity": (
        "year",
        "month",
        "day",
        "day_of_week",
        "quarter",
        "is_weekend",
        "hour",
    ),
}

# Explicit capability matrix: only models listed here are enabled.
FUTURE_SUPPORTED_MODELS: dict[str, frozenset[str]] = {
    "retail": frozenset({"random_forest", "xgboost"}),
    "electricity": frozenset({"random_forest", "xgboost", "arima"}),
}

UNSUPPORTED_FUTURE_REASONS: dict[str, str] = {
    "lstm": (
        "Future forecasting is not currently supported for model lstm. "
        "Sequence models require carefully constructed future lookback windows "
        "and are not enabled in Phase B."
    ),
    "cnn_lstm": (
        "Future forecasting is not currently supported for model cnn_lstm. "
        "Sequence models require carefully constructed future lookback windows "
        "and are not enabled in Phase B."
    ),
    "hybrid": (
        "Future forecasting is not currently supported for model hybrid. "
        "Hybrid reuse requires every component model to support legitimate "
        "future forecasts; component future support is incomplete."
    ),
    "naive": (
        "Future forecasting is not currently supported for model naive."
    ),
    "seasonal_naive": (
        "Future forecasting is not currently supported for model seasonal_naive."
    ),
    "baseline": (
        "Future forecasting is not currently supported for model baseline."
    ),
}


def is_future_forecast_supported(domain: str, model_key: str) -> bool:
    domain = domain.lower().strip()
    model_key = model_key.lower().strip()
    return model_key in FUTURE_SUPPORTED_MODELS.get(domain, frozenset())


def future_support_error(domain: str, model_key: str) -> str:
    domain = domain.lower().strip()
    model_key = model_key.lower().strip()
    if model_key == "arima" and domain == "retail":
        return "Future forecasting is not currently supported for model arima on the retail domain."
    if model_key in UNSUPPORTED_FUTURE_REASONS:
        return UNSUPPORTED_FUTURE_REASONS[model_key]
    if domain not in FUTURE_SUPPORTED_MODELS:
        return f"Future forecasting is not currently supported for domain '{domain}'."
    return f"Future forecasting is not currently supported for model {model_key}."


def validate_horizon(horizon: Any, *, domain: str) -> int:
    """Validate and return a positive integer horizon within domain-safe bounds."""
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        # Reject bool explicitly (bool is a subclass of int) and non-integers.
        raise FutureForecastError("Parameter 'horizon' must be a positive integer")
    if horizon <= 0:
        raise FutureForecastError("Parameter 'horizon' must be a positive integer")
    if horizon > MAX_HORIZON:
        raise FutureForecastError(
            f"Parameter 'horizon' exceeds maximum allowed value of {MAX_HORIZON}"
        )
    # Soft domain guidance: allow any 1..MAX but electricity often uses hourly
    # presets; retail daily presets. No extra hard domain max beyond MAX_HORIZON.
    _ = domain
    return horizon


def load_chronological_history(features_dir, target_column: str) -> pd.DataFrame:
    """Load train+validation+test feature rows in chronological order."""
    from pathlib import Path

    features_dir = Path(features_dir)
    frames = []
    for name in ("train.parquet", "validation.parquet", "test.parquet"):
        path = features_dir / name
        if not path.exists():
            raise FutureForecastError(f"Required feature artifact '{name}' is missing")
        frames.append(pd.read_parquet(path))
    history = pd.concat(frames, ignore_index=True)
    if "timestamp" not in history.columns or target_column not in history.columns:
        raise FutureForecastError("Historical feature data missing timestamp or target column")
    history = history.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    history = history.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable").reset_index(drop=True)
    if history.empty:
        raise FutureForecastError("Historical feature data is empty")
    return history


def _calendar_features_for_timestamp(ts: pd.Timestamp, domain: str, *, region: str = "other") -> dict[str, Any]:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    values: dict[str, Any] = {
        "year": int(ts.year),
        "month": int(ts.month),
        "day": int(ts.day),
        "day_of_week": int(ts.dayofweek),
        "quarter": int(ts.quarter),
        "is_weekend": int(ts.dayofweek >= 5),
    }
    if domain == "electricity":
        values["hour"] = int(ts.hour)
    if domain == "retail":
        values["week_of_month"] = int(((ts.day - 1) // 7) + 1)
    # Reuse the preprocessing calendar feature generator so future rows match
    # the selected dataset's regional calendar without a parallel pipeline.
    calendar_frame, _ = add_region_calendar_features(pd.DataFrame({"timestamp": [ts]}), region=region)
    values.update({key: value for key, value in calendar_frame.iloc[0].to_dict().items() if key != "timestamp"})
    values["calendar_region"] = region
    return values


def _lag_rolling_from_history(
    target_history: list[float],
    *,
    domain: str,
) -> dict[str, float]:
    lags = DOMAIN_LAGS[domain]
    windows = DOMAIN_ROLLING_WINDOWS[domain]
    values: dict[str, float] = {}
    n = len(target_history)

    for lag in lags:
        if n >= lag:
            values[f"lag_{lag}"] = float(target_history[-lag])
        else:
            values[f"lag_{lag}"] = float("nan")

    # Rolling stats use history shifted by one relative to the current step:
    # only past targets (already in target_history) are visible.
    for window in windows:
        if n >= window:
            window_vals = np.asarray(target_history[-window:], dtype=float)
            values[f"rolling_mean_{window}"] = float(np.mean(window_vals))
            values[f"rolling_std_{window}"] = float(np.std(window_vals, ddof=1)) if window > 1 else 0.0
        else:
            values[f"rolling_mean_{window}"] = float("nan")
            values[f"rolling_std_{window}"] = float("nan")
    return values


def build_future_feature_frame(
    history_df: pd.DataFrame,
    *,
    domain: str,
    target_column: str,
    horizon: int,
    predicted_targets: list[float] | None = None,
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Build future feature rows using recursive target history and LOCF exogenous.

    If ``predicted_targets`` is provided it must already contain ``horizon`` values
    (used only for offline reconstruction/tests). Otherwise the frame is built
    step-by-step with NaN placeholders for the unknown future target column;
    callers that need recursive prediction should use
    ``recursive_tabular_future_forecast`` instead.
    """
    domain = domain.lower().strip()
    if domain not in DOMAIN_FREQUENCY:
        raise FutureForecastError(f"Unsupported domain '{domain}'")

    last_ts = pd.Timestamp(history_df["timestamp"].iloc[-1])
    freq = DOMAIN_FREQUENCY[domain]
    future_timestamps = pd.date_range(last_ts + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq, tz="UTC")

    base = history_df.iloc[-1].to_dict()
    region = str(base.get("calendar_region") or "other").lower().strip()
    target_history = [float(v) for v in history_df[target_column].astype(float).tolist()]
    rows: list[dict[str, Any]] = []

    for step, ts in enumerate(future_timestamps):
        row = dict(base)
        row["timestamp"] = ts
        row.update(_calendar_features_for_timestamp(ts, domain, region=region))
        row.update(_lag_rolling_from_history(target_history, domain=domain))
        # Future actual target is unknown — never fabricate an actual value for metrics.
        if predicted_targets is not None:
            pred = float(predicted_targets[step])
            row[target_column] = pred
            target_history.append(pred)
        else:
            row[target_column] = np.nan
        rows.append(row)

    future_df = pd.DataFrame(rows)
    # Preserve column order from history where possible.
    ordered_cols = [c for c in history_df.columns if c in future_df.columns]
    extra = [c for c in future_df.columns if c not in ordered_cols]
    future_df = future_df[ordered_cols + extra]
    return future_df, list(future_timestamps)


def recursive_tabular_future_forecast(
    *,
    model: Any,
    history_df: pd.DataFrame,
    domain: str,
    target_column: str,
    horizon: int,
) -> list[dict[str, Any]]:
    """Recursive one-step-ahead future forecasts for RF / XGBoost."""
    domain = domain.lower().strip()
    last_ts = pd.Timestamp(history_df["timestamp"].iloc[-1])
    freq = DOMAIN_FREQUENCY[domain]
    offset = pd.tseries.frequencies.to_offset(freq)

    base = history_df.iloc[-1].to_dict()
    region = str(base.get("calendar_region") or "other").lower().strip()
    target_history = [float(v) for v in history_df[target_column].astype(float).tolist()]
    predictions: list[dict[str, Any]] = []

    # Ensure transformer is not refit: RandomForestForecaster.predict / XGBoost use transform only.
    transformer = getattr(model, "_transformer", None)
    if transformer is not None and not getattr(transformer, "_is_fitted", False):
        raise FutureForecastError("Model preprocessing transformer is not fitted")

    current_ts = last_ts
    for _ in range(horizon):
        current_ts = current_ts + offset
        if current_ts.tzinfo is None:
            current_ts = current_ts.tz_localize("UTC")
        else:
            current_ts = current_ts.tz_convert("UTC")

        if not (current_ts > last_ts):
            raise FutureForecastError("Future timestamp must be strictly after the last historical timestamp")

        row = dict(base)
        row["timestamp"] = current_ts
        row.update(_calendar_features_for_timestamp(current_ts, domain, region=region))
        row.update(_lag_rolling_from_history(target_history, domain=domain))
        row[target_column] = np.nan

        step_df = pd.DataFrame([row])
        # Preserve historical column order for transformer feature lookup.
        for col in history_df.columns:
            if col not in step_df.columns:
                step_df[col] = np.nan
        step_df = step_df[history_df.columns]

        pred_series = model.predict(step_df)
        pred_value = float(pred_series.iloc[0])
        if not np.isfinite(pred_value):
            raise FutureForecastError("Future forecast produced a non-finite prediction")

        target_history.append(pred_value)
        # Carry predicted target only into recursive history — not as an "actual".
        base = row
        base[target_column] = pred_value

        predictions.append(
            {
                "timestamp": current_ts.isoformat(),
                "predicted": pred_value,
            }
        )

    return predictions


def arima_future_forecast(
    *,
    model: ARIMAForecaster,
    history_df: pd.DataFrame,
    target_column: str,
    horizon: int,
    domain: str = "electricity",
) -> list[dict[str, Any]]:
    """Multi-step ARIMA forecast from the end of the full historical target series."""
    import warnings

    if model._fitted_model is None:
        raise FutureForecastError("ARIMA model must be fitted before future forecasting")

    last_ts = pd.Timestamp(history_df["timestamp"].iloc[-1])
    freq = DOMAIN_FREQUENCY[domain]
    future_timestamps = pd.date_range(
        last_ts + pd.tseries.frequencies.to_offset(freq),
        periods=horizon,
        freq=freq,
        tz="UTC",
    )
    if len(future_timestamps) != horizon:
        raise FutureForecastError("Failed to generate the requested future timestamp horizon")
    if not (future_timestamps[0] > last_ts):
        raise FutureForecastError("Future timestamp must be strictly after the last historical timestamp")

    history_target = history_df[target_column].astype(float).reset_index(drop=True)
    if history_target.isna().any():
        raise FutureForecastError("Historical target series contains NaN values")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = model._fitted_model.apply(history_target, refit=False)
            forecast_values = applied.forecast(steps=horizon)
    except Exception as err:
        raise FutureForecastError(f"ARIMA future forecast failed: {err}") from err

    predictions: list[dict[str, Any]] = []
    for ts, value in zip(future_timestamps, forecast_values):
        pred = float(value)
        if not np.isfinite(pred):
            raise FutureForecastError("ARIMA future forecast produced a non-finite prediction")
        predictions.append({"timestamp": pd.Timestamp(ts).isoformat(), "predicted": pred})
    return predictions


def generate_future_forecast(
    *,
    model: Any,
    model_key: str,
    domain: str,
    features_dir,
    horizon: int,
) -> dict[str, Any]:
    """Generate a future forecast payload from a fitted model and feature splits."""
    domain = domain.lower().strip()
    model_key = model_key.lower().strip()
    horizon = validate_horizon(horizon, domain=domain)

    if not is_future_forecast_supported(domain, model_key):
        raise FutureForecastError(future_support_error(domain, model_key))

    target_column = DOMAIN_TARGET_MAP[domain]
    history_df = load_chronological_history(features_dir, target_column)
    last_historical_timestamp = pd.Timestamp(history_df["timestamp"].iloc[-1]).isoformat()

    if model_key == "arima":
        predictions = arima_future_forecast(
            model=model,
            history_df=history_df,
            target_column=target_column,
            horizon=horizon,
            domain=domain,
        )
        exogenous_policy = "not_applicable"
    else:
        predictions = recursive_tabular_future_forecast(
            model=model,
            history_df=history_df,
            domain=domain,
            target_column=target_column,
            horizon=horizon,
        )
        exogenous_policy = (
            "last_observation_carried_forward"
            if domain == "retail"
            else "not_applicable"
        )

    # Hard guarantees for API consumers / tests
    if len(predictions) != horizon:
        raise FutureForecastError("Future forecast length does not match requested horizon")
    for item in predictions:
        if set(item.keys()) != {"timestamp", "predicted"}:
            raise FutureForecastError("Future predictions must contain only timestamp and predicted")
        if "actual" in item:
            raise FutureForecastError("Future predictions must not include actual values")

    timestamps = [pd.Timestamp(p["timestamp"]) for p in predictions]
    if timestamps != sorted(timestamps):
        raise FutureForecastError("Future timestamps are not chronological")
    if not (timestamps[0] > pd.Timestamp(history_df["timestamp"].iloc[-1])):
        raise FutureForecastError("First future timestamp must follow the last historical timestamp")

    return {
        "forecast_type": "future",
        "domain": domain,
        "model": model_key,
        "target": target_column,
        "horizon": horizon,
        "frequency": DOMAIN_FREQUENCY[domain],
        "last_historical_timestamp": last_historical_timestamp,
        "exogenous_policy": exogenous_policy,
        "predictions": predictions,
        "prediction_count": len(predictions),
    }
