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

import warnings
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

DOMAIN_MAX_HORIZON = {
    "retail": 90,
    "electricity": 2160,
}

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
    "electricity": frozenset({"random_forest", "xgboost", "trend_xgboost", "arima", "sarima", "hybrid"}),
}

MODEL_MAX_HORIZON: dict[str, int] = {
    "sarima": 168,
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
        "Future forecasting for hybrid is supported for electricity using future-capable "
        "components (arima, random_forest, xgboost). For retail or unsupported components, "
        "future forecasting is disabled."
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


def is_future_forecast_supported(domain: str, model_key: str, *, horizon: int | None = None) -> bool:
    domain = domain.lower().strip()
    model_key = model_key.lower().strip()
    base_supported = model_key in FUTURE_SUPPORTED_MODELS.get(domain, frozenset())
    if not base_supported:
        return False
    if horizon is not None and model_key in MODEL_MAX_HORIZON:
        if horizon > MODEL_MAX_HORIZON[model_key]:
            return False
    return True


def future_support_error(domain: str, model_key: str, *, horizon: int | None = None) -> str:
    domain = domain.lower().strip()
    model_key = model_key.lower().strip()
    if horizon is not None and model_key in MODEL_MAX_HORIZON and horizon > MODEL_MAX_HORIZON[model_key]:
        return (
            f"Future forecasting for model '{model_key}' is restricted to horizon <= {MODEL_MAX_HORIZON[model_key]} "
            f"(requested {horizon}). For longer horizons, use 'trend_xgboost' or Auto Select."
        )
    if model_key in ("arima", "sarima", "trend_xgboost") and domain == "retail":
        return f"Future forecasting is not currently supported for model {model_key} on the retail domain."
    if model_key == "hybrid" and domain == "retail":
        return "Future forecasting is not currently supported for model hybrid on the retail domain."
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
    clean_domain = str(domain).lower().strip()
    max_allowed = DOMAIN_MAX_HORIZON.get(clean_domain, MAX_HORIZON)
    if horizon > max_allowed:
        raise FutureForecastError(
            f"Parameter 'horizon' ({horizon}) exceeds maximum allowed value of {max_allowed} for domain '{domain}'"
        )
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


def _detect_calendar_region(base: dict[str, Any], history_df: pd.DataFrame) -> str:
    declared = base.get("calendar_region")
    if declared and str(declared).strip().lower() in ("india", "uk", "other"):
        return str(declared).strip().lower()
    country = str(base.get("country") or "").strip().lower()
    if country == "india":
        return "india"
    state = str(base.get("state") or "").strip().lower()
    if state in {"maharashtra", "delhi", "karnataka", "tamil nadu", "gujarat", "uttar pradesh", "west bengal", "rajasthan"}:
        return "india"
    if "festival_name" in history_df.columns:
        fest_names = history_df["festival_name"].dropna().astype(str).str.lower()
        if fest_names.str.contains("diwali|holi|republic day|independence day|dussehra|makar sankranti").any():
            return "india"
    return "other"


def _collect_festival_dates(history_df: pd.DataFrame, region: str) -> tuple[set[Any], dict[tuple[int, int], str]]:
    """Collect all known historical festival dates and future regional festival dates."""
    fest_dates: set[Any] = set()
    named_festivals: dict[tuple[int, int], str] = {}

    # 1. Historical festival dates from dataset
    matched_col = None
    for cand in ("festival_flag", "is_festival", "holiday_flag", "is_holiday"):
        if cand in history_df.columns:
            matched_col = cand
            break

    if matched_col and "timestamp" in history_df.columns:
        ts_series = pd.to_datetime(history_df["timestamp"], errors="coerce", utc=True)
        fest_series = pd.to_numeric(history_df[matched_col], errors="coerce").fillna(0)
        active_dates = ts_series.loc[fest_series == 1].dt.date.dropna()
        fest_dates.update(active_dates)

    # 2. Regional calendar dates for future periods if region is known
    if region == "india":
        from ml_engine.preprocessing.calendar_features import _INDIA_FIXED, INDIA_FESTIVALS
        years = {ts.year for ts in pd.to_datetime(history_df["timestamp"], errors="coerce", utc=True).dropna()}
        if years:
            max_year = max(years)
            years.update(range(min(years), max_year + 3))
        else:
            years = {2024, 2025, 2026, 2027}

        for y in years:
            for (m, d), name in _INDIA_FIXED.items():
                try:
                    dt = pd.Timestamp(year=y, month=m, day=d).date()
                    fest_dates.add(dt)
                    named_festivals[(m, d)] = name
                except Exception:
                    pass
            for (m, d), name in INDIA_FESTIVALS.get(y, {}).items():
                try:
                    dt = pd.Timestamp(year=y, month=m, day=d).date()
                    fest_dates.add(dt)
                    named_festivals[(m, d)] = name
                except Exception:
                    pass

    return fest_dates, named_festivals


def _compute_festival_distance(dt: Any, fest_dates: set[Any]) -> float:
    """Compute signed festival distance in days with [-4.0, +4.0] saturation matching common.py."""
    if not fest_dates:
        return 4.0

    min_abs = 999
    best_signed = 999
    for fd in fest_dates:
        diff = (dt - fd).days
        if abs(diff) < min_abs:
            min_abs = abs(diff)
            best_signed = diff
        elif abs(diff) == min_abs:
            best_signed = min(best_signed, diff)

    if best_signed <= -4:
        return -4.0
    if best_signed >= 4:
        return 4.0
    return float(best_signed)


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
    region = _detect_calendar_region(base, history_df)
    fest_dates, named_festivals = _collect_festival_dates(history_df, region)
    target_history = [float(v) for v in history_df[target_column].astype(float).tolist()]

    has_ewma = "ewma_3" in history_df.columns
    current_ewma = float(history_df["ewma_3"].iloc[-1]) if has_ewma else None
    prev_target = float(history_df[target_column].iloc[-1])

    rows: list[dict[str, Any]] = []

    for step, ts in enumerate(future_timestamps):
        current_date = ts.date()
        row = dict(base)
        row["timestamp"] = ts
        row.update(_calendar_features_for_timestamp(ts, domain, region=region))
        row.update(_lag_rolling_from_history(target_history, domain=domain))

        # Deterministic payday_week
        if "payday_week" in history_df.columns:
            days_in_month = ts.days_in_month
            row["payday_week"] = 1.0 if (ts.day <= 7 or (days_in_month - ts.day) <= 2) else 0.0

        # Recursive EWMA update
        if has_ewma and current_ewma is not None:
            current_ewma = 0.5 * prev_target + 0.5 * current_ewma
            row["ewma_3"] = current_ewma

        # Deterministic Festival Proximity and Flags
        if fest_dates:
            dist = _compute_festival_distance(current_date, fest_dates)
            if "festival_distance" in history_df.columns:
                row["festival_distance"] = dist

            is_fest = 1.0 if current_date in fest_dates else 0.0
            if "festival_flag" in history_df.columns:
                row["festival_flag"] = is_fest
            if "is_festival" in history_df.columns:
                row["is_festival"] = int(is_fest)
            if "holiday_flag" in history_df.columns:
                row["holiday_flag"] = is_fest
            if "is_holiday" in history_df.columns:
                row["is_holiday"] = int(is_fest)

            fest_name = named_festivals.get((current_date.month, current_date.day))
            if "festival_name" in history_df.columns:
                row["festival_name"] = fest_name
            if "holiday_name" in history_df.columns:
                row["holiday_name"] = fest_name

            season_flag = 1.0 if abs(dist) <= 3 else 0.0
            if "festival_season_flag" in history_df.columns:
                row["festival_season_flag"] = season_flag
            if "festival_season" in history_df.columns:
                row["festival_season"] = fest_name if season_flag == 1.0 else None

        # Dynamic Festival Interactions with current recursive values
        fest_indicator = float(row.get("festival_flag", row.get("is_festival", 0.0)))
        if "fest_x_lag_1" in history_df.columns:
            lag1 = float(row.get("lag_1", 0.0)) if not np.isnan(float(row.get("lag_1", 0.0))) else 0.0
            row["fest_x_lag_1"] = fest_indicator * lag1
        if "fest_x_lag_7" in history_df.columns:
            lag7 = float(row.get("lag_7", 0.0)) if not np.isnan(float(row.get("lag_7", 0.0))) else 0.0
            row["fest_x_lag_7"] = fest_indicator * lag7
        if "fest_x_lag_14" in history_df.columns:
            lag14 = float(row.get("lag_14", 0.0)) if not np.isnan(float(row.get("lag_14", 0.0))) else 0.0
            row["fest_x_lag_14"] = fest_indicator * lag14
        if "fest_x_rolling_mean_7" in history_df.columns:
            rm7 = float(row.get("rolling_mean_7", 0.0)) if not np.isnan(float(row.get("rolling_mean_7", 0.0))) else 0.0
            row["fest_x_rolling_mean_7"] = fest_indicator * rm7
        if "fest_x_rolling_mean_14" in history_df.columns:
            rm14 = float(row.get("rolling_mean_14", 0.0)) if not np.isnan(float(row.get("rolling_mean_14", 0.0))) else 0.0
            row["fest_x_rolling_mean_14"] = fest_indicator * rm14
        if "fest_season_x_lag1" in history_df.columns:
            season_flag = float(row.get("festival_season_flag", 0.0))
            lag1 = float(row.get("lag_1", 0.0)) if not np.isnan(float(row.get("lag_1", 0.0))) else 0.0
            row["fest_season_x_lag1"] = season_flag * lag1

        # Future actual target is unknown — never fabricate an actual value for metrics.
        if predicted_targets is not None:
            pred = float(predicted_targets[step])
            row[target_column] = pred
            target_history.append(pred)
            prev_target = pred
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
    region = _detect_calendar_region(base, history_df)
    fest_dates, named_festivals = _collect_festival_dates(history_df, region)
    target_history = [float(v) for v in history_df[target_column].astype(float).tolist()]

    has_ewma = "ewma_3" in history_df.columns
    current_ewma = float(history_df["ewma_3"].iloc[-1]) if has_ewma else None
    prev_target = float(history_df[target_column].iloc[-1])

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

        current_date = current_ts.date()
        row = dict(base)
        row["timestamp"] = current_ts
        row.update(_calendar_features_for_timestamp(current_ts, domain, region=region))
        row.update(_lag_rolling_from_history(target_history, domain=domain))
        row[target_column] = np.nan

        # Deterministic payday_week
        if "payday_week" in history_df.columns:
            days_in_month = current_ts.days_in_month
            row["payday_week"] = 1.0 if (current_ts.day <= 7 or (days_in_month - current_ts.day) <= 2) else 0.0

        # Recursive EWMA update
        if has_ewma and current_ewma is not None:
            current_ewma = 0.5 * prev_target + 0.5 * current_ewma
            row["ewma_3"] = current_ewma

        # Deterministic Festival Proximity and Flags
        if fest_dates:
            dist = _compute_festival_distance(current_date, fest_dates)
            if "festival_distance" in history_df.columns:
                row["festival_distance"] = dist

            is_fest = 1.0 if current_date in fest_dates else 0.0
            if "festival_flag" in history_df.columns:
                row["festival_flag"] = is_fest
            if "is_festival" in history_df.columns:
                row["is_festival"] = int(is_fest)
            if "holiday_flag" in history_df.columns:
                row["holiday_flag"] = is_fest
            if "is_holiday" in history_df.columns:
                row["is_holiday"] = int(is_fest)

            fest_name = named_festivals.get((current_date.month, current_date.day))
            if "festival_name" in history_df.columns:
                row["festival_name"] = fest_name
            if "holiday_name" in history_df.columns:
                row["holiday_name"] = fest_name

            season_flag = 1.0 if abs(dist) <= 3 else 0.0
            if "festival_season_flag" in history_df.columns:
                row["festival_season_flag"] = season_flag
            if "festival_season" in history_df.columns:
                row["festival_season"] = fest_name if season_flag == 1.0 else None

        # Dynamic Festival Interactions with current recursive values
        fest_indicator = float(row.get("festival_flag", row.get("is_festival", 0.0)))
        if "fest_x_lag_1" in history_df.columns:
            lag1 = float(row.get("lag_1", 0.0)) if not np.isnan(float(row.get("lag_1", 0.0))) else 0.0
            row["fest_x_lag_1"] = fest_indicator * lag1
        if "fest_x_lag_7" in history_df.columns:
            lag7 = float(row.get("lag_7", 0.0)) if not np.isnan(float(row.get("lag_7", 0.0))) else 0.0
            row["fest_x_lag_7"] = fest_indicator * lag7
        if "fest_x_lag_14" in history_df.columns:
            lag14 = float(row.get("lag_14", 0.0)) if not np.isnan(float(row.get("lag_14", 0.0))) else 0.0
            row["fest_x_lag_14"] = fest_indicator * lag14
        if "fest_x_rolling_mean_7" in history_df.columns:
            rm7 = float(row.get("rolling_mean_7", 0.0)) if not np.isnan(float(row.get("rolling_mean_7", 0.0))) else 0.0
            row["fest_x_rolling_mean_7"] = fest_indicator * rm7
        if "fest_x_rolling_mean_14" in history_df.columns:
            rm14 = float(row.get("rolling_mean_14", 0.0)) if not np.isnan(float(row.get("rolling_mean_14", 0.0))) else 0.0
            row["fest_x_rolling_mean_14"] = fest_indicator * rm14
        if "fest_season_x_lag1" in history_df.columns:
            season_flag = float(row.get("festival_season_flag", 0.0))
            lag1 = float(row.get("lag_1", 0.0)) if not np.isnan(float(row.get("lag_1", 0.0))) else 0.0
            row["fest_season_x_lag1"] = season_flag * lag1

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
        prev_target = pred_value
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
    if getattr(model, "seasonal_order", None) is not None and horizon > 168:
        raise FutureForecastError(
            f"SARIMA future forecasting is restricted to horizon <= 168 (requested {horizon}). "
            f"For longer horizons, use 'trend_xgboost' or Auto Select."
        )

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
    if getattr(model, "training_window", None) is not None:
        window = int(model.training_window)
        if len(history_target) > window:
            history_target = history_target.iloc[-window:].reset_index(drop=True)
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


def hybrid_future_forecast(
    *,
    model: Any,
    history_df: pd.DataFrame,
    target_column: str,
    horizon: int,
    domain: str = "electricity",
) -> list[dict[str, Any]]:
    """Multi-step hybrid future forecast combining valid component models with frozen validation weights."""
    if domain != "electricity":
        raise FutureForecastError(f"Hybrid future forecasting is not supported for domain '{domain}'")

    from ml_engine.models.hybrid import HybridForecaster

    if not isinstance(model, HybridForecaster):
        raise FutureForecastError(f"Expected HybridForecaster instance, got {type(model).__name__}")

    if not hasattr(model, "weights") or not model.weights:
        raise FutureForecastError("Hybrid model must have fitted validation weights before future forecasting")

    if not hasattr(model, "component_models") or not model.component_models:
        raise FutureForecastError("Hybrid model must have fitted component_models before future forecasting")

    # Verify all components are future-capable
    for comp in model.components:
        if not is_future_forecast_supported(domain, comp) or comp == "hybrid":
            raise FutureForecastError(
                f"Component '{comp}' does not support future forecasting in domain '{domain}'"
            )

    component_future_preds: dict[str, list[dict[str, Any]]] = {}
    for comp in model.components:
        comp_model = model.component_models[comp]
        if comp in ("arima", "sarima"):
            preds = arima_future_forecast(
                model=comp_model,
                history_df=history_df,
                target_column=target_column,
                horizon=horizon,
                domain=domain,
            )
        else:
            preds = recursive_tabular_future_forecast(
                model=comp_model,
                history_df=history_df,
                domain=domain,
                target_column=target_column,
                horizon=horizon,
            )
        component_future_preds[comp] = preds

    # Validate alignment and blend using frozen validation weights
    first_comp = model.components[0]
    first_preds = component_future_preds[first_comp]
    combined_predictions: list[dict[str, Any]] = []

    for step_idx in range(horizon):
        ref_ts = first_preds[step_idx]["timestamp"]
        combined_val = 0.0
        for comp in model.components:
            comp_step = component_future_preds[comp][step_idx]
            if comp_step["timestamp"] != ref_ts:
                raise FutureForecastError(
                    f"Timestamp mismatch at step {step_idx} between components '{first_comp}' and '{comp}'"
                )
            weight = float(model.weights[comp])
            combined_val += weight * float(comp_step["predicted"])

        if not np.isfinite(combined_val):
            raise FutureForecastError("Hybrid future forecast produced a non-finite prediction")

        combined_predictions.append({
            "timestamp": ref_ts,
            "predicted": float(combined_val),
        })

    return combined_predictions


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

    if not is_future_forecast_supported(domain, model_key, horizon=horizon):
        raise FutureForecastError(future_support_error(domain, model_key, horizon=horizon))

    target_column = DOMAIN_TARGET_MAP[domain]
    history_df = load_chronological_history(features_dir, target_column)
    last_historical_timestamp = pd.Timestamp(history_df["timestamp"].iloc[-1]).isoformat()

    if model_key in ("arima", "sarima"):
        predictions = arima_future_forecast(
            model=model,
            history_df=history_df,
            target_column=target_column,
            horizon=horizon,
            domain=domain,
        )
        exogenous_policy = "not_applicable"
    elif model_key == "hybrid":
        predictions = hybrid_future_forecast(
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
