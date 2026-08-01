"""Leakage-safe, conservative shared time-series cleaning helpers."""

from __future__ import annotations

import pandas as pd


class PreprocessingError(ValueError):
    """Raised when a file cannot form a usable forecasting dataset."""


def require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise PreprocessingError(f"Missing required columns: {', '.join(missing)}")


def clean_timestamp_and_target(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, int]:
    """Drop rows with unusable timestamp/target values and sort chronologically."""
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
    result[target] = pd.to_numeric(result[target], errors="coerce")
    before = len(result)
    result = result.dropna(subset=["timestamp", target])
    result = result.drop_duplicates().sort_values("timestamp", kind="stable").reset_index(drop=True)
    if result.empty:
        raise PreprocessingError("No valid timestamp and numeric target rows remain after cleaning")
    return result, before - len(result)


def add_calendar_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Create deterministic calendar features only; no lag or rolling values."""
    result = frame.copy()
    timestamp = result["timestamp"].dt
    result["year"] = timestamp.year
    result["month"] = timestamp.month
    result["day"] = timestamp.day
    result["day_of_week"] = timestamp.dayofweek
    result["quarter"] = timestamp.quarter
    result["is_weekend"] = (timestamp.dayofweek >= 5).astype("int8")
    features = ["year", "month", "day", "day_of_week", "quarter", "is_weekend"]
    if timestamp.hour.nunique() > 1 or (timestamp.hour != 0).any():
        result["hour"] = timestamp.hour
        features.append("hour")
    # Parquet represents timezone-aware values consistently; retaining UTC
    # eliminates ambiguous local timestamps from mixed source files.
    result["timestamp"] = result["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return result, features
