"""Shared invariants for target-history forecasting features."""

from __future__ import annotations

import pandas as pd


class FeatureEngineeringError(ValueError):
    """Raised when canonical processed data cannot support feature generation."""


def canonicalize_time_target(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """Validate canonical columns and return a stable chronological copy."""
    if "timestamp" not in frame.columns or target not in frame.columns:
        missing = sorted({"timestamp", target} - set(frame.columns))
        raise FeatureEngineeringError(f"Missing required columns: {', '.join(missing)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce", utc=True)
    result[target] = pd.to_numeric(result[target], errors="coerce")
    result = result.dropna(subset=["timestamp", target])
    if result.empty:
        raise FeatureEngineeringError("No valid timestamp and numeric target rows remain")
    return result.sort_values("timestamp", kind="stable").reset_index(drop=True)


def add_history_features(
    frame: pd.DataFrame,
    *,
    target: str,
    lags: tuple[int, ...] = (1, 7, 14, 28),
    rolling_windows: tuple[int, ...] = (7, 14, 28),
    group_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add lag/rolling features from a target shifted one observation backward."""
    result = frame.copy()
    feature_names: list[str] = []

    if group_columns:
        grouped = result.groupby(group_columns, dropna=False, sort=False)[target]
        for lag in lags:
            name = f"lag_{lag}"
            result[name] = grouped.shift(lag)
            feature_names.append(name)
        # transform preserves the original frame index and isolates every series.
        for window in rolling_windows:
            history = grouped.shift(1)
            mean_name, std_name = f"rolling_mean_{window}", f"rolling_std_{window}"
            result[mean_name] = history.groupby([result[column] for column in group_columns], dropna=False, sort=False).transform(lambda values: values.rolling(window, min_periods=window).mean())
            result[std_name] = history.groupby([result[column] for column in group_columns], dropna=False, sort=False).transform(lambda values: values.rolling(window, min_periods=window).std())
            feature_names.extend([mean_name, std_name])
    else:
        history = result[target].shift(1)
        for lag in lags:
            name = f"lag_{lag}"
            result[name] = result[target].shift(lag)
            feature_names.append(name)
        for window in rolling_windows:
            mean_name, std_name = f"rolling_mean_{window}", f"rolling_std_{window}"
            result[mean_name] = history.rolling(window, min_periods=window).mean()
            result[std_name] = history.rolling(window, min_periods=window).std()
            feature_names.extend([mean_name, std_name])
    return result, feature_names


def irregular_hourly_gaps(timestamps: pd.Series) -> list[dict[str, str]]:
    """Report observed gaps without adding missing rows or imputing targets."""
    gaps = timestamps.diff()
    return [
        {"after": timestamps.iloc[index - 1].isoformat(), "before": timestamps.iloc[index].isoformat(), "gap": str(gap)}
        for index, gap in enumerate(gaps)
        if index and gap.value != 3_600_000_000_000
    ]
