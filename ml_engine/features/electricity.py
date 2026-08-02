"""Hourly electricity demand forecasting features."""

from __future__ import annotations

import pandas as pd

from .common import add_history_features, canonicalize_time_target, irregular_hourly_gaps


def build_electricity_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """Build shifted hourly target-history features without filling source gaps."""
    result = canonicalize_time_target(frame, "demand_mw")
    before = len(result)
    result = result.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp", kind="stable").reset_index(drop=True)
    gaps = irregular_hourly_gaps(result["timestamp"])
    result, names = add_history_features(result, target="demand_mw", lags=(1, 24, 48, 168), rolling_windows=(24, 168))
    report = {"duplicate_timestamps_removed": before - len(result), "irregular_gaps": gaps, "rows_missing_history": int(result[names].isna().any(axis=1).sum())}
    return result, names, report
