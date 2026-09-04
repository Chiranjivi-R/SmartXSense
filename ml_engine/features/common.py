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
    ewma_spans: tuple[int, ...] = (3,),
    group_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add lag/rolling/EWMA features from a target shifted one observation backward."""
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
        for span in ewma_spans:
            history = grouped.shift(1)
            ewma_name = f"ewma_{span}"
            result[ewma_name] = history.groupby([result[column] for column in group_columns], dropna=False, sort=False).transform(lambda values: values.ewm(span=span, adjust=False).mean())
            feature_names.append(ewma_name)
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
        for span in ewma_spans:
            ewma_name = f"ewma_{span}"
            result[ewma_name] = history.ewm(span=span, adjust=False).mean()
            feature_names.append(ewma_name)
    return result, feature_names


def add_festival_interaction_features(
    frame: pd.DataFrame,
    *,
    candidate_festival_columns: tuple[str, ...] = ("festival_flag", "holiday_flag", "event_flag", "promo_flag", "is_holiday", "is_festival"),
) -> tuple[pd.DataFrame, list[str]]:
    """Add festival x lag and festival x rolling mean interaction features if a valid festival indicator exists."""
    result = frame.copy()
    feature_names: list[str] = []

    # Find first existing valid festival indicator with variance > 0
    matched_fest_col = None
    for col in candidate_festival_columns:
        if col in result.columns and pd.api.types.is_numeric_dtype(result[col]):
            s = result[col].dropna()
            if not s.empty and s.nunique() > 1 and s.max() > s.min():
                matched_fest_col = col
                break

    if not matched_fest_col:
        return result, []

    fest_series = result[matched_fest_col].fillna(0).astype(float)

    # 0. Distance-aware festival proximity feature (signed distance -3..+3, neutral saturation at -4.0 / +4.0 outside window)
    ts_series = pd.to_datetime(result["timestamp"], errors="coerce", utc=True) if "timestamp" in result.columns else None
    if ts_series is not None and not ts_series.isna().all():
        fest_dates = sorted(list(set(ts_series.loc[fest_series == 1].dt.date)))
        if fest_dates:
            dist_col: list[float] = []
            for dt in ts_series.dt.date:
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
                    dist_col.append(-4.0)
                elif best_signed >= 4:
                    dist_col.append(4.0)
                else:
                    dist_col.append(float(best_signed))
            result["festival_distance"] = dist_col
            feature_names.append("festival_distance")


    # 1. Festival x Lag interactions
    for lag in (1, 7, 14):
        lag_col = f"lag_{lag}"
        if lag_col in result.columns:
            name = f"fest_x_lag_{lag}"
            result[name] = fest_series * result[lag_col].fillna(0)
            feature_names.append(name)

    # 2. Festival x Rolling mean interactions
    for w in (7, 14):
        roll_col = f"rolling_mean_{w}"
        if roll_col in result.columns:
            name = f"fest_x_rolling_mean_{w}"
            result[name] = fest_series * result[roll_col].fillna(0)
            feature_names.append(name)

    # 3. Festival season window flag and interaction with lag_1
    # festival_season covers the ±7-day neighbourhood around each named festival date
    # (a wider event-window signal than the point-day festival_flag).
    # It is purely calendar-derived: no target values are used in its construction.
    # fest_season_x_lag1 = festival_season_flag × lag_1, giving the model a
    # "recent-demand-during-the-event-period" interaction signal.
    if "festival_season" in result.columns:
        result["festival_season_flag"] = result["festival_season"].notna().astype(float)
        feature_names.append("festival_season_flag")
        if "lag_1" in result.columns:
            result["fest_season_x_lag1"] = result["festival_season_flag"] * result["lag_1"].fillna(0)
            feature_names.append("fest_season_x_lag1")

    return result, feature_names





def irregular_hourly_gaps(timestamps: pd.Series) -> list[dict[str, str]]:
    """Report observed gaps without adding missing rows or imputing targets."""
    gaps = timestamps.diff()
    return [
        {"after": timestamps.iloc[index - 1].isoformat(), "before": timestamps.iloc[index].isoformat(), "gap": str(gap)}
        for index, gap in enumerate(gaps)
        if index and gap.value != 3_600_000_000_000
    ]
