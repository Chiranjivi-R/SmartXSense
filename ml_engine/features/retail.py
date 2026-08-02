"""Domain-aware retail forecasting features."""

from __future__ import annotations

import pandas as pd

from .common import add_history_features, canonicalize_time_target


SERIES_DIMENSIONS = ("store_id", "product_id", "category", "city", "channel")
LEAKAGE_PRONE_CURRENT_COLUMNS = ("revenue",)


def _series_columns(frame: pd.DataFrame) -> list[str]:
    candidate = [column for column in SERIES_DIMENSIONS if column in frame.columns and frame[column].notna().any()]
    if not candidate:
        return []
    if len(frame["timestamp"].drop_duplicates()) == len(frame):
        return []
    group_sizes = frame.groupby(candidate, dropna=False).size()
    if group_sizes.median() > 1:
        return candidate
    return []


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a defensible target aggregate while preserving descriptive metadata."""
    clean = frame.drop(columns=[column for column in LEAKAGE_PRONE_CURRENT_COLUMNS if column in frame], errors="ignore")
    numeric = clean.select_dtypes(include="number").columns.difference(["sales"])
    aggregations: dict[str, str] = {"sales": "sum"}
    aggregations.update({column: "mean" for column in numeric})
    other = [column for column in clean.columns if column not in {"timestamp", "sales", *numeric}]
    aggregations.update({column: "first" for column in other})
    return clean.groupby("timestamp", as_index=False, sort=True).agg(aggregations)


def build_retail_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    """Build within-series observation lags and shifted rolling target features.

    Retail records are not assumed to be daily; lag_7 means seven earlier
    observations in the same series. Current-period revenue is excluded because
    it can be mechanically determined by current sales and price.
    """
    result = canonicalize_time_target(frame, "sales")
    result = result.drop(columns=[column for column in LEAKAGE_PRONE_CURRENT_COLUMNS if column in result], errors="ignore")
    groups = _series_columns(result)
    if groups:
        # Repeated observations for the same series/time are summed deterministically.
        numeric = result.select_dtypes(include="number").columns.difference(["sales", *groups])
        aggregations: dict[str, str] = {"sales": "sum"}
        aggregations.update({column: "mean" for column in numeric})
        other = [column for column in result.columns if column not in {"timestamp", "sales", *groups, *numeric}]
        aggregations.update({column: "first" for column in other})
        result = result.groupby(["timestamp", *groups], as_index=False, dropna=False, sort=True).agg(aggregations)
        result = result.sort_values([*groups, "timestamp"], kind="stable").reset_index(drop=True)
    else:
        result = _aggregate(result)
    result, names = add_history_features(result, target="sales", lags=(1, 7, 14, 28), rolling_windows=(7, 14, 28), group_columns=groups or None)
    report = {"series_columns": groups, "aggregate_fallback": not bool(groups), "excluded_columns": ["revenue"] if "revenue" in frame.columns else [], "rows_missing_history": int(result[names].isna().any(axis=1).sum())}
    return result.sort_values("timestamp", kind="stable").reset_index(drop=True), names, report
