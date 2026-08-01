"""Domain-aware retail and electricity preprocessing transformations."""

from __future__ import annotations

import pandas as pd

from .cleaner import PreprocessingError, add_calendar_features, clean_timestamp_and_target, require_columns
from .schema import normalise_domain_schema


def preprocess_retail(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    """Normalize heterogeneous retail datasets to timestamp and numeric sales."""
    result = normalise_domain_schema(frame, "retail")
    require_columns(result, {"timestamp", "sales"})
    cleaned, removed = clean_timestamp_and_target(result, "sales")
    cleaned, features = add_calendar_features(cleaned)
    return cleaned, features, removed


def preprocess_electricity(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
    """Normalize Indian hourly demand profiles to timestamp and demand_mw."""
    result = normalise_domain_schema(frame, "electricity")
    require_columns(result, {"timestamp", "demand_mw"})
    cleaned, removed = clean_timestamp_and_target(result, "demand_mw")
    # An exact source-row duplicate is already removed above. Timestamp-only
    # duplicates are resolved deterministically without inventing demand data.
    before = len(cleaned)
    cleaned = cleaned.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    removed += before - len(cleaned)
    cleaned, features = add_calendar_features(cleaned)
    return cleaned, features, removed


def preprocess_frame(frame: pd.DataFrame, domain: str) -> tuple[pd.DataFrame, list[str], int]:
    if domain == "retail":
        return preprocess_retail(frame)
    if domain == "electricity":
        return preprocess_electricity(frame)
    raise PreprocessingError("Unsupported dataset domain")
