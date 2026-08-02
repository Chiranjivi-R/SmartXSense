"""Column-name normalization and source-schema alias resolution."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


RETAIL_ALIASES = {
    "timestamp": {"date", "invoice_date", "datetime", "date_time", "transaction_date", "order_date"},
    "sales": {"units_sold", "units", "quantity", "qty", "demand", "sales_units"},
    "product_id": {"product_name", "product", "item_id", "item", "sku", "brand"},
    "store_id": {"store", "store_code", "store_format", "location"},
    "price": {"selling_price", "base_price_inr", "base_price", "price_inr"},
    "revenue": {"revenue_inr", "sales_revenue"},
    "holiday_flag": {"holiday", "is_holiday"},
    "festival_flag": {"festival", "is_festival"},
    "temperature_c": {"temperature", "temperature_celsius"},
}

ELECTRICITY_ALIASES = {
    "timestamp": {"date_time", "datetime", "date", "time", "datehour", "date_hour"},
    "demand_mw": {
        "hourly_demand_met_mw",
        "hourly_demand_met_in_mw",
        "hourly_demand_met",
        "demand_met_mw",
        "demand_met",
        "grid_load_mw",
        "load_mw",
        "load",
        "demand",
    },
}

DOMAIN_TARGETS = {"retail": "sales", "electricity": "demand_mw"}


def normalise_column_name(name: object) -> str:
    """Convert source-specific headings to stable snake_case names."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return re.sub(r"_+", "_", text)


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize headings and make duplicate normalized headings unique."""
    result = frame.copy()
    used: dict[str, int] = {}
    normalized: list[str] = []
    for column in result.columns:
        base = normalise_column_name(column) or "unnamed_column"
        used[base] = used.get(base, 0) + 1
        normalized.append(base if used[base] == 1 else f"{base}_{used[base]}")
    result.columns = normalized
    return result


def rename_aliases(frame: pd.DataFrame, aliases: dict[str, set[str]]) -> pd.DataFrame:
    """Rename the first available source alias for each canonical concept."""
    result = frame.copy()
    rename_map: dict[str, str] = {}
    existing = set(result.columns)
    for canonical, candidates in aliases.items():
        if canonical in existing:
            continue
        match = next((candidate for candidate in candidates if candidate in existing), None)
        if match is not None:
            rename_map[match] = canonical
            existing.remove(match)
            existing.add(canonical)
    return result.rename(columns=rename_map)


def _combine_electricity_year(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a separate Year value when the timestamp source lacks one."""
    if "timestamp" not in frame.columns or "year" not in frame.columns:
        return frame
    timestamp_text = frame["timestamp"].astype("string").str.strip()
    has_year = timestamp_text.str.contains(r"\b\d{4}\b", na=False)
    result = frame.copy()
    result.loc[~has_year, "timestamp"] = result.loc[~has_year, "year"].astype("string") + "-" + timestamp_text[~has_year]
    return result


def normalise_domain_schema(frame: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Apply the single supported alias map for a dataset domain."""
    if domain == "retail":
        return rename_aliases(normalise_columns(frame), RETAIL_ALIASES)
    if domain == "electricity":
        return _combine_electricity_year(rename_aliases(normalise_columns(frame), ELECTRICITY_ALIASES))
    raise ValueError(f"Unsupported dataset domain: {domain}")
