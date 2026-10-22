"""Column-name normalization and source-schema alias resolution."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd


RETAIL_ALIASES = {
    "timestamp": {
        "date", "invoice_date", "datetime", "date_time", "transaction_date", "order_date",
        "invoicedate", "orderdate", "transactiondate", "purchase_date", "sales_date",
        "created_at", "updated_at", "event_time", "event_date", "timestamp_utc",
    },
    "sales": {
        "units_sold", "units", "quantity", "qty", "demand", "sales_units",
        "quantity_sold", "demand_units", "orders", "order_count", "revenue",
        "amount", "volume", "sales_revenue", "total_sales",
    },
    "product_id": {"item_id", "sku", "brand", "stock_code", "stockcode"},
    "store_id": {"store", "store_code", "store_format", "location"},

    "price": {"selling_price", "base_price_inr", "base_price", "price_inr", "unit_price", "unitprice"},
    "revenue": {"revenue_inr", "sales_revenue"},
    "holiday_flag": {"holiday", "is_holiday"},
    "festival_flag": {"festival", "is_festival"},
    "temperature_c": {"temperature", "temperature_celsius"},
}


ELECTRICITY_ALIASES = {
    "timestamp": {"date_time", "datetime", "date", "time", "datehour", "date_hour", "timestamp_utc"},
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
        "power",
        "consumption",
        "energy",
        "usage",
    },
}

DOMAIN_TARGETS = {"retail": "sales", "electricity": "demand_mw"}


def normalise_column_name(name: object) -> str:
    """Convert source-specific headings to stable snake_case names."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
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


def normalise_domain_schema(
    frame: pd.DataFrame,
    domain: str,
    *,
    target: str | None = None,
    time_column: str | None = None,
) -> pd.DataFrame:
    """Apply automatic feature-role resolution and alias mapping for a dataset domain."""
    domain_clean = (domain or "retail").lower().strip()
    if domain_clean not in ("retail", "electricity"):
        raise ValueError(f"Unsupported dataset domain: {domain}")

    # Step 1: Use centralized feature discovery to resolve target and timestamp roles
    from .feature_discovery import discover_features_and_resolve_schema

    try:
        resolved = discover_features_and_resolve_schema(
            frame,
            domain=domain_clean,
            explicit_target=target,
            explicit_time_col=time_column,
        )
        if resolved.time_column and resolved.target_column:
            # Adapt frame non-destructively
            adapted = resolved.adapt_frame(frame)
        else:
            adapted = normalise_columns(frame)
    except Exception:
        adapted = normalise_columns(frame)

    # Step 2: Apply legacy alias mapping for secondary concept normalization (store, product, price, etc.)
    norm_frame = normalise_columns(adapted)
    aliases = RETAIL_ALIASES if domain_clean == "retail" else ELECTRICITY_ALIASES
    result = rename_aliases(norm_frame, aliases)

    if domain_clean == "electricity":
        result = _combine_electricity_year(result)

    return result
