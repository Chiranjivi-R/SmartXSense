"""Centralized Automatic Feature Discovery and Schema Resolution Layer.

SmartXSense Framework — Dataset-Adaptive Schema Resolution.

Architectural Guarantees:
1. Multi-signal inference: inspects column names, aliases, data types, sample values,
   missingness, temporal frequency, unique counts, and variance.
2. Non-destructive mapping: distinguishes between original uploaded column names,
   inferred semantic roles, and internal canonical names expected by ML models.
3. Explicit user selection precedence: explicit user selections for target, timestamp,
   product, or location are respected and never silently replaced.
4. Safe ambiguity detection: flags datasets with multiple competing plausible targets
   instead of silently guessing.
5. Rich validation diagnostics: provides actionable explanations for constant targets,
   missing timestamps, insufficient history, or incompatible structures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import difflib
import math
import re
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Keyword Dictionaries & Regex Patterns
# ---------------------------------------------------------------------------

TIME_KEYWORDS = {
    "date", "datetime", "timestamp", "time", "day", "month", "year",
    "invoice_date", "order_date", "transaction_date", "purchase_date",
    "sales_date", "created_at", "updated_at", "event_time", "event_date",
    "invoicedate", "orderdate", "transactiondate", "date_time", "datehour",
    "date_hour", "period", "timestamp_utc", "time_stamp", "ds",
    "time_index", "datum", "observation_date", "record_date", "read_date",
    "measurement_date", "period_start", "period_end",
}

PERSONAL_DATE_KEYWORDS = {
    "date_of_birth", "dob", "birth_date", "birthdate", "expiry_date", "due_date",
}

# Unit and concept patterns for electricity demand
ELECTRICITY_TARGET_KEYWORDS = {
    "load", "demand", "demand_mw", "power", "consumption", "energy", "usage",
    "grid_load", "grid_load_mw", "hourly_demand_met_mw", "hourly_demand_met",
    "load_mw", "demand_met", "mw", "kwh", "mwh", "kw", "gw", "gwh", "megawatts",
    "kilowatts", "energy_consumption", "generation", "net_load", "active_power",
    "total_load", "actual_load", "system_load",
}

ELECTRICITY_TARGET_REGEX = re.compile(
    r"(?:^|_)(?:mw|mwh|kw|kwh|gw|gwh|power|load|demand|consumption|usage|energy|grid_load|generation|net_load)(?:$|_)",
    re.IGNORECASE,
)

# Concept patterns for retail sales and demand
RETAIL_TARGET_KEYWORDS = {
    "sales", "quantity", "qty", "units", "units_sold", "quantity_sold",
    "sales_units", "demand", "demand_units", "orders", "order_count",
    "revenue", "amount", "sale_amount", "volume", "sales_revenue", "total_sales",
    "item_count", "item_qty", "units_sold_qty", "order_qty", "sold_qty",
    "sales_qty", "turnover", "gross_sales", "net_sales", "purchases", "unit_sales",
}

RETAIL_TARGET_REGEX = re.compile(
    r"(?:^|_)(?:sales|units_sold|quantity_sold|qty_sold|units|quantity|qty|revenue|turnover|orders|order_count|volume|demand|amount)(?:$|_)",
    re.IGNORECASE,
)

PRICE_KEYWORDS = {
    "price", "unit_price", "selling_price", "cost", "rate", "mrp", "base_price",
    "unit_cost", "base_price_inr", "fee", "cpi_combined", "cpi_food", "wpi_index",
    "fuel_price_petrol", "msrp", "unitprice",
}

PROMO_KEYWORDS = {
    "promotion", "promo", "discount", "offer", "coupon", "is_promo", "holiday",
    "discount_pct", "discount_percentage", "discount_amount", "deal_size", "offer_pct",
    "event", "festival", "festival_flag",
}

GEOGRAPHIC_KEYWORDS = {
    "country", "country_name", "country_code", "nation", "state", "state_name",
    "province", "district", "district_name", "city", "city_name", "town",
    "region", "location", "place", "area", "zone", "territory", "county",
    "zipcode", "zip_code", "postal_code", "pincode", "pin_code", "postcode",
    "site", "branch", "store", "store_location", "market", "depot", "hub",
    "terminal", "facility", "station", "address",
}

ENTITY_ID_KEYWORDS = {
    "id", "no", "number", "code", "sku", "key", "index", "customer", "invoice",
    "stock", "product", "store", "item", "user", "order_id", "invoice_no",
    "stock_code", "customer_id", "invoiceno", "stockcode", "customerid",
    "meter_id", "meter_no", "meter", "substation", "feeder", "sensor_id",
}

EXCLUDED_PERSONAL_OR_IDENTIFIER_KEYWORDS = {
    "customer_id", "customer_name", "first_name", "last_name", "name",
    "date_of_birth", "dob", "birth_date", "birthdate", "phone", "mobile",
    "email", "address", "postal_code", "zipcode", "zip_code", "pincode",
    "order_id", "transaction_id", "invoice_no", "invoice_id", "product_id",
    "product_name", "sku_id", "user_id", "index", "unnamed_0", "uuid", "row_id",
    "ssn", "credit_card", "customerid", "invoiceno", "stockcode", "orderid",
    "ship_date", "shipdate", "profit", "net_profit", "margin", "gross_profit",
}

LEAKAGE_COLUMNS = {
    "retail": {"revenue", "sales_revenue", "gross_revenue", "net_revenue", "profit", "margin"},
    "electricity": set(),
}

CANONICAL_TARGETS = {
    "retail": "sales",
    "electricity": "demand_mw",
}


def _clean_name(name: Any) -> str:
    """Normalize arbitrary header strings to clean snake_case."""
    s = str(name).strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).lower().strip("_")
    return re.sub(r"_+", "_", s)


def detect_target_unit(
    col_name: str | None,
    domain: str = "retail",
    user_unit: str | None = None,
) -> tuple[str, str, str]:
    """Detect target measurement unit, source of knowledge, and user-facing display label.
    
    Returns (unit, unit_source, unit_label):
    - unit: 'MW', 'kWh', 'MWh', 'GW', 'INR', 'USD', 'units', or 'Unspecified in source'
    - unit_source: 'user_confirmed', 'source_metadata', or 'unspecified'
    - unit_label: e.g. 'Demand (MW)', 'Electricity Demand (unit unspecified)', 'Sales (INR)', etc.
    """
    clean_dom = (domain or "retail").lower().strip()

    if user_unit and str(user_unit).strip():
        u = str(user_unit).strip()
        lbl = f"Demand ({u})" if clean_dom == "electricity" else f"Sales / Quantity ({u})"
        return u, "user_confirmed", lbl

    if not col_name:
        def_lbl = "Electricity Demand (unit unspecified)" if clean_dom == "electricity" else "Sales / Quantity (unit unspecified)"
        return "Unspecified in source", "unspecified", def_lbl

    clean = _clean_name(col_name)
    raw_lower = str(col_name).lower()

    if clean_dom == "electricity":
        # 1. Check explicit Megawatts markers in header
        if any(m in clean for m in ("load_mw", "demand_mw", "grid_load_mw", "_mw", "mw_", "megawatts", "in_mw")) or any(p in raw_lower for p in ("(mw)", "[mw]", " mw", "_mw", "megawatt")):
            return "MW", "source_metadata", "Demand (MW)"
        # 2. Check explicit Kilowatt-hours markers
        if any(m in clean for m in ("_kwh", "kwh_", "in_kwh", "kilowatt_hours", "kwh")) or any(p in raw_lower for p in ("(kwh)", "[kwh]", " kwh", "_kwh")):
            return "kWh", "source_metadata", "Demand (kWh)"
        # 3. Check explicit Megawatt-hours markers
        if any(m in clean for m in ("_mwh", "mwh_", "in_mwh")) or any(p in raw_lower for p in ("(mwh)", "[mwh]", " mwh", "_mwh")):
            return "MWh", "source_metadata", "Demand (MWh)"
        # 4. Check explicit Gigawatts markers
        if any(m in clean for m in ("_gw", "gw_", "in_gw")) or any(p in raw_lower for p in ("(gw)", "[gw]", " gw")):
            return "GW", "source_metadata", "Demand (GW)"

        # Default for electricity when unit is unverified in source column name
        return "Unspecified in source", "unspecified", "Electricity Demand (unit unspecified)"

    if clean_dom == "retail":
        if any(m in clean for m in ("_inr", "inr_", "price_inr", "sales_inr", "revenue_inr")) or any(p in raw_lower for p in ("(inr)", "[inr]", " inr", "₹", "rs")):
            return "INR", "source_metadata", "Sales (INR)"
        if any(m in clean for m in ("_usd", "usd_")) or any(p in raw_lower for p in ("(usd)", "[usd]", " usd", "$")):
            return "USD", "source_metadata", "Sales (USD)"
        if any(m in clean for m in ("_eur", "eur_")) or any(p in raw_lower for p in ("(eur)", "[eur]", " eur", "€")):
            return "EUR", "source_metadata", "Sales (EUR)"
        if any(m in clean for m in ("units_sold", "quantity", "qty", "item_count", "sales_units")):
            return "units", "source_metadata", "Quantity (units)"

        return "Unspecified in source", "unspecified", "Retail Sales (unit unspecified)"


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    """Statistical and semantic characteristics of a single column."""
    original_name: str
    clean_name: str
    inferred_type: str
    non_null_count: int
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    is_constant: bool
    is_numeric: bool
    is_temporal: bool
    sample_values: list[Any] = field(default_factory=list)
    numeric_variance: float = 0.0
    semantic_role: str = "unknown"
    confidence: float = 0.0
    role_reason: str = ""


@dataclass
class ResolvedSchema:
    """Authoritative resolved schema mapping between original and canonical spaces."""
    domain: str
    original_columns: list[str]
    time_column: str | None
    target_column: str | None
    canonical_target: str
    time_confidence: float
    target_confidence: float
    is_valid: bool
    is_ambiguous_target: bool
    ambiguous_targets: list[str] = field(default_factory=list)
    target_candidates: list[dict[str, Any]] = field(default_factory=list)
    time_candidates: list[dict[str, Any]] = field(default_factory=list)
    column_roles: dict[str, str] = field(default_factory=dict)
    canonical_mapping: dict[str, str] = field(default_factory=dict)
    reverse_mapping: dict[str, str] = field(default_factory=dict)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    geographic_columns: list[str] = field(default_factory=list)
    entity_columns: list[str] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    excluded_reasons: dict[str, str] = field(default_factory=dict)
    temporal_frequency: str = "unknown"
    unique_timestamps: int = 0
    duplicate_timestamps: int = 0
    is_panel: bool = False
    panel_entities: list[str] = field(default_factory=list)
    validation_issues: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    explanation: str = ""
    column_profiles: dict[str, Any] = field(default_factory=dict)
    is_multiseries: bool = False
    dataset_format: str = "tabular_single"
    detected_geographic_series: list[str] = field(default_factory=list)
    geographic_level: str | None = None
    parent_level: str | None = None
    detected_parent_totals: list[str] = field(default_factory=list)
    excluded_series_info: dict[str, str] = field(default_factory=dict)
    aggregation_coverage: dict[str, Any] = field(default_factory=dict)
    unit: str = "Unspecified in source"
    unit_source: str = "unspecified"
    unit_label: str = ""
    allow_aggregate: bool = True
    aggregation_disabled_reason: str = ""

    @property
    def frequency(self) -> str:
        """Alias for temporal_frequency."""
        return self.temporal_frequency

    @property
    def errors(self) -> list[str]:
        """Alias for validation_issues."""
        return self.validation_issues

    @property
    def is_chronological(self) -> bool:
        """Whether the detected time column is chronologically ordered."""
        if self.time_column and self.column_profiles:
            prof = self.column_profiles.get(self.time_column)
            if prof and hasattr(prof, "is_chronological"):
                return bool(prof.is_chronological)
        return True

    @property
    def target_profile(self) -> Any:
        """Profile metadata of the detected or chosen target column."""
        if self.target_column and self.column_profiles:
            prof = self.column_profiles.get(self.target_column)
            if isinstance(prof, dict):
                class _ProfWrapper(dict):
                    def __getattr__(self, name):
                        return self.get(name)
                return _ProfWrapper(prof)
            return prof
        return None

    @property
    def time_profile(self) -> Any:
        """Profile metadata of the detected or chosen timestamp column."""
        if self.time_column and self.column_profiles:
            prof = self.column_profiles.get(self.time_column)
            if isinstance(prof, dict):
                class _ProfWrapper(dict):
                    def __getattr__(self, name):
                        return self.get(name)
                return _ProfWrapper(prof)
            return prof
        return None

    def adapt_frame(
        self,
        df: pd.DataFrame,
        *,
        drop_excluded: bool = False,
        impute_target_time: bool = False,
        preserve_original: bool = False,
    ) -> pd.DataFrame:
        """Create an adapted DataFrame with canonical columns expected by internal models.
        
        Maps the identified timestamp and target to internal canonical roles
        ('timestamp' and 'demand_mw' / 'sales').
        If preserve_original=True, adds canonical alias columns without removing original columns.
        Handles both row-based and wide-format multi-series geographic datasets transparently.
        """
        # Wide-format geographic dataset adaptation
        if getattr(self, "is_multiseries", False):
            res = pd.DataFrame()
            # 1. Timestamp column
            if self.time_column and self.time_column in df.columns:
                res["timestamp"] = pd.to_datetime(df[self.time_column], errors="coerce")
            else:
                res["timestamp"] = pd.date_range("2024-01-01", periods=len(df), freq="D")

            # 2. Target variable adaptation
            canonical = self.canonical_target or ("demand_mw" if self.domain == "electricity" else "sales")
            target_col = getattr(self, "target_column", None)

            if target_col and target_col in df.columns:
                # Single location series selected
                s = pd.to_numeric(df[target_col], errors="coerce")
                # Bounded interpolation (limit=3, no silent 0 fill or indefinite forward fill)
                res[canonical] = s.interpolate(method="linear", limit=3).bfill(limit=2)
            else:
                # Aggregation requested: verify allow_aggregate
                if not getattr(self, "allow_aggregate", True):
                    reason = getattr(self, "aggregation_disabled_reason", "aggregation restricted due to incompatible series units or scale disparity")
                    raise ValueError(f"Geographic aggregation is unavailable for this dataset: {reason}")

                # Valid aggregate across detected compatible component series
                valid_series = [c for c in getattr(self, "detected_geographic_series", []) if c in df.columns]
                if valid_series:
                    comp_df = df[valid_series].apply(pd.to_numeric, errors="coerce")
                    # Transparent bounded interpolation per series (limit=3)
                    comp_interp = comp_df.interpolate(method="linear", limit=3).bfill(limit=2)
                    res[canonical] = comp_interp.sum(axis=1)
                elif target_col and target_col in df.columns:
                    res[canonical] = pd.to_numeric(df[target_col], errors="coerce").interpolate(method="linear", limit=3).bfill(limit=2)
                else:
                    res[canonical] = 0.0

            if preserve_original:
                out = df.copy()
                out["timestamp"] = res["timestamp"]
                out[canonical] = res[canonical]
                return out

            return res

        result = df.copy()

        # Handle column drop of excluded personal attributes if requested
        if drop_excluded and self.excluded_columns:
            drop_cols = [c for c in self.excluded_columns if c in result.columns and c not in (self.time_column, self.target_column)]
            if drop_cols:
                result = result.drop(columns=drop_cols)

        if preserve_original:
            if self.target_column and self.target_column in result.columns and self.canonical_target not in result.columns:
                result[self.canonical_target] = result[self.target_column]
            if self.time_column and self.time_column in result.columns and "timestamp" not in result.columns:
                result["timestamp"] = result[self.time_column]
            return result

        rename_dict: dict[str, str] = {}

        # 1. Target column adaptation
        if self.target_column and self.target_column in result.columns:
            if self.target_column != self.canonical_target:
                # If canonical target already exists in columns as a different column, protect it
                if self.canonical_target in result.columns and self.canonical_target != self.target_column:
                    result = result.rename(columns={self.canonical_target: f"{self.canonical_target}_orig"})
                rename_dict[self.target_column] = self.canonical_target

        # 2. Time column adaptation
        if self.time_column and self.time_column in result.columns:
            if self.time_column != "timestamp":
                if "timestamp" in result.columns and "timestamp" != self.time_column:
                    result = result.rename(columns={"timestamp": "timestamp_orig"})
                rename_dict[self.time_column] = "timestamp"

        if rename_dict:
            result = result.rename(columns=rename_dict)

        return result

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        d = asdict(self)
        # Ensure clean serialization
        return d


def adapt_frame(
    df: pd.DataFrame,
    schema: ResolvedSchema,
    *,
    drop_excluded: bool = False,
    impute_target_time: bool = False,
    preserve_original: bool = False,
) -> pd.DataFrame:
    """Convenience module helper to adapt DataFrame using a ResolvedSchema."""
    return schema.adapt_frame(
        df,
        drop_excluded=drop_excluded,
        impute_target_time=impute_target_time,
        preserve_original=preserve_original,
    )


# ---------------------------------------------------------------------------
# Feature Discovery Engine
# ---------------------------------------------------------------------------

def profile_columns(df: pd.DataFrame, domain: str = "retail") -> dict[str, ColumnProfile]:
    """Inspect every column in the dataframe across statistical and semantic dimensions."""
    profiles: dict[str, ColumnProfile] = {}
    n_rows = len(df)
    domain_clean = (domain or "retail").lower().strip()

    for col in df.columns:
        col_str = str(col)
        clean = _clean_name(col_str)
        series = df[col]
        non_null = series.dropna()
        non_null_count = len(non_null)
        missing_count = n_rows - non_null_count
        missing_ratio = float(missing_count / n_rows) if n_rows > 0 else 1.0
        unique_count = int(non_null.nunique()) if non_null_count > 0 else 0
        unique_ratio = float(unique_count / non_null_count) if non_null_count > 0 else 0.0
        is_const = bool(unique_count <= 1 and n_rows > 1)

        # Statistical type inference
        is_num = False
        is_temp = False
        inferred_type = "string"
        variance = 0.0

        if pd.api.types.is_numeric_dtype(series):
            is_num = True
            inferred_type = "integer" if pd.api.types.is_integer_dtype(series) else "float"
            if len(non_null) > 1:
                variance = float(non_null.var())
        elif pd.api.types.is_datetime64_any_dtype(series):
            is_temp = True
            inferred_type = "datetime"
        else:
            # Test numeric sample coercibility
            sample_head = non_null.head(100)
            if not sample_head.empty:
                coerced_num = pd.to_numeric(sample_head.astype(str).str.replace(",", "").str.strip(), errors="coerce")
                target_kw_match = any(kw in clean for kw in ("sales", "units", "demand", "load", "qty", "volume", "mw", "price", "count"))
                num_thresh = 0.40 if target_kw_match else 0.60
                if float(coerced_num.notna().mean()) >= num_thresh:
                    is_num = True
                    inferred_type = "numeric_coercible"
                    if coerced_num.dropna().shape[0] > 1:
                        variance = float(coerced_num.dropna().var())
                else:
                    # Test datetime sample coercibility
                    if clean not in PERSONAL_DATE_KEYWORDS:
                        try:
                            coerced_date = pd.to_datetime(sample_head, errors="coerce", format="mixed")
                            if float(coerced_date.notna().mean()) > 0.70:
                                is_temp = True
                                inferred_type = "datetime_coercible"
                        except Exception:
                            pass

        sample_vals = [val for val in non_null.head(5).tolist()]

        profiles[col_str] = ColumnProfile(
            original_name=col_str,
            clean_name=clean,
            inferred_type=inferred_type,
            non_null_count=non_null_count,
            missing_count=missing_count,
            missing_ratio=round(missing_ratio, 4),
            unique_count=unique_count,
            unique_ratio=round(unique_ratio, 4),
            is_constant=is_const,
            is_numeric=is_num,
            is_temporal=is_temp,
            sample_values=sample_vals,
            numeric_variance=round(variance, 4) if not math.isnan(variance) else 0.0,
        )

    return profiles


def detect_candidate_timestamps(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
) -> list[tuple[str, float, str]]:
    """Detect and rank candidate date/timestamp columns using multiple signals."""
    candidates: list[tuple[str, float, str]] = []

    for col_str, prof in profiles.items():
        clean = prof.clean_name
        clean_no_sep = clean.replace("_", "")

        # Personal dates (DOB, etc.) are excluded
        if clean in PERSONAL_DATE_KEYWORDS or any(kw in clean for kw in PERSONAL_DATE_KEYWORDS):
            continue

        score = 0.0
        reason_parts = []

        is_exact_time_kw = any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in TIME_KEYWORDS if kw != "year")
        is_sub_time_kw = any(kw in clean for kw in TIME_KEYWORDS if len(kw) >= 4 and kw != "year")

        if is_exact_time_kw:
            score += 0.55
            reason_parts.append(f"Name matched timestamp keyword '{clean}'")
        elif is_sub_time_kw:
            score += 0.35
            reason_parts.append(f"Substring matched timestamp keyword")
        elif clean == "year":
            score += 0.20
            reason_parts.append("Matched Year column")

        if prof.is_temporal:
            if prof.inferred_type == "datetime":
                score += 0.50
                reason_parts.append("Native datetime dtype")
            else:
                score += 0.40
                reason_parts.append("Successfully parsed datetime sample (>70% valid dates)")

        # Monotonicity check on non-null values
        if prof.is_temporal or is_exact_time_kw:
            sample_s = df[col_str].dropna().head(200)
            if not sample_s.empty:
                try:
                    parsed_s = pd.to_datetime(sample_s, errors="coerce")
                    valid_parsed = parsed_s.dropna()
                    if len(valid_parsed) > 5 and valid_parsed.is_monotonic_increasing:
                        score += 0.10
                        reason_parts.append("Monotonically increasing chronological order")
                except Exception:
                    pass

        # Check generic headers (e.g. Unnamed: 0, index) containing valid timestamps
        is_generic_header = clean in {"unnamed_0", "unnamed", "index", "col_0", "column_0"} or clean.startswith("unnamed")
        if is_generic_header and prof.is_temporal:
            score += 0.45
            reason_parts.append(f"Generic header '{col_str}' contains valid chronological timestamps")
        elif is_generic_header and prof.inferred_type == "integer":
            score -= 0.60

        # Penalize integer IDs masquerading as timestamps
        if prof.inferred_type == "integer" and clean not in {"year", "date", "time", "datetime", "timestamp"}:
            score -= 0.30

        if score >= 0.35:
            candidates.append((col_str, min(1.0, round(score, 2)), "; ".join(reason_parts)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def detect_candidate_targets(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    domain: str = "retail",
) -> list[tuple[str, float, str]]:
    """Detect and rank candidate forecasting target columns using multiple signals."""
    domain_clean = (domain or "retail").lower().strip()
    target_kws = RETAIL_TARGET_KEYWORDS if domain_clean == "retail" else ELECTRICITY_TARGET_KEYWORDS
    target_regex = RETAIL_TARGET_REGEX if domain_clean == "retail" else ELECTRICITY_TARGET_REGEX
    n_rows = len(df)

    candidates: list[tuple[str, float, str]] = []

    for col_str, prof in profiles.items():
        clean = prof.clean_name
        clean_no_sep = clean.replace("_", "")

        # Target must be numeric (or coercible to numeric)
        if not prof.is_numeric:
            continue

        # Zero variance (constant target) is disqualified from time-series forecasting
        if n_rows > 1 and (prof.is_constant or prof.numeric_variance <= 1e-8):
            continue

        # High missingness (> 50%) disqualified
        if prof.missing_ratio > 0.50:
            continue

        score = 0.25
        reason_parts = [f"Numeric column with positive variance (var={prof.numeric_variance:.2f})"]

        # Check keyword matches
        exact_kw = any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in target_kws)
        regex_kw = bool(target_regex.search(clean))
        sub_kw = any(kw in clean for kw in target_kws if len(kw) >= 3)

        if exact_kw:
            score += 0.65
            reason_parts.append(f"Exact match for {domain_clean} target keyword '{clean}'")
        elif regex_kw:
            score += 0.50
            reason_parts.append(f"Regex pattern matched {domain_clean} demand/sales unit")
        elif sub_kw:
            score += 0.35
            reason_parts.append(f"Matched substring {domain_clean} target keyword")

        # Domain-specific bonuses
        if domain_clean == "electricity":
            # Primary electricity indicators: mw, kwh, load, demand, power
            if any(unit in clean for unit in ("_mw", "mw_", "mw", "_kwh", "kwh", "_mwh", "mwh", "grid_load", "demand_met")):
                score += 0.15
                reason_parts.append("Matched electricity measurement unit (MW/kWh/MWh)")
            if clean in {"demand_mw", "load_mw", "grid_load_mw", "hourly_demand_met_mw", "hourly_demand_met"}:
                score += 0.10
        elif domain_clean == "retail":
            # Primary retail indicators: sales, units_sold, quantity
            if clean in {"sales", "units_sold", "quantity", "qty", "sales_units", "quantity_sold"}:
                score += 0.15
                reason_parts.append("Matched primary retail target concept")

        # Penalize ID-like names
        is_id = any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in ENTITY_ID_KEYWORDS) or (
            any(kw in clean for kw in {"id", "code", "no", "number", "index", "sku"}) and not exact_kw
        )
        if is_id:
            score -= 0.55
            reason_parts.append("Penalized for identifier/code naming convention")

        # Penalize high-uniqueness integer columns without target keywords (e.g. auto-increment IDs)
        if prof.unique_ratio > 0.90 and prof.inferred_type == "integer" and n_rows > 50 and not exact_kw and not regex_kw:
            score -= 0.40
            reason_parts.append("Penalized for high-cardinality integer pattern")

        # Price / Cost / Rate penalty for retail target (price is an exogenous predictor, not sales)
        if domain_clean == "retail" and any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in PRICE_KEYWORDS):
            score -= 0.35
            reason_parts.append("Identified as price/cost predictor rather than target volume")

        if score >= 0.35:
            candidates.append((col_str, min(1.0, round(score, 2)), "; ".join(reason_parts)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def detect_frequency_and_temporal_stats(
    df: pd.DataFrame,
    time_col: str | None,
) -> tuple[str, int, int, bool]:
    """Inspect temporal frequency, unique timestamps, duplicates, and sorting."""
    if not time_col or time_col not in df.columns or df.empty:
        return "unknown", 0, 0, False

    parsed = pd.to_datetime(df[time_col], errors="coerce")
    valid = parsed.dropna()
    total_valid = len(valid)
    if total_valid == 0:
        return "unknown", 0, 0, False

    unique_ts = int(valid.nunique())
    dup_ts = total_valid - unique_ts
    is_chronological = bool(valid.is_monotonic_increasing)

    detected_freq = "unknown"
    if unique_ts > 2:
        sorted_ts = valid.drop_duplicates().sort_values()
        diffs = sorted_ts.diff().dropna()
        mode_diff = diffs.mode()
        if not mode_diff.empty:
            td = mode_diff.iloc[0]
            secs = td.total_seconds()
            if 3500 <= secs <= 3700:
                detected_freq = "hourly"
            elif 86000 <= secs <= 87000:
                detected_freq = "daily"
            elif 600000 <= secs <= 610000:
                detected_freq = "weekly"
            elif 2400000 <= secs <= 2700000:
                detected_freq = "monthly"
            else:
                detected_freq = str(td)
        else:
            detected_freq = "irregular"

    return detected_freq, unique_ts, dup_ts, is_chronological


def detect_wide_geographic_series(
    df: pd.DataFrame,
    profiles: dict[str, ColumnProfile],
    time_col: str | None,
    domain: str = "electricity",
) -> dict[str, Any]:
    """Detect independent geographic time series formatted as separate numeric columns.

    Guarantees:
    - Never hardcodes specific countries, states, or location lists.
    - Excludes index columns (e.g. 'Unnamed: 0', auto-increment integers) and time columns.
    - Excludes non-geographic metrics (temperature, price, promo, holiday, identifiers).
    - Detects parent-level totals (via name keyword or mathematical scale check where column mean approx sum of child means) to prevent double-counting.
    - Validates unit and scale compatibility across component series.
    - Transparently measures missingness and reports coverage percentage.
    """
    n_rows = len(df)
    if n_rows < 5 or len(df.columns) < 3:
        return {
            "is_multiseries": False,
            "dataset_format": "tabular_single",
            "geographic_level": None,
            "parent_level": None,
            "component_series": [],
            "parent_totals": [],
            "excluded_series": {},
            "coverage_meta": {},
            "allow_aggregate": False,
        }

    # If the dataset contains row-based non-numeric geographic columns (e.g. City, State, Country),
    # it is row-based and must not be treated as wide-multiseries.
    row_geo_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
        and (
            any(_clean_name(c) == kw for kw in GEOGRAPHIC_KEYWORDS)
            or any(kw in _clean_name(c) for kw in ("country", "state", "city", "region", "district", "location", "zone"))
        )
    ]
    if row_geo_cols:
        return {
            "is_multiseries": False,
            "dataset_format": "tabular_single",
            "geographic_level": None,
            "parent_level": None,
            "component_series": [],
            "parent_totals": [],
            "excluded_series": {},
            "coverage_meta": {},
            "allow_aggregate": False,
        }

    excluded_info: dict[str, str] = {}
    candidate_cols: list[str] = []

    NON_GEO_KEYWORDS = {
        "temp", "temperature", "humidity", "wind", "pressure", "rainfall", "rain",
        "price", "cost", "fee", "mrp", "rate", "holiday", "promo", "discount",
        "order_id", "customer_id", "sku", "code", "index", "row_id", "unnamed",
        "cpi", "ewma", "lag", "rolling", "mean", "std", "min", "max", "var", "diff",
        "sales", "revenue", "demand", "quantity", "qty", "count", "day", "month",
        "year", "week", "hour", "minute", "dayofweek", "weekday", "weekend", "is_weekend"
    }

    for col_str, prof in profiles.items():
        if col_str == time_col:
            continue

        clean = prof.clean_name
        clean_low = col_str.strip().lower()

        # 1. Exclude index columns
        if (
            clean_low.startswith("unnamed")
            or clean in ("index", "id", "row_id", "level_0", "record_id")
            or re.match(r"^unnamed[_\s:]*\d+$", clean_low)
        ):
            excluded_info[col_str] = "Identified as row index / identifier column"
            continue

        if prof.inferred_type == "integer" and prof.unique_ratio > 0.95:
            s_num = pd.to_numeric(df[col_str], errors="coerce").dropna()
            if len(s_num) > 10 and (
                (s_num.min() == 0 and s_num.max() == len(s_num) - 1)
                or (s_num.min() == 1 and s_num.max() == len(s_num))
            ):
                excluded_info[col_str] = "Identified as sequential row index integer"
                continue

        # 2. Exclude non-numeric or constant columns
        if not prof.is_numeric:
            excluded_info[col_str] = "Non-numeric column"
            continue
        if prof.is_constant or prof.numeric_variance <= 0:
            excluded_info[col_str] = "Constant column with zero variance"
            continue

        # 3. Exclude non-geographic metrics
        is_non_geo = any(clean == k or clean.startswith(f"{k}_") or clean.endswith(f"_{k}") for k in NON_GEO_KEYWORDS)
        if is_non_geo:
            excluded_info[col_str] = "Non-geographic physical metric / parameter"
            continue

        # 4. Canonical target name in already formatted tabular single datasets
        if domain == "electricity" and clean in {"demand_mw", "load_mw", "grid_load_mw"}:
            continue

        candidate_cols.append(col_str)

    if len(candidate_cols) < 2:
        return {
            "is_multiseries": False,
            "dataset_format": "tabular_single",
            "geographic_level": None,
            "parent_level": None,
            "component_series": [],
            "parent_totals": [],
            "excluded_series": excluded_info,
            "coverage_meta": {},
            "allow_aggregate": False,
        }

    # Classify candidate columns via LLM routing / deterministic fallback
    llm_map: dict[str, dict[str, Any]] = {}
    try:
        from backend.app.services.llm_service import classify_geographic_columns_llm
        llm_map = classify_geographic_columns_llm(candidate_cols)
    except Exception:
        llm_map = {}

    geo_units: list[str] = []
    parent_totals: list[str] = []
    detected_levels: list[str] = []

    TOTAL_KWS = {"total", "sum", "all", "overall", "aggregate", "national", "grand_total", "all_india", "us_total", "world_total"}

    means = {}
    for col in candidate_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        means[col] = float(s.mean()) if not s.empty else 0.0

    for col in candidate_cols:
        clean = _clean_name(col)
        meta = llm_map.get(col, {})
        is_geo = meta.get("is_geographic", True)
        geo_level = meta.get("geographic_level", "state")

        # Check total / aggregate keywords
        is_total_kw = clean in TOTAL_KWS or clean.startswith("total_") or clean.endswith("_total")
        if is_total_kw or geo_level == "aggregate_total":
            parent_totals.append(col)
            excluded_info[col] = "Parent aggregate total column (excluded from child summation to prevent double-counting)"
            continue

        if not is_geo:
            excluded_info[col] = "Classified as non-geographic parameter"
            continue

        geo_units.append(col)
        if geo_level and geo_level not in ("none", "aggregate_total"):
            detected_levels.append(geo_level)

    # Scale check for parent-level totals:
    # If a column's mean is approximately equal to the sum of the means of the other candidate series (+- 15%),
    # it is a pre-aggregated total!
    sum_child_means = sum(means[c] for c in geo_units)
    verified_children: list[str] = []
    for c in geo_units:
        c_mean = means[c]
        other_means_sum = sum_child_means - c_mean
        if other_means_sum > 0 and abs(c_mean - other_means_sum) / other_means_sum <= 0.15:
            parent_totals.append(c)
            excluded_info[c] = "Parent aggregate total column detected by mathematical scale check (preventing double-counting)"
        else:
            verified_children.append(c)

    if len(verified_children) < 2:
        return {
            "is_multiseries": False,
            "dataset_format": "tabular_single",
            "geographic_level": None,
            "parent_level": None,
            "component_series": [],
            "parent_totals": parent_totals,
            "excluded_series": excluded_info,
            "coverage_meta": {},
            "allow_aggregate": False,
            "aggregation_disabled_reason": "Fewer than 2 independent geographic series detected",
            "unit": "Unspecified in source",
            "unit_source": "unspecified",
            "unit_label": "Electricity Demand (unit unspecified)" if domain == "electricity" else "Sales / Quantity (unit unspecified)",
        }

    # Unit compatibility check across child series
    allow_aggregate = True
    aggregation_disabled_reason = ""
    dataset_unit = "Unspecified in source"
    dataset_unit_source = "unspecified"
    dataset_unit_label = "Electricity Demand (unit unspecified)" if domain == "electricity" else "Sales / Quantity (unit unspecified)"

    child_units = {}
    verified_unit_names = set()
    for c in verified_children:
        u, u_src, u_lbl = detect_target_unit(c, domain=domain)
        child_units[c] = (u, u_src, u_lbl)
        if u_src in ("source_metadata", "user_confirmed"):
            verified_unit_names.add(u)

    if len(verified_unit_names) > 1:
        # Incompatible verified units across candidate series
        allow_aggregate = False
        aggregation_disabled_reason = f"Conflicting measurement units ({', '.join(sorted(verified_unit_names))}) detected across series. Incompatible units cannot be summed."
    elif len(verified_unit_names) == 1:
        dataset_unit = next(iter(verified_unit_names))
        dataset_unit_source = "source_metadata"
        dataset_unit_label = f"Demand ({dataset_unit})" if domain == "electricity" else f"Sales ({dataset_unit})"

    # Scale compatibility check across child series
    valid_means = [means[c] for c in verified_children if means[c] > 0]
    if valid_means and allow_aggregate:
        min_m, max_m = min(valid_means), max(valid_means)
        if min_m > 0 and (max_m / min_m) > 1000:
            allow_aggregate = False
            aggregation_disabled_reason = f"Extreme scale disparity (ratio {max_m / min_m:.1f}x > 1000x) across geographic series indicates differing measurement schemes or mixed physical quantities."

    # Determine dominant geographic level
    primary_level = "state"
    if detected_levels:
        from collections import Counter
        primary_level = Counter(detected_levels).most_common(1)[0][0]

    parent_level = "country" if primary_level in ("state", "province") else ("state" if primary_level == "city" else "national")

    # Coverage metrics
    total_cells = len(df) * len(verified_children)
    missing_cells = int(df[verified_children].isna().sum().sum())
    coverage_pct = round(100.0 * (1.0 - (missing_cells / max(1, total_cells))), 2)

    return {
        "is_multiseries": True,
        "dataset_format": "wide_multiseries",
        "geographic_level": primary_level,
        "parent_level": parent_level,
        "component_series": sorted(verified_children),
        "parent_totals": sorted(parent_totals),
        "excluded_series": excluded_info,
        "coverage_meta": {
            "coverage_pct": coverage_pct,
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "included_count": len(verified_children),
            "assumptions": f"Sum across compatible {primary_level} series with bounded linear interpolation" if allow_aggregate else f"Individual series forecasting only (aggregation disabled: {aggregation_disabled_reason})",
        },
        "unit": dataset_unit,
        "unit_source": dataset_unit_source,
        "unit_label": dataset_unit_label,
        "allow_aggregate": allow_aggregate,
        "aggregation_disabled_reason": aggregation_disabled_reason,
    }


def discover_features_and_resolve_schema(
    df: pd.DataFrame,
    domain: str = "retail",
    explicit_target: str | None = None,
    explicit_time_col: str | None = None,
    explicit_mapping: dict[str, str] | None = None,
    strict: bool = False,
) -> ResolvedSchema:
    """Centralized schema resolution and automatic feature-role discovery.
    
    Inspects:
    - Column names, aliases, types, sample values, missingness, and variance.
    - Timestamp parsing, temporal frequency, duplicates, and ordering.
    - Numeric target candidates, variance, and ambiguity.
    - Geographic and entity identifier candidates.
    - Excluded personal attributes, row indices, and leakage columns.
    
    Returns authoritative ResolvedSchema without destructive renaming.
    """
    domain_clean = (domain or "retail").lower().strip()
    original_cols = [str(c) for c in df.columns]
    n_rows = len(df)
    n_cols = len(original_cols)

    validation_issues: list[str] = []
    validation_warnings: list[str] = []

    # 1. Profile all columns
    profiles = profile_columns(df, domain=domain_clean)

    # 2. Time column discovery & resolution
    time_candidates = detect_candidate_timestamps(df, profiles)
    selected_time: str | None = None
    time_conf = 0.0

    if explicit_time_col and explicit_time_col in original_cols:
        # User explicit selection takes absolute precedence
        prof = profiles[explicit_time_col]
        # Validate that explicit selection can actually parse dates
        sample = df[explicit_time_col].dropna().head(50)
        parsed_ok = True
        if not sample.empty:
            try:
                parsed_sample = pd.to_datetime(sample, errors="coerce")
                if float(parsed_sample.notna().mean()) < 0.50:
                    parsed_ok = False
            except Exception:
                parsed_ok = False
        if parsed_ok:
            selected_time = explicit_time_col
            time_conf = 1.0
        else:
            validation_issues.append(
                f"Explicitly selected timestamp column '{explicit_time_col}' could not be parsed as valid dates."
            )
    elif time_candidates:
        selected_time, time_conf, _ = time_candidates[0]

    if not selected_time:
        validation_issues.append(
            "No valid chronological date/time column detected in dataset. "
            "Forecasting requires a date/time index column (e.g. Datetime, Date, timestamp)."
        )

    # 3. Target column discovery, resolution & ambiguity handling
    is_multiseries = False
    dataset_format = "tabular_single"
    detected_geographic_series: list[str] = []
    geographic_level: str | None = None
    parent_level: str | None = None
    detected_parent_totals: list[str] = []
    excluded_series_info: dict[str, str] = {}
    aggregation_coverage: dict[str, Any] = {}

    target_candidates = detect_candidate_targets(df, profiles, domain=domain_clean)
    selected_target: str | None = None
    target_conf = 0.0
    is_ambiguous = False
    ambiguous_targets: list[str] = []

    canonical_target = CANONICAL_TARGETS.get(domain_clean, "target")

    # Inspect for wide-format geographic multi-series
    wide_geo_meta = detect_wide_geographic_series(df, profiles, selected_time, domain=domain_clean)
    allow_aggregate = True
    aggregation_disabled_reason = ""
    unit = "Unspecified in source"
    unit_source = "unspecified"
    unit_label = "Electricity Demand (unit unspecified)" if domain_clean == "electricity" else "Sales / Quantity (unit unspecified)"

    if wide_geo_meta["is_multiseries"]:
        is_multiseries = True
        dataset_format = "wide_multiseries"
        detected_geographic_series = wide_geo_meta["component_series"]
        geographic_level = wide_geo_meta["geographic_level"]
        parent_level = wide_geo_meta["parent_level"]
        detected_parent_totals = wide_geo_meta["parent_totals"]
        excluded_series_info = wide_geo_meta["excluded_series"]
        aggregation_coverage = wide_geo_meta["coverage_meta"]
        allow_aggregate = wide_geo_meta.get("allow_aggregate", True)
        aggregation_disabled_reason = wide_geo_meta.get("aggregation_disabled_reason", "")
        unit = wide_geo_meta.get("unit", "Unspecified in source")
        unit_source = wide_geo_meta.get("unit_source", "unspecified")
        unit_label = wide_geo_meta.get("unit_label", "Electricity Demand (unit unspecified)")

    aggregate_label = f"All {str(geographic_level or 'state').replace('_', ' ').title()}s (Aggregate)"

    if explicit_target and explicit_target in original_cols:
        # User explicit selection takes absolute precedence
        prof = profiles[explicit_target]
        if not prof.is_numeric:
            validation_issues.append(
                f"Selected target column '{explicit_target}' contains non-numeric data. "
                "Forecasting target must be numeric."
            )
        elif n_rows > 1 and (prof.is_constant or prof.numeric_variance <= 1e-8):
            validation_issues.append(
                f"Selected target column '{explicit_target}' has zero variance (constant value). "
                "Time-series forecasting requires a varying target."
            )
        elif prof.missing_ratio > 0.50:
            validation_issues.append(
                f"Selected target column '{explicit_target}' has {prof.missing_ratio*100:.1f}% missing values (exceeds 50% limit)."
            )
        else:
            selected_target = explicit_target
            target_conf = 1.0
            is_ambiguous = False
            unit, unit_source, unit_label = detect_target_unit(explicit_target, domain=domain_clean)
    elif (explicit_target in ("__aggregate__", "aggregate") or (explicit_target and str(explicit_target).strip().endswith("(Aggregate)"))) and is_multiseries:
        selected_target = aggregate_label
        target_conf = 0.95
        is_ambiguous = False
        if not allow_aggregate:
            validation_warnings.append(
                f"Geographic aggregation is unavailable: {aggregation_disabled_reason}. Please select an individual series."
            )
    else:
        # Automatic inference with strict ambiguity safety
        if not target_candidates:
            if is_multiseries:
                if allow_aggregate:
                    selected_target = aggregate_label
                    target_conf = 0.95
                else:
                    selected_target = detected_geographic_series[0] if detected_geographic_series else None
                    target_conf = 0.85
                    validation_warnings.append(
                        f"Geographic aggregation is unavailable: {aggregation_disabled_reason}. Selected individual series '{selected_target}'."
                    )
                    if selected_target:
                        unit, unit_source, unit_label = detect_target_unit(selected_target, domain=domain_clean)
                is_ambiguous = False
            else:
                # Check if all numeric columns were constant
                constant_nums = [c for c, p in profiles.items() if p.is_numeric and p.is_constant and c != selected_time] if n_rows > 1 else []
                if constant_nums:
                    validation_issues.append(
                        f"Target column candidate '{constant_nums[0]}' has zero variance (constant value). "
                        "Forecasting requires a varying numerical target."
                    )
                else:
                    validation_issues.append(
                        f"No valid numeric target column detected in dataset for {domain_clean.capitalize()} domain."
                    )
        elif len(target_candidates) == 1:
            selected_target, target_conf, _ = target_candidates[0]
            is_ambiguous = False
        else:
            # Check score gap between top two candidates
            top_name, top_score, _ = target_candidates[0]
            second_name, second_score, _ = target_candidates[1]

            # If top candidate is a decisive winner (score >= 0.70 and margin >= 0.20), resolve automatically
            if top_score >= 0.70 and (top_score - second_score) >= 0.20:
                selected_target = top_name
                target_conf = top_score
                is_ambiguous = False
            else:
                # Ambiguous: multiple competing plausible target variables
                selected_target = top_name  # Best guess default, but flagged
                target_conf = top_score
                is_ambiguous = True
                ambiguous_targets = [c[0] for c in target_candidates if c[1] >= 0.45]
                validation_warnings.append(
                    f"Multiple plausible target columns detected: {ambiguous_targets}. "
                    f"Please confirm or select which variable to forecast."
                )

    if selected_target and not (is_multiseries and selected_target == aggregate_label):
        unit, unit_source, unit_label = detect_target_unit(selected_target, domain=domain_clean)

    # 4. Temporal Profile
    temporal_freq, unique_ts, dup_ts, is_chronological = detect_frequency_and_temporal_stats(df, selected_time)

    if dup_ts > 0 and n_rows > 0:
        validation_warnings.append(
            f"Dataset contains {dup_ts} duplicate timestamp observations. "
            "They will be aggregated deterministically to forecasting frequency."
        )

    if n_rows >= 30 and unique_ts > 0 and unique_ts < 5:
        validation_issues.append(
            "Timestamp column contains insufficient unique temporal observations (< 5 unique timestamps)."
        )

    # 5. Role Classification across all columns
    column_roles: dict[str, str] = {}
    canonical_mapping: dict[str, str] = {}
    reverse_mapping: dict[str, str] = {}
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    geographic_cols: list[str] = []
    entity_cols: list[str] = []
    excluded_cols: list[str] = []
    excluded_reasons: dict[str, str] = {}

    leakage_set = LEAKAGE_COLUMNS.get(domain_clean, set())

    for col_str, prof in profiles.items():
        clean = prof.clean_name
        clean_no_sep = clean.replace("_", "")

        # 1. Primary Timestamp Role
        if col_str == selected_time:
            column_roles[col_str] = "timestamp"
            canonical_mapping[col_str] = "timestamp"
            reverse_mapping["timestamp"] = col_str
            continue

        # 2. Primary Target Role
        if col_str == selected_target:
            column_roles[col_str] = "target"
            canonical_mapping[col_str] = canonical_target
            reverse_mapping[canonical_target] = col_str
            continue

        # 3. Leakage Exclusion (e.g. current-period revenue when forecasting sales)
        if clean in leakage_set:
            excluded_cols.append(col_str)
            excluded_reasons[col_str] = f"Potential target leakage variable ({clean}) excluded from predictors."
            column_roles[col_str] = "excluded_leakage"
            continue

        # 4. Personal Information & Index Exclusion
        if any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in EXCLUDED_PERSONAL_OR_IDENTIFIER_KEYWORDS):
            excluded_cols.append(col_str)
            excluded_reasons[col_str] = "Personal identifier or row index excluded from model predictors."
            column_roles[col_str] = "excluded_identifier"
            continue

        # 5. Constant Columns Exclusion
        if prof.is_constant:
            excluded_cols.append(col_str)
            excluded_reasons[col_str] = "Zero-variance constant column excluded from predictors."
            column_roles[col_str] = "excluded_constant"
            continue

        # 6. Geographic Columns
        is_geo = any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in GEOGRAPHIC_KEYWORDS) or any(
            kw in clean for kw in ("country", "state", "city", "region", "district", "location", "zone")
        )
        if is_geo:
            geographic_cols.append(col_str)
            column_roles[col_str] = "geographic"
            # Map common geographic names to canonical equivalents
            if clean in {"city", "town", "city_name"} and "city" not in canonical_mapping.values():
                canonical_mapping[col_str] = "city"
                reverse_mapping["city"] = col_str
            elif clean in {"state", "province", "state_name"} and "state" not in canonical_mapping.values():
                canonical_mapping[col_str] = "state"
                reverse_mapping["state"] = col_str
            elif clean in {"region", "zone", "territory"} and "region" not in canonical_mapping.values():
                canonical_mapping[col_str] = "region"
                reverse_mapping["region"] = col_str
            continue

        # 7. Entity Identifiers
        is_entity = any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in ENTITY_ID_KEYWORDS)
        if is_entity:
            entity_cols.append(col_str)
            column_roles[col_str] = "entity_id"
            if clean in {"store_id", "store", "store_code"} and "store_id" not in canonical_mapping.values():
                canonical_mapping[col_str] = "store_id"
                reverse_mapping["store_id"] = col_str
            elif clean in {"product_id", "item_id", "sku", "product"} and "product_id" not in canonical_mapping.values():
                canonical_mapping[col_str] = "product_id"
                reverse_mapping["product_id"] = col_str
            continue

        # 8. Numeric Predictors
        if prof.is_numeric:
            numeric_cols.append(col_str)
            column_roles[col_str] = "numeric_predictor"
            # Common retail predictor mapping
            if clean in {"price", "unit_price", "base_price", "selling_price", "base_price_inr"}:
                canonical_mapping[col_str] = "price"
                reverse_mapping["price"] = col_str
            continue

        # 9. Categorical Predictors
        if prof.unique_ratio > 0.85 and n_rows > 30 and col_str not in geographic_cols:
            excluded_cols.append(col_str)
            excluded_reasons[col_str] = "High uniqueness text/identifier excluded to prevent overfitting."
            column_roles[col_str] = "excluded_high_cardinality"
        else:
            categorical_cols.append(col_str)
            column_roles[col_str] = "categorical_predictor"
            if clean in {"category", "product_category"} and "category" not in canonical_mapping.values():
                canonical_mapping[col_str] = "category"
                reverse_mapping["category"] = col_str

    # Apply explicit mapping overrides if provided
    if explicit_mapping and isinstance(explicit_mapping, dict):
        for orig, canon in explicit_mapping.items():
            if orig in original_cols:
                canonical_mapping[orig] = canon
                reverse_mapping[canon] = orig

    # 6. Panel vs Single Series Detection
    is_panel = False
    panel_entities: list[str] = []
    if entity_cols and dup_ts > 0:
        for ent in entity_cols:
            if profiles[ent].unique_count > 1:
                panel_entities.append(ent)
        if panel_entities:
            is_panel = True

    # 7. Final Validity Decision
    is_valid = len(validation_issues) == 0 and selected_time is not None and (selected_target is not None or is_multiseries)

    if not is_valid:
        explanation = f"Dataset validation failed for {domain_clean.capitalize()} forecasting: " + " ".join(validation_issues)
    elif is_multiseries and (selected_target == aggregate_label or not selected_target):
        explanation = (
            f"Dataset adaptive schema: Wide-format geographic dataset detected with {len(detected_geographic_series)} {geographic_level} series. "
            f"National/parent aggregate forecasting ({parent_level.title()}) and individual {geographic_level} forecasting enabled."
        )
    elif is_ambiguous:
        explanation = f"Ambiguous target selection: Dataset has multiple plausible target variables ({', '.join(ambiguous_targets)}). Please confirm target selection."
    elif validation_warnings:
        explanation = f"Dataset is suitable with warnings for {domain_clean.capitalize()} forecasting: " + " ".join(validation_warnings)
    else:
        explanation = f"Dataset schema resolved successfully for {domain_clean.capitalize()} forecasting (Target: '{selected_target}' → '{canonical_target}', Time: '{selected_time}' → 'timestamp')."

    # Compile candidate summaries for API/UI presentation
    target_cand_summary = [
        {
            "name": name,
            "confidence": score,
            "reason": reason,
            "variance": profiles[name].numeric_variance,
            "sample_values": profiles[name].sample_values,
        }
        for name, score, reason in target_candidates
    ]

    time_cand_summary = [
        {
            "name": name,
            "confidence": score,
            "reason": reason,
            "sample_values": profiles[name].sample_values,
        }
        for name, score, reason in time_candidates
    ]

    return ResolvedSchema(
        domain=domain_clean,
        original_columns=original_cols,
        time_column=selected_time,
        target_column=selected_target,
        canonical_target=canonical_target,
        time_confidence=time_conf,
        target_confidence=target_conf,
        is_valid=is_valid,
        is_ambiguous_target=is_ambiguous,
        ambiguous_targets=ambiguous_targets,
        target_candidates=target_cand_summary,
        time_candidates=time_cand_summary,
        column_roles=column_roles,
        canonical_mapping=canonical_mapping,
        reverse_mapping=reverse_mapping,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        geographic_columns=geographic_cols,
        entity_columns=entity_cols,
        excluded_columns=excluded_cols,
        excluded_reasons=excluded_reasons,
        temporal_frequency=temporal_freq,
        unique_timestamps=unique_ts,
        duplicate_timestamps=dup_ts,
        is_panel=is_panel,
        panel_entities=panel_entities,
        validation_issues=validation_issues,
        validation_warnings=validation_warnings,
        explanation=explanation,
        column_profiles={c: asdict(p) for c, p in profiles.items()},
        is_multiseries=is_multiseries,
        dataset_format=dataset_format,
        detected_geographic_series=detected_geographic_series,
        geographic_level=geographic_level,
        parent_level=parent_level,
        detected_parent_totals=detected_parent_totals,
        excluded_series_info=excluded_series_info,
        aggregation_coverage=aggregation_coverage,
        unit=unit,
        unit_source=unit_source,
        unit_label=unit_label,
        allow_aggregate=allow_aggregate,
        aggregation_disabled_reason=aggregation_disabled_reason,
    )
