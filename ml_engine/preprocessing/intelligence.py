"""Dataset intelligence, universal semantic feature matching, forecasting suitability analysis, feature governance, and domain profiling."""

from __future__ import annotations

import difflib
import math
import re
from typing import Any

import numpy as np
import pandas as pd


TIME_NAME_KEYWORDS = {
    "date", "datetime", "timestamp", "time", "day", "month", "year",
    "invoice_date", "order_date", "transaction_date", "purchase_date",
    "sales_date", "created_at", "updated_at", "event_time", "event_date",
    "invoicedate", "orderdate", "transactiondate", "date_time", "datehour",
    "date_hour", "period", "timestamp_utc", "time_stamp", "ds",
}

RETAIL_TARGET_KEYWORDS = {
    "sales", "quantity", "qty", "units", "units_sold", "quantity_sold",
    "sales_units", "demand", "demand_units", "orders", "order_count",
    "revenue", "amount", "sale_amount", "volume", "sales_revenue", "total_sales",
    "item_count", "item_qty", "units_sold_qty",
}

ELECTRICITY_TARGET_KEYWORDS = {
    "load", "demand", "demand_mw", "power", "consumption", "energy", "usage",
    "grid_load", "grid_load_mw", "hourly_demand_met_mw", "hourly_demand_met",
    "load_mw", "demand_met", "mw", "kwh", "mwh", "energy_consumption",
}

PRICE_KEYWORDS = {
    "price", "unit_price", "selling_price", "cost", "rate", "mrp", "base_price",
    "unit_cost", "base_price_inr", "fee", "cpi_combined", "cpi_food", "wpi_index",
    "fuel_price_petrol", "msrp",
}

PROMO_KEYWORDS = {
    "promotion", "promo", "discount", "offer", "coupon", "is_promo", "holiday",
    "discount_pct", "discount_percentage", "discount_amount", "deal_size", "offer_pct",
    "event", "festival", "festival_flag",
}

GEOGRAPHIC_KEYWORDS = {
    "country", "country_name", "country_code", "nation", "state", "state_name",
    "province", "district", "district_name", "city", "city_name", "town",
    "region", "location", "place", "area", "zone", "territory", "county", "zipcode",
    "zip_code", "postal_code", "pincode", "pin_code", "postcode", "site", "branch",
    "store", "store_location", "market", "depot", "hub", "terminal", "facility",
    "station", "address",
}


ID_KEYWORDS = {
    "id", "no", "number", "code", "sku", "key", "index", "customer", "invoice",
    "stock", "product", "store", "item", "user", "order_id", "invoice_no",
    "stock_code", "customer_id", "invoiceno", "stockcode", "customerid",
}

EXCLUDED_PERSONAL_OR_IDENTIFIER_KEYWORDS = {
    "customer_id", "customer_name", "first_name", "last_name", "name",
    "date_of_birth", "dob", "birth_date", "birthdate", "phone", "mobile",
    "email", "address", "postal_code", "zipcode", "zip_code", "pincode",
    "order_id", "transaction_id", "invoice_no", "invoice_id", "product_id",
    "product_name", "sku_id", "user_id", "index", "unnamed_0", "uuid", "row_id",
    "ssn", "credit_card", "customerid", "invoiceno", "stockcode", "orderid",
    "order_date", "orderdate", "ship_date", "shipdate", "purchase_date",
    "profit", "net_profit", "margin", "gross_profit", "quantity", "qty"
}



SEMANTIC_FAMILIES = {
    "date_time": {
        "keywords": TIME_NAME_KEYWORDS,
        "description": "Date / Timestamp temporal index",
    },
    "sales_revenue": {
        "keywords": {"sales", "revenue", "amount", "sales_amount", "sale_amount", "total_sales", "grand_total", "order_value", "billing_amount", "gross_amount", "turnover", "sales_revenue", "revenue_inr", "price_total"},
        "description": "Monetary Sales & Revenue",
    },
    "units_quantity": {
        "keywords": {"units_sold", "quantity", "qty", "demand", "units", "volume", "order_quantity", "item_count", "item_qty", "sales_units", "count", "units_sold_qty"},
        "description": "Volume & Units Sold",
    },
    "power_energy": {
        "keywords": ELECTRICITY_TARGET_KEYWORDS,
        "description": "Electricity Load & Energy Demand",
    },
    "price_cost": {
        "keywords": PRICE_KEYWORDS,
        "description": "Unit Price / Product Cost / Index",
    },
    "discount_promo": {
        "keywords": PROMO_KEYWORDS,
        "description": "Discount & Promotional Incentive",
    },
    "temperature_climate": {
        "keywords": {"temperature", "temp", "avg_temp", "temperature_c", "temperature_f", "ambient_temp", "weather_temp", "rainfall_mm", "humidity_pct"},
        "description": "Weather & Climate Conditions",
    },
    "holiday_event": {
        "keywords": {"holiday", "is_holiday", "holiday_flag", "festival", "event_flag", "holiday_name", "festival_flag", "payday_week", "is_weekend"},
        "description": "Holiday / Calendar Event Flag",
    },
    "geography": {
        "keywords": GEOGRAPHIC_KEYWORDS,
        "description": "Geographic Location & Territory",
    },
    "identifier": {
        "keywords": ID_KEYWORDS,
        "description": "Entity Identifier / Code",
    },
}


def _clean_name(column_name: Any) -> str:
    s = str(column_name).strip()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s).lower().strip("_")
    return re.sub(r"_+", "_", s)


def match_column_by_candidates(
    df: pd.DataFrame,
    candidate_keywords: set[str] | list[str],
    numeric_only: bool = False,
    non_numeric_only: bool = False,
    fuzzy_threshold: float = 0.80,
) -> list[tuple[str, float, str]]:
    """Perform multi-tier matching for columns in df against candidate keywords.
    
    Priority tiers:
    1. Exact case-sensitive match (score: 1.0)
    2. Case-insensitive exact match (score: 0.98)
    3. Normalized clean match (score: 0.95)
    4. Known synonym / substring match (score: 0.88)
    5. Fuzzy similarity via SequenceMatcher (score: 0.70 - 0.85)
    
    Returns sorted list of (column_name, confidence, method).
    """
    results: list[tuple[str, float, str]] = []
    seen_cols: set[str] = set()

    for col in df.columns:
        col_str = str(col)
        if col_str in seen_cols:
            continue

        if numeric_only and not pd.api.types.is_numeric_dtype(df[col]):
            sample = df[col].dropna().head(20)
            if sample.empty or pd.to_numeric(sample, errors="coerce").isna().all():
                continue
        if non_numeric_only and pd.api.types.is_numeric_dtype(df[col]):
            continue

        clean = _clean_name(col_str)
        clean_no_sep = clean.replace("_", "")

        # 1. Exact case-sensitive
        if col_str in candidate_keywords:
            results.append((col_str, 1.0, "exact"))
            seen_cols.add(col_str)
            continue

        # 2. Case-insensitive
        if any(col_str.lower() == str(kw).lower() for kw in candidate_keywords):
            results.append((col_str, 0.98, "case_insensitive"))
            seen_cols.add(col_str)
            continue

        # 3. Normalized match
        if any(clean == _clean_name(kw) or clean_no_sep == _clean_name(kw).replace("_", "") for kw in candidate_keywords):
            results.append((col_str, 0.95, "normalized"))
            seen_cols.add(col_str)
            continue

        # 4. Synonym / Substring match
        if any(kw in clean or clean in kw or kw.replace("_", "") in clean_no_sep for kw in candidate_keywords if len(kw) >= 3):
            results.append((col_str, 0.88, "synonym"))
            seen_cols.add(col_str)
            continue

        # 5. Fuzzy similarity
        best_ratio = 0.0
        for kw in candidate_keywords:
            ratio = difflib.SequenceMatcher(None, clean, _clean_name(kw)).ratio()
            if ratio > best_ratio:
                best_ratio = ratio

        if best_ratio >= fuzzy_threshold:
            conf = round(0.70 + (best_ratio - fuzzy_threshold) * 0.75, 2)
            results.append((col_str, min(0.85, conf), "fuzzy"))
            seen_cols.add(col_str)

    results.sort(key=lambda x: x[1], reverse=True)
    return results



def match_semantic_features(df: pd.DataFrame, domain: str = "retail") -> dict[str, dict[str, Any]]:
    """Perform generic semantic feature matching for all columns in an uploaded dataset."""
    matches: dict[str, dict[str, Any]] = {}
    n_rows = len(df)
    domain_clean = (domain or "retail").lower().strip()

    for col in df.columns:
        col_str = str(col)
        clean = _clean_name(col_str)
        clean_no_sep = clean.replace("_", "")
        series = df[col]
        non_null = series.dropna()

        matched_family = "generic_categorical" if not pd.api.types.is_numeric_dtype(series) else "generic_numeric"
        confidence = 0.4
        reason = f"Classified as {matched_family} based on data type {series.dtype}"

        # 1. Date/Time check
        is_date_kw = any(clean == kw or clean_no_sep == kw.replace("_", "") or kw in clean for kw in TIME_NAME_KEYWORDS)
        if pd.api.types.is_datetime64_any_dtype(series):
            matched_family = "date_time"
            confidence = 1.0
            reason = f"Matched datetime dtype ({series.dtype})"
        elif is_date_kw:
            sample = non_null.head(50)
            if not sample.empty:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    if float(parsed.notna().mean()) > 0.6:
                        matched_family = "date_time"
                        confidence = 0.95
                        reason = f"Matched date family based on name '{col_str}' and sample datetime parsing"
                except Exception:
                    pass

        # 2. Family keyword checks if not date_time
        if matched_family != "date_time":
            best_kw_score = 0.0
            best_family = matched_family

            for family_key, family_meta in SEMANTIC_FAMILIES.items():
                if family_key == "date_time":
                    continue
                kws = family_meta["keywords"]
                exact = any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in kws)
                sub = any(kw in clean or kw.replace("_", "") in clean_no_sep for kw in kws)

                score = 0.0
                if exact:
                    score = 0.95
                elif sub:
                    score = 0.75

                if score > best_kw_score:
                    best_kw_score = score
                    best_family = family_key

            if best_kw_score > 0.5:
                matched_family = best_family
                confidence = best_kw_score
                desc = SEMANTIC_FAMILIES[best_family]["description"]
                reason = f"Matched '{best_family}' ({desc}) based on column name '{col_str}'"

        # 3. ID / High cardinality check
        if matched_family in {"generic_numeric", "generic_categorical"} and not non_null.empty:
            unique_ratio = non_null.nunique() / len(non_null)
            is_id_name = any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in ID_KEYWORDS) or any(
                kw in clean for kw in {"id", "no", "code", "sku"}
            )
            if is_id_name or (unique_ratio > 0.85 and n_rows > 30 and matched_family != "geography"):
                matched_family = "identifier"
                confidence = 0.85
                reason = f"Identified entity ID column based on name/high uniqueness ratio ({unique_ratio:.2f})"

        matches[col_str] = {
            "column_name": col_str,
            "matched_family": matched_family,
            "confidence_score": round(confidence, 2),
            "reason": reason,
            "dtype": str(series.dtype),
        }

    return matches


def detect_time_column(df: pd.DataFrame) -> tuple[str | None, list[str], float]:
    """Detect primary date/timestamp column and return (selected, candidates, confidence)."""
    candidates: list[tuple[str, float]] = []

    for col in df.columns:
        col_str = str(col)
        clean = _clean_name(col_str)
        clean_no_sep = clean.replace("_", "")

        # Skip personal attributes like Date of Birth or DOB from timestamp detection
        if clean in {"date_of_birth", "dob", "birth_date", "birthdate"}:
            continue

        score = 0.0

        kw_exact_match = any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in TIME_NAME_KEYWORDS if kw != "year")
        kw_sub_match = any(kw in clean or kw.replace("_", "") in clean_no_sep for kw in TIME_NAME_KEYWORDS if kw != "year")

        if clean == "year":
            score += 0.2
        elif kw_exact_match:
            score += 0.55
        elif kw_sub_match:
            score += 0.35

        if pd.api.types.is_datetime64_any_dtype(df[col]):
            score += 0.5
        else:
            sample = df[col].dropna().head(100)
            if not sample.empty:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    valid_ratio = float(parsed.notna().mean())
                    if valid_ratio > 0.7:
                        score += 0.45 * valid_ratio
                        # Extra bonus for actual full date string columns over single year integers
                        if not pd.api.types.is_integer_dtype(df[col]) and sample.nunique() > 10:
                            score += 0.2
                except (ValueError, TypeError, OverflowError):
                    pass

        if score >= 0.35:
            candidates.append((col_str, min(1.0, score)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidate_names = [c[0] for c in candidates]
    if candidates:
        selected, best_score = candidates[0]
        return selected, candidate_names, best_score
    return None, [], 0.0



def detect_target_column(df: pd.DataFrame, domain: str = "retail") -> tuple[str | None, list[str], float]:
    """Detect domain-specific target column candidates and return (selected, candidates, confidence)."""
    domain_clean = (domain or "retail").lower().strip()
    target_keywords = RETAIL_TARGET_KEYWORDS if domain_clean == "retail" else ELECTRICITY_TARGET_KEYWORDS

    candidates: list[tuple[str, float]] = []
    n_rows = len(df)

    for col in df.columns:
        col_str = str(col)
        clean = _clean_name(col_str)
        clean_no_sep = clean.replace("_", "")

        if not pd.api.types.is_numeric_dtype(df[col]):
            sample = df[col].dropna().head(100)
            if sample.empty or pd.to_numeric(sample, errors="coerce").isna().all():
                continue

        non_null = df[col].dropna()
        if non_null.empty or (n_rows > 1 and non_null.nunique() <= 1):
            continue

        unique_ratio = non_null.nunique() / len(non_null) if len(non_null) > 0 else 0
        if unique_ratio > 0.95 and n_rows > 50:
            is_target_kw = any(kw in clean or kw.replace("_", "") in clean_no_sep for kw in target_keywords)
            if not is_target_kw and pd.api.types.is_integer_dtype(df[col]):
                continue

        is_id_name = any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in ID_KEYWORDS) or (
            any(kw in clean for kw in {"id", "no", "code", "customer", "invoice", "stock"}) and clean not in target_keywords
        )

        score = 0.25
        if is_id_name:
            score -= 0.5

        exact_match = any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in target_keywords)
        sub_match = any(kw in clean or kw.replace("_", "") in clean_no_sep for kw in target_keywords)

        if exact_match:
            score += 0.75
        elif sub_match:
            score += 0.45

        if domain_clean == "retail" and clean in {"sales", "units_sold", "quantity", "revenue", "qty"}:
            score += 0.2
        elif domain_clean == "electricity" and clean in {"demand_mw", "load_mw", "grid_load", "demand", "load"}:
            score += 0.2

        if score >= 0.35:
            candidates.append((col_str, min(1.0, score)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    candidate_names = [c[0] for c in candidates]
    if candidates:
        selected, best_score = candidates[0]
        return selected, candidate_names, best_score
    return None, [], 0.0


def detect_geographic_columns(df: pd.DataFrame) -> list[str]:
    """Identify likely geographic columns in dataset."""
    geo_cols: list[str] = []
    for col in df.columns:
        col_str = str(col)
        clean = _clean_name(col_str)
        clean_no_sep = clean.replace("_", "")

        matches_keyword = any(clean == kw or clean_no_sep == kw.replace("_", "") for kw in GEOGRAPHIC_KEYWORDS) or any(
            kw in clean for kw in GEOGRAPHIC_KEYWORDS
        )
        if matches_keyword:
            geo_cols.append(col_str)
    return geo_cols


def classify_columns(
    df: pd.DataFrame,
    time_column: str | None = None,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Classify columns into numeric, categorical, geographic, and excluded categories with strict feature governance."""
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    geographic_cols: list[str] = detect_geographic_columns(df)
    ignored_cols: list[str] = []
    ignored_reasons: dict[str, str] = {}

    n_rows = len(df)

    for col in df.columns:
        col_str = str(col)
        if col_str in (time_column, target_column):
            continue

        clean = _clean_name(col_str)
        series = df[col]
        non_null_count = series.notna().sum()

        if non_null_count == 0:
            ignored_cols.append(col_str)
            ignored_reasons[col_str] = "Column contains 100% missing values"
            continue

        # Strict feature governance for personal information & identifiers
        is_personal_or_id = any(clean == kw or clean.replace("_", "") == kw.replace("_", "") for kw in EXCLUDED_PERSONAL_OR_IDENTIFIER_KEYWORDS)
        if is_personal_or_id:
            ignored_cols.append(col_str)
            ignored_reasons[col_str] = "Personal attribute or identifier excluded from model predictors"
            continue

        if pd.api.types.is_numeric_dtype(series):
            if series.nunique(dropna=True) <= 1:
                ignored_cols.append(col_str)
                ignored_reasons[col_str] = "Constant column with single value"
            else:
                numeric_cols.append(col_str)
        else:
            unique_count = series.nunique(dropna=True)
            unique_ratio = unique_count / n_rows if n_rows > 0 else 0

            if unique_ratio > 0.85 and n_rows > 30 and col_str not in geographic_cols:
                ignored_cols.append(col_str)
                ignored_reasons[col_str] = "High uniqueness (identifier / text ID)"
            elif col_str in geographic_cols:
                pass
            else:
                categorical_cols.append(col_str)

    return {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "geographic_columns": geographic_cols,
        "ignored_columns": ignored_cols,
        "ignored_reasons": ignored_reasons,
    }


def clean_and_impute_missing_values(
    df: pd.DataFrame,
    time_col: str | None = None,
    target_col: str | None = None,
    domain: str = "retail",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Perform deterministic missing-value imputation and invalid date/target row removal."""
    working_df = df.copy()
    missing_before = int(working_df.isna().sum().sum())
    initial_rows = len(working_df)

    rows_removed = 0

    # 1. Clean Time Column
    if time_col and time_col in working_df.columns:
        parsed_dates = pd.to_datetime(working_df[time_col], errors="coerce")
        invalid_dates_mask = parsed_dates.isna()
        if invalid_dates_mask.any():
            rows_removed += int(invalid_dates_mask.sum())
            working_df = working_df[~invalid_dates_mask].copy()
            working_df[time_col] = parsed_dates[~invalid_dates_mask]

    # 2. Clean Target Column
    if target_col and target_col in working_df.columns:
        numeric_target = pd.to_numeric(working_df[target_col], errors="coerce")
        invalid_target_mask = numeric_target.isna()
        if invalid_target_mask.any():
            rows_removed += int(invalid_target_mask.sum())
            working_df = working_df[~invalid_target_mask].copy()
            working_df[target_col] = numeric_target[~invalid_target_mask]

    # 3. Numeric Columns Imputation
    for col in working_df.select_dtypes(include="number").columns:
        if col in (time_col, target_col):
            continue
        s = working_df[col]
        if s.isna().any():
            # Interpolate linear -> ffill -> bfill -> median
            s_filled = s.interpolate(method="linear").ffill().bfill()
            if s_filled.isna().any():
                med_val = s_filled.median()
                s_filled = s_filled.fillna(med_val if not pd.isna(med_val) else 0.0)
            working_df[col] = s_filled

    # 4. Categorical Columns Imputation
    for col in working_df.select_dtypes(exclude="number").columns:
        if col in (time_col, target_col):
            continue
        s = working_df[col]
        if s.isna().any():
            mode_val = s.mode()
            fill_val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            working_df[col] = s.fillna(fill_val)

    missing_after = int(working_df.isna().sum().sum())
    missing_handled = missing_before - missing_after

    report = {
        "missing_values_before": missing_before,
        "missing_values_handled": max(0, missing_handled),
        "rows_removed_due_to_invalid_data": rows_removed,
        "missing_values_after": missing_after,
    }

    return working_df, report


def analyze_forecasting_suitability(
    df: pd.DataFrame,
    domain: str = "retail",
    requested_target: str | None = None,
) -> dict[str, Any]:
    """Perform thorough forecasting suitability analysis and classify dataset as SUITABLE, SUITABLE_WITH_WARNINGS, or UNSUITABLE."""
    n_rows = len(df)
    n_cols = len(df.columns)
    warnings: list[str] = []
    reasons: list[str] = []

    time_col, time_candidates, time_conf = detect_time_column(df)
    target_col, target_candidates, target_conf = detect_target_column(df, domain=domain)

    if requested_target and requested_target in df.columns:
        target_col = requested_target

    # 1. Check Row Count
    if n_rows < 10:
        reasons.append(f"Dataset has only {n_rows} observations; minimum 10 required for time-series modeling.")
    elif n_rows < 30:
        warnings.append(f"Dataset has only {n_rows} observations. Forecast reliability may be limited.")

    # 2. Check Column Count
    if n_cols < 2:
        reasons.append(f"Dataset has only {n_cols} column; minimum 2 required (date/time index and target variable).")

    # 3. Check Time Column & Timestamp Uniqueness Quality
    unique_timestamps = 0
    dup_timestamps = 0
    uniqueness_ratio = 0.0

    if not time_col:
        reasons.append("No valid chronological date/time column detected in dataset.")
    else:
        parsed_dates = pd.to_datetime(df[time_col], errors="coerce")
        unique_timestamps = int(parsed_dates.nunique())
        dup_timestamps = int(parsed_dates.duplicated().sum())
        uniqueness_ratio = round(float(unique_timestamps / n_rows), 4) if n_rows > 0 else 0.0

        if dup_timestamps > 0:
            warnings.append(f"Transaction-level data detected: {dup_timestamps} observations share the same timestamp. They will be aggregated to the forecasting frequency.")


        # Extremely low timestamp uniqueness check (< 5 unique timestamps)
        if n_rows >= 30 and unique_timestamps < 5:
            reasons.append("Dataset is not suitable for reliable time-series forecasting. The selected timestamp column contains insufficient unique temporal observations. Please upload a dataset with adequate temporal coverage.")


        valid_dates = parsed_dates.dropna().sort_values()
        if len(valid_dates) > 1:
            diffs = valid_dates.diff().dropna()
            mode_diff = diffs.mode()
            freq_str = f"Interval ~{mode_diff.iloc[0]}" if not mode_diff.empty else "Irregular"
        else:
            freq_str = "Unknown"

    # 4. Check Target Column
    target_stats: dict[str, Any] = {}
    if not target_col or target_col not in df.columns:
        # Check if constant target column was rejected
        constant_cols = [c for c in df.select_dtypes(include="number").columns if c not in (time_col,) and df[c].nunique() <= 1]
        if constant_cols:
            reasons.append(f"Target column '{constant_cols[0]}' has zero variance (constant value {df[constant_cols[0]].iloc[0] if not df.empty else 0}).")
        else:
            reasons.append(f"No valid numeric target column detected for {domain.capitalize()} domain.")
    else:
        target_s = pd.to_numeric(df[target_col], errors="coerce").dropna()
        if target_s.empty:
            reasons.append(f"Target column '{target_col}' contains zero numeric values.")
        else:
            mean_val = float(target_s.mean())
            std_val = float(target_s.std())
            min_val = float(target_s.min())
            max_val = float(target_s.max())
            nunique = int(target_s.nunique())

            target_stats = {
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "min": round(min_val, 4),
                "max": round(max_val, 4),
                "nunique": nunique,
                "missing_count": int(df[target_col].isna().sum()),
            }

            if nunique <= 1 or math.isnan(std_val) or std_val == 0:
                reasons.append(f"Target column '{target_col}' has zero variance (constant value {mean_val}).")
            elif std_val / (abs(mean_val) + 1e-8) < 0.001:
                warnings.append(f"Target column '{target_col}' has near-zero variation over time.")

            missing_ratio = df[target_col].isna().mean()
            if missing_ratio > 0.5:
                reasons.append(f"Target column '{target_col}' has >50% missing values ({missing_ratio*100:.1f}%).")
            elif missing_ratio > 0.1:
                warnings.append(f"Target column '{target_col}' has {missing_ratio*100:.1f}% missing values.")

    # 5. Classify Suitability
    if len(reasons) > 0:
        classification = "UNSUITABLE"
        is_suitable = False
        status_label = "Unsuitable for Forecasting"
    elif len(warnings) > 0:
        classification = "SUITABLE_WITH_WARNINGS"
        is_suitable = True
        status_label = "Suitable with Warnings"
    else:
        classification = "SUITABLE"
        is_suitable = True
        status_label = "Fully Suitable"

    semantic_matches = match_semantic_features(df, domain=domain)
    classifications = classify_columns(df, time_column=time_col, target_column=target_col)

    if classification == "UNSUITABLE":
        explanation = f"Dataset may not be fully suitable for {domain.capitalize()} forecasting. Reason: {' '.join(reasons)}"
    elif classification == "SUITABLE_WITH_WARNINGS":
        explanation = f"Dataset is valid for forecasting with warnings. Note: {' '.join(warnings)}"
    else:
        explanation = f"Dataset is fully suitable for {domain.capitalize()} time-series forecasting."

    return {
        "classification": classification,
        "is_suitable": is_suitable,
        "status_label": status_label,
        "warnings": warnings,
        "reasons": reasons,
        "explanation": explanation,
        "row_count": n_rows,
        "column_count": n_cols,
        "unique_timestamps": unique_timestamps,
        "duplicate_timestamps": dup_timestamps,
        "timestamp_uniqueness_ratio": uniqueness_ratio,
        "detected_time_column": time_col,
        "detected_target_column": target_col,
        "target_statistics": target_stats,
        "excluded_features_count": len(classifications["ignored_columns"]),
        "excluded_features": classifications["ignored_columns"],
        "excluded_reasons": classifications["ignored_reasons"],
        "final_forecasting_features_count": len(classifications["numeric_columns"]) + len(classifications["categorical_columns"]),
        "semantic_matches": semantic_matches,
    }


def aggregate_duplicate_timestamps(
    df: pd.DataFrame,
    time_col: str,
    target_col: str,
    domain: str = "retail",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate records with duplicate timestamps based on semantic column families."""
    if time_col not in df.columns or not df[time_col].duplicated().any():
        return df, {
            "aggregated": False,
            "raw_rows": len(df),
            "aggregated_rows": len(df),
            "duplicate_timestamp_count": 0,
            "rows_before_aggregation": len(df),
            "rows_after_aggregation": len(df),
            "unique_timestamps_before": int(df[time_col].nunique()) if time_col in df.columns else len(df),
            "unique_timestamps_after": int(df[time_col].nunique()) if time_col in df.columns else len(df),
            "methods": {}
        }

    raw_rows = len(df)
    prep_df = df.copy()
    prep_df[time_col] = pd.to_datetime(prep_df[time_col], errors="coerce")
    prep_df = prep_df.dropna(subset=[time_col])

    dup_count = int(prep_df[time_col].duplicated().sum())
    u_before = int(prep_df[time_col].nunique())

    semantic_matches = match_semantic_features(prep_df, domain=domain)
    agg_dict: dict[str, str] = {}

    for col in prep_df.columns:
        if col == time_col:
            continue
        sem_family = semantic_matches.get(col, {}).get("matched_family", "")

        if col == target_col:
            agg_dict[col] = "sum" if domain == "retail" or sem_family in {"sales_revenue", "units_quantity"} else "mean"
        elif pd.api.types.is_numeric_dtype(prep_df[col]):
            if sem_family in {"sales_revenue", "units_quantity"}:
                agg_dict[col] = "sum"
            elif sem_family in {"price_cost", "discount_promo", "temperature_climate", "power_energy"}:
                agg_dict[col] = "mean"
            elif sem_family == "holiday_event":
                agg_dict[col] = "max"
            else:
                agg_dict[col] = "mean"
        else:
            agg_dict[col] = "first"

    agg_df = prep_df.groupby(time_col, as_index=False).agg(agg_dict)
    agg_df = agg_df.sort_values(time_col).reset_index(drop=True)

    summary = {
        "aggregated": True,
        "raw_rows": raw_rows,
        "aggregated_rows": len(agg_df),
        "duplicate_timestamp_count": dup_count,
        "rows_before_aggregation": raw_rows,
        "rows_after_aggregation": len(agg_df),
        "unique_timestamps_before": u_before,
        "unique_timestamps_after": len(agg_df),
        "methods": agg_dict,
    }

    return agg_df, summary


def profile_dataset(df: pd.DataFrame, domain: str = "retail", requested_target: str | None = None) -> dict[str, Any]:
    """Inspect dataset and compute intelligence, schema breakdown, suitability, and data quality metadata."""
    n_rows = len(df)
    n_cols = len(df.columns)
    column_names = [str(c) for c in df.columns]

    time_col, time_candidates, time_conf = detect_time_column(df)
    target_col, target_candidates, target_conf = detect_target_column(df, domain=domain)
    if requested_target and requested_target in column_names:
        target_col = requested_target
        if requested_target not in target_candidates:
            target_candidates.insert(0, requested_target)

    # Missing value cleaning & report
    cleaned_df, missing_report = clean_and_impute_missing_values(df, time_col=time_col, target_col=target_col, domain=domain)

    suitability = analyze_forecasting_suitability(cleaned_df, domain=domain, requested_target=requested_target)
    classifications = classify_columns(df, time_column=time_col, target_column=target_col)


    exact_dup_count = int(df.duplicated().sum())
    dup_timestamp_count = suitability.get("duplicate_timestamps", 0)
    unique_timestamps = suitability.get("unique_timestamps", 0)

    cleaned_rows = max(1, len(cleaned_df) - exact_dup_count)

    if not suitability["is_suitable"]:
        explanation = f"Dataset may not be fully suitable for {domain.capitalize()} forecasting. Reason: {' '.join(suitability['reasons'])}"
    elif len(suitability["warnings"]) > 0:
        explanation = f"Dataset is valid for forecasting with warnings. Note: {' '.join(suitability['warnings'])}"
    else:
        explanation = f"Dataset is fully suitable for {domain.capitalize()} time-series forecasting."

    return {
        "domain": (domain or "retail").lower().strip(),
        "row_count": n_rows,
        "column_count": n_cols,
        "columns": column_names,
        "suitable": suitability["is_suitable"],
        "suitability": suitability,
        "confidence": 0.95 if suitability["is_suitable"] else 0.3,
        "confidence_label": "High" if suitability["is_suitable"] else "Low",
        "time_column": time_col,
        "time_candidates": time_candidates,
        "target_candidates": target_candidates,
        "selected_target": target_col,
        "unique_timestamps": unique_timestamps,
        "semantic_matches": suitability["semantic_matches"],
        "numeric_columns": classifications["numeric_columns"],
        "categorical_columns": classifications["categorical_columns"],
        "geographic_columns": classifications["geographic_columns"],
        "ignored_columns": classifications["ignored_columns"],
        "ignored_reasons": classifications["ignored_reasons"],
        "excluded_features_count": len(classifications["ignored_columns"]),
        "final_forecasting_features_count": len(classifications["numeric_columns"]) + len(classifications["categorical_columns"]),
        "missing_summary": {str(col): int(count) for col, count in df.isna().sum().items() if count > 0},
        "missing_percentages": {str(col): round(float(count / n_rows * 100), 2) for col, count in df.isna().sum().items() if count > 0},
        "missing_report": missing_report,
        "missing_values_handled": missing_report["missing_values_handled"],
        "duplicate_rows": exact_dup_count,

        "duplicate_rows_removed": exact_dup_count,
        "duplicate_timestamps_count": dup_timestamp_count,
        "cleaned_rows": cleaned_rows,
        "issues": suitability["reasons"],
        "explanation": explanation,
    }


def get_geographic_hierarchy(df: pd.DataFrame, active_filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect dataset geographic columns and return available cascading options, locations list, and row counts."""
    filters = active_filters or {}
    col_map: dict[str, str] = {}
    for col in df.columns:
        clean = _clean_name(col)
        if clean in {"country", "country_name", "nation"} and "country" not in col_map:
            col_map["country"] = str(col)
        elif clean in {"state", "province", "state_name"} and "state" not in col_map:
            col_map["state"] = str(col)
        elif clean in {"district", "county", "district_name"} and "district" not in col_map:
            col_map["district"] = str(col)
def _is_no_filter_val(val: Any) -> bool:
    """Return True if the filter value represents an unselected / 'All' placeholder."""
    if val is None:
        return True
    s = str(val).strip()
    if not s:
        return True
    s_low = s.lower()
    if s_low in ("none", "null", "nan", "all", "all data", "all geographies", "all_data", "all_geographies", "undefined"):
        return True
    if s_low.startswith("all ") or s_low.startswith("all_"):
        return True
    return False


def get_geographic_hierarchy(df: pd.DataFrame, active_filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect dataset geographic columns and return available cascading options, locations list, and row counts."""
    filters = active_filters or {}
    all_geo_cols = detect_geographic_columns(df)
    col_map: dict[str, str] = {}
    for col in df.columns:
        clean = _clean_name(col)
        if clean in {"country", "country_name", "nation"} and "country" not in col_map:
            col_map["country"] = str(col)
        elif clean in {"state", "province", "state_name"} and "state" not in col_map:
            col_map["state"] = str(col)
        elif clean in {"district", "county", "district_name"} and "district" not in col_map:
            col_map["district"] = str(col)
        elif clean in {"region", "territory", "zone"} and "region" not in col_map:
            col_map["region"] = str(col)
        elif clean in {"city", "town", "city_name"} and "city" not in col_map:
            col_map["city"] = str(col)
        elif clean in {"location", "place", "area", "site", "branch", "store_location", "market"} and "location" not in col_map:
            col_map["location"] = str(col)

    for g_col in all_geo_cols:
        if g_col not in col_map.values():
            clean = _clean_name(g_col)
            if "city" not in col_map and any(k in clean for k in ("city", "town")):
                col_map["city"] = g_col
            elif "state" not in col_map and any(k in clean for k in ("state", "prov")):
                col_map["state"] = g_col
            elif "district" not in col_map and any(k in clean for k in ("dist", "county")):
                col_map["district"] = g_col
            elif "region" not in col_map and any(k in clean for k in ("region", "zone")):
                col_map["region"] = g_col
            else:
                col_map[clean] = g_col

    # Compile all unique available individual location values from unfiltered dataset
    all_locs = set()
    for col_name in list(col_map.values()) + [c for c in all_geo_cols if c in df.columns]:
        if col_name in df.columns:
            for val in df[col_name].dropna().unique():
                val_str = str(val).strip()
                if val_str and not _is_no_filter_val(val_str):
                    all_locs.add(val_str)

    has_geo = len(all_locs) > 0

    if not has_geo:
        return {
            "has_geography": False,
            "geographic_columns": {},
            "hierarchy_levels": [],
            "levels": {"countries": [], "states": [], "districts": [], "regions": [], "cities": [], "locations": []},
            "available_locations": [],
            "active_filters": {},
            "filtered_rows": len(df),
            "total_rows": len(df),
            "message": "Geographic filtering is not available for this dataset.",
        }

    # Sequential cascading filter
    filtered = df

    # Direct location matching: if location is given, dynamically find which column contains it
    loc_val = filters.get("location")
    if not _is_no_filter_val(loc_val):
        loc_str = str(loc_val).strip()
        loc_lower = loc_str.lower()
        matched_col = None
        for col_name in list(col_map.values()) + [c for c in all_geo_cols if c in df.columns]:
            if col_name in df.columns:
                unique_vals = [str(v).strip().lower() for v in df[col_name].dropna().unique()]
                if loc_lower in unique_vals:
                    matched_col = col_name
                    break
        if matched_col:
            filtered = filtered[filtered[matched_col].astype(str).str.strip().str.lower() == loc_lower]
        else:
            filtered = filtered.iloc[0:0]

    # Specific hierarchy level filters
    for level_key, col_name in col_map.items():
        if level_key != "location" and col_name in filtered.columns:
            f_val = filters.get(level_key) or filters.get(col_name)
            if not _is_no_filter_val(f_val):
                filtered = filtered[filtered[col_name].astype(str).str.strip().str.lower() == str(f_val).strip().lower()]

    countries = sorted([str(v) for v in df[col_map["country"]].dropna().unique() if str(v).strip() and not _is_no_filter_val(v)]) if "country" in col_map and col_map["country"] in df.columns else []
    states = sorted([str(v) for v in filtered[col_map["state"]].dropna().unique() if str(v).strip() and not _is_no_filter_val(v)]) if "state" in col_map and col_map["state"] in filtered.columns else []
    districts = sorted([str(v) for v in filtered[col_map["district"]].dropna().unique() if str(v).strip() and not _is_no_filter_val(v)]) if "district" in col_map and col_map["district"] in filtered.columns else []
    regions = sorted([str(v) for v in filtered[col_map["region"]].dropna().unique() if str(v).strip() and not _is_no_filter_val(v)]) if "region" in col_map and col_map["region"] in filtered.columns else []
    cities = sorted([str(v) for v in filtered[col_map["city"]].dropna().unique() if str(v).strip() and not _is_no_filter_val(v)]) if "city" in col_map and col_map["city"] in filtered.columns else []
    locations = sorted([str(v) for v in filtered[col_map["location"]].dropna().unique() if str(v).strip() and not _is_no_filter_val(v)]) if "location" in col_map and col_map["location"] in filtered.columns else []

    # Dynamic structured hierarchy levels for UI rendering
    hierarchy_levels = []
    level_order = ["country", "state", "district", "region", "city"]
    for g_key in col_map.keys():
        if g_key not in level_order:
            level_order.append(g_key)

    for l_key in level_order:
        if l_key in col_map and col_map[l_key] in df.columns:
            col_name = col_map[l_key]
            source_df = df if l_key == "country" else filtered
            col_vals = sorted([str(v) for v in source_df[col_name].dropna().unique() if str(v).strip() and not _is_no_filter_val(v)])
            hierarchy_levels.append({
                "level_key": l_key,
                "column_name": col_name,
                "label": col_name.replace("_", " ").title(),
                "values": col_vals,
                "active_value": filters.get(l_key, ""),
            })

    return {
        "has_geography": len(all_locs) > 0,
        "geographic_columns": col_map,
        "hierarchy_levels": hierarchy_levels,
        "levels": {
            "countries": countries,
            "states": states,
            "districts": districts,
            "regions": regions,
            "cities": cities,
            "locations": locations,
        },
        "available_locations": sorted(list(all_locs)),
        "active_filters": filters,
        "filtered_rows": len(filtered),
        "total_rows": len(df),
    }



