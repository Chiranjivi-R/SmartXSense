"""Dataset intelligence, schema profiling, column classification, and domain suitability scoring."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


TIME_NAME_KEYWORDS = {
    "date", "datetime", "timestamp", "time", "day", "month", "year",
    "invoice_date", "order_date", "transaction_date", "date_time", "datehour",
    "date_hour", "period", "timestamp_utc",
}

RETAIL_TARGET_KEYWORDS = {
    "sales", "quantity", "units", "demand", "revenue", "orders", "units_sold",
    "qty", "sales_units", "volume", "sales_revenue", "amount",
}

ELECTRICITY_TARGET_KEYWORDS = {
    "load", "demand", "demand_mw", "power", "consumption", "energy", "usage",
    "grid_load", "grid_load_mw", "hourly_demand_met_mw", "hourly_demand_met",
    "load_mw", "demand_met", "mw", "kwh", "mwh",
}

GEOGRAPHIC_KEYWORDS = {
    "country", "state", "district", "city", "region", "zone", "location",
    "area", "province", "county", "zipcode", "pincode", "postcode",
}


def _clean_name(column_name: Any) -> str:
    return re.sub(r"[^a-z0-9]", "_", str(column_name).lower()).strip("_")


def detect_time_column(df: pd.DataFrame) -> tuple[str | None, list[str], float]:
    """Detect primary date/timestamp column and return (selected, candidates, confidence)."""
    candidates: list[tuple[str, float]] = []

    for col in df.columns:
        col_str = str(col)
        clean = _clean_name(col_str)
        score = 0.0

        # Name match signal
        if clean in TIME_NAME_KEYWORDS:
            score += 0.5
        elif any(kw in clean for kw in TIME_NAME_KEYWORDS):
            score += 0.35

        # Dtype signal
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            score += 0.5
        else:
            # Sample date parsing
            sample = df[col].dropna().head(100)
            if not sample.empty:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                    valid_ratio = parsed.notna().mean()
                    if valid_ratio > 0.8:
                        score += 0.45 * valid_ratio
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

        # Must be numeric to be a forecasting target
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Try numeric coercion check
            sample = df[col].dropna().head(100)
            if sample.empty or pd.to_numeric(sample, errors="coerce").isna().all():
                continue

        # Check variance & uniqueness (avoid ID columns)
        non_null = df[col].dropna()
        if non_null.empty or non_null.nunique() <= 1:
            continue

        unique_ratio = non_null.nunique() / len(non_null) if len(non_null) > 0 else 0
        if unique_ratio > 0.95 and n_rows > 50:
            # High cardinality numeric ID column (e.g., sequential integer ID)
            continue

        score = 0.2  # Base numeric candidate score

        if clean in target_keywords:
            score += 0.65
        elif any(kw in clean for kw in target_keywords):
            score += 0.45

        # Domain canonical match
        if domain_clean == "retail" and clean in {"sales", "units_sold", "quantity", "revenue"}:
            score += 0.2
        elif domain_clean == "electricity" and clean in {"demand_mw", "load_mw", "grid_load", "demand"}:
            score += 0.2

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
        clean = _clean_name(col)
        if clean in GEOGRAPHIC_KEYWORDS or any(kw in clean for kw in GEOGRAPHIC_KEYWORDS):
            geo_cols.append(str(col))
    return geo_cols


def classify_columns(
    df: pd.DataFrame,
    time_column: str | None = None,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Classify all columns into numeric, categorical, geographic, and ignored categories."""
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

        series = df[col]
        non_null_count = series.notna().sum()

        if non_null_count == 0:
            ignored_cols.append(col_str)
            ignored_reasons[col_str] = "Column contains 100% missing values"
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
                pass  # Classified as geographic
            else:
                categorical_cols.append(col_str)

    return {
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "geographic_columns": geographic_cols,
        "ignored_columns": ignored_cols,
        "ignored_reasons": ignored_reasons,
    }


def profile_dataset(df: pd.DataFrame, domain: str = "retail", requested_target: str | None = None) -> dict[str, Any]:
    """Inspect dataset and compute intelligence, schema breakdown, domain fit score, and human explanation."""
    n_rows = len(df)
    n_cols = len(df.columns)
    column_names = [str(c) for c in df.columns]

    # Sample for fast profiling on large dataframes
    profiling_df = df if n_rows <= 10000 else df.sample(n=10000, random_state=42)

    time_col, time_candidates, time_conf = detect_time_column(profiling_df)
    target_col, target_candidates, target_conf = detect_target_column(profiling_df, domain=domain)

    # Use requested target if valid and present
    if requested_target and requested_target in column_names:
        target_col = requested_target
        if requested_target not in target_candidates:
            target_candidates.insert(0, requested_target)

    classifications = classify_columns(df, time_column=time_col, target_column=target_col)

    missing_summary = {str(col): int(count) for col, count in df.isna().sum().items() if count > 0}
    missing_percentages = {str(col): round(float(count / n_rows * 100), 2) for col, count in df.isna().sum().items() if count > 0}
    duplicate_rows = int(df.duplicated().sum())

    # Assess Domain Suitability
    issues: list[str] = []
    if time_col is None:
        issues.append("No date/time column detected in dataset.")
    if target_col is None:
        issues.append(f"No plausible {domain.capitalize()} target column (sales/demand/load) detected.")
    if n_rows < 10:
        issues.append("Dataset has fewer than 10 rows for time series modeling.")

    suitable = len(issues) == 0
    confidence_score = (time_conf * 0.5) + (target_conf * 0.5) if suitable else max(0.1, (time_conf * 0.3) + (target_conf * 0.3))

    if confidence_score >= 0.8:
        confidence_label = "High"
    elif confidence_score >= 0.5:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    # Human-readable explanation
    if suitable:
        explanation = (
            f"Dataset is highly suitable for {domain.capitalize()} demand forecasting. "
            f"Detected date column '{time_col}' and target column '{target_col}'. "
            f"Found {len(classifications['numeric_columns'])} numerical features and {len(classifications['categorical_columns'])} categorical features."
        )
    else:
        explanation = (
            f"Dataset may not be fully suitable for {domain.capitalize()} forecasting. "
            + " ".join(issues)
        )

    return {
        "domain": (domain or "retail").lower().strip(),
        "row_count": n_rows,
        "column_count": n_cols,
        "columns": column_names,
        "suitable": suitable,
        "confidence": round(confidence_score, 2),
        "confidence_label": confidence_label,
        "time_column": time_col,
        "time_candidates": time_candidates,
        "target_candidates": target_candidates,
        "selected_target": target_col,
        "numeric_columns": classifications["numeric_columns"],
        "categorical_columns": classifications["categorical_columns"],
        "geographic_columns": classifications["geographic_columns"],
        "ignored_columns": classifications["ignored_columns"],
        "ignored_reasons": classifications["ignored_reasons"],
        "missing_summary": missing_summary,
        "missing_percentages": missing_percentages,
        "duplicate_rows": duplicate_rows,
        "issues": issues,
        "explanation": explanation,
    }
