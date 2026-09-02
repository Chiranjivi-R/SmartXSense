"""Dynamic dataset preparation, target override, geographic subset filtering, and feature split materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ml_engine.features.artifacts import materialize_forecasting_features
from ml_engine.preprocessing.domain import preprocess_frame
from ml_engine.preprocessing.intelligence import (
    _is_no_filter_val,
    aggregate_duplicate_timestamps,
    analyze_forecasting_suitability,
    clean_and_impute_missing_values,
    classify_columns,
    detect_geographic_columns,
    detect_target_column,
    detect_time_column,
    get_geographic_hierarchy,
)


from ml_engine.splitting.time_series import TimeSeriesSplitError


class DynamicPrepError(Exception):
    """Execution error during dynamic dataset filtering and preparation."""

    def __init__(self, message: str, code: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code



def filter_df_by_geography(
    df: pd.DataFrame,
    geo_filters: dict[str, Any] | None = None,
    location: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Filter dataframe according to a location string or cascading geographic filters.
    
    Strict validation rules:
    - If location is requested on a dataset with no geographic columns, raises 400 error.
    - If location does not exist in the dataset, raises 400 error.
    - If 0 rows remain after filtering, raises 400 error.
    """
    clean_loc = str(location).strip() if not _is_no_filter_val(location) else None

    # Check if geo_filters has any non-placeholder values
    valid_filters: dict[str, str] = {}
    if geo_filters and isinstance(geo_filters, dict):
        for k, v in geo_filters.items():
            if not _is_no_filter_val(v):
                valid_filters[k] = str(v).strip()

    if not clean_loc and not valid_filters:
        return df, "All Geographies"

    geo_meta = get_geographic_hierarchy(df)
    if not geo_meta["has_geography"]:
        raise DynamicPrepError("Location-specific forecasting is unavailable for this dataset (no location or geographic column found).", 400)

    all_geo_cols = detect_geographic_columns(df)
    col_map = geo_meta["geographic_columns"]
    filtered = df

    # Case 1: Direct location string provided (or present in valid_filters)
    target_loc = clean_loc or valid_filters.get("location")
    if target_loc:
        loc_lower = str(target_loc).strip().lower()
        matched_col = None
        
        # Check all detected geographic columns
        for col_name in list(col_map.values()) + [c for c in all_geo_cols if c in df.columns]:
            if col_name in df.columns:
                unique_vals = [str(v).strip().lower() for v in df[col_name].dropna().unique()]
                if loc_lower in unique_vals:
                    matched_col = col_name
                    break

        if not matched_col:
            avail = geo_meta.get("available_locations", [])[:8]
            hint = f" Available locations: {', '.join(avail)}" if avail else ""
            raise DynamicPrepError(f"Selected location '{target_loc}' does not exist in dataset.{hint}", 400)

        filtered = df[df[matched_col].astype(str).str.strip().str.lower() == loc_lower]
        if len(filtered) == 0:
            raise DynamicPrepError(f"0 rows selected for location '{target_loc}'.", 400)
        return filtered, str(target_loc).strip()

    # Case 2: Cascading valid_filters provided
    applied_labels: list[str] = []
    level_order = ["location", "city", "district", "region", "state", "country"]
    for k in valid_filters.keys():
        if k not in level_order:
            level_order.append(k)

    for level in level_order:
        if level in valid_filters:
            val_str = valid_filters[level]
            col = col_map.get(level) or (level if level in filtered.columns else None)
            if col and col in filtered.columns:
                unique_vals = [str(v).strip().lower() for v in filtered[col].dropna().unique()]
                if val_str.lower() not in unique_vals:
                    raise DynamicPrepError(f"0 rows selected: Selected {level} '{val_str}' does not exist in dataset column '{col}'.", 400)
                filtered = filtered[filtered[col].astype(str).str.strip().str.lower() == val_str.lower()]
                applied_labels.append(val_str)

    geo_label = ", ".join(applied_labels) if applied_labels else "All Geographies"
    if len(filtered) == 0:
        raise DynamicPrepError(f"0 rows selected for location scope '{geo_label}'. Select a broader location.", 400)

    return filtered, geo_label





def compute_dynamic_fingerprint(
    df: pd.DataFrame,
    domain: str,
    target: str,
    geography_label: str = "All Geographies",
) -> str:
    """Compute deterministic short fingerprint of preprocessed dataset."""
    hasher = hashlib.sha256()
    hasher.update(domain.lower().encode("utf-8"))
    hasher.update(target.lower().encode("utf-8"))
    hasher.update(geography_label.lower().encode("utf-8"))
    hasher.update(str(len(df)).encode("utf-8"))
    hasher.update(str(list(df.columns)).encode("utf-8"))

    sample_data = pd.concat([df.head(5), df.tail(5)]).reset_index(drop=True).to_json()
    hasher.update(sample_data.encode("utf-8"))
    return hasher.hexdigest()[:16]


def prepare_dynamic_dataset(
    df: pd.DataFrame,
    domain: str = "retail",
    target: str | None = None,
    geo_filters: dict[str, Any] | None = None,
    location: str | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Filter, clean, prepare, and materialize feature splits for a dynamic dataset selection."""
    domain_clean = (domain or "retail").lower().strip()

    # Step 1: Geographic filtering
    filtered_df, geo_label = filter_df_by_geography(df, geo_filters=geo_filters, location=location)
    raw_row_count = len(filtered_df)

    if raw_row_count == 0:
        raise DynamicPrepError("0 rows selected after applying geographic filters. Select a broader geographic scope.", 400)


    # Step 2: Exact Duplicate Row Removal
    exact_duplicates_count = int(filtered_df.duplicated().sum())
    dedup_df = filtered_df.drop_duplicates().copy()
    cleaned_row_count_after_dedup = len(dedup_df)

    # Step 3: Missing Value Imputation & Invalid Row Removal
    time_col, _, _ = detect_time_column(dedup_df)
    selected_target, _, _ = detect_target_column(dedup_df, domain=domain_clean)
    if target and target in dedup_df.columns:
        selected_target = target

    clean_df, missing_report = clean_and_impute_missing_values(
        dedup_df, time_col=time_col, target_col=selected_target, domain=domain_clean
    )

    # Step 4: Forecasting Suitability Analysis on clean_df
    suitability = analyze_forecasting_suitability(clean_df, domain=domain_clean, requested_target=target)
    if not suitability["is_suitable"] or suitability["classification"] == "UNSUITABLE":
        raise DynamicPrepError(suitability["explanation"], 400)

    time_col = suitability["detected_time_column"]
    selected_target = suitability["detected_target_column"]

    if not time_col:
        raise DynamicPrepError(f"Dataset has no parseable date/time column for {domain_clean.capitalize()} forecasting.", 400)
    if not selected_target or selected_target not in clean_df.columns:
        raise DynamicPrepError(f"No valid numeric target column found in dataset for {domain_clean.capitalize()} forecasting.", 400)

    # Step 5: Feature Governance - Exclude personal/identifier/unsuitable columns
    classifications = classify_columns(clean_df, time_column=time_col, target_column=selected_target)
    ignored_cols = classifications["ignored_columns"]

    canonical_target = "sales" if domain_clean == "retail" else "demand_mw"
    prep_df = clean_df.copy()

    cols_to_drop = [c for c in ignored_cols if c in prep_df.columns and c not in (time_col, selected_target, canonical_target, "timestamp")]
    if cols_to_drop:
        prep_df = prep_df.drop(columns=cols_to_drop)

    if selected_target != canonical_target and canonical_target not in prep_df.columns:
        prep_df = prep_df.rename(columns={selected_target: canonical_target})

    if time_col != "timestamp" and "timestamp" not in prep_df.columns:
        prep_df = prep_df.rename(columns={time_col: "timestamp"})

    # Step 6: Semantic Duplicate Timestamp Aggregation (once)
    prep_df, agg_summary = aggregate_duplicate_timestamps(prep_df, "timestamp", canonical_target, domain=domain_clean)

    cleaning_summary = {
        "raw_row_count": raw_row_count,
        "exact_duplicates_removed": exact_duplicates_count,
        "cleaned_row_count_after_dedup": cleaned_row_count_after_dedup,
        "missing_values_before": missing_report["missing_values_before"],
        "missing_values_handled": missing_report["missing_values_handled"],
        "rows_removed_due_to_invalid_data": missing_report["rows_removed_due_to_invalid_data"],
        "duplicate_timestamps_count": agg_summary.get("duplicate_timestamp_count", 0),
        "rows_before_timestamp_agg": agg_summary.get("rows_before_aggregation", len(prep_df)),
        "final_cleaned_rows": len(prep_df),
        "excluded_features": ignored_cols,
        "excluded_features_count": len(ignored_cols),
        "aggregation_methods": agg_summary.get("methods", {}),
    }


    processed, features, _ = preprocess_frame(prep_df, domain_clean)


    # Step 5: Compute Fingerprint & Materialize Feature Splits
    fingerprint = compute_dynamic_fingerprint(processed, domain_clean, selected_target, geo_label)
    artifact_id = f"dyn_{fingerprint}"

    if output_root is None:
        project_root = Path(__file__).resolve().parents[2]
        output_root = project_root / "ml_engine" / "data" / "features"

    feature_dir = output_root / domain_clean / f"features_{artifact_id}"

    # Re-use existing feature directory if already materialized
    if (feature_dir / "train.parquet").exists() and (feature_dir / "test.parquet").exists():
        train_df = pd.read_parquet(feature_dir / "train.parquet")
        val_df = pd.read_parquet(feature_dir / "validation.parquet")
        test_df = pd.read_parquet(feature_dir / "test.parquet")
        return {
            "features_dir": str(feature_dir),
            "features_id": artifact_id,
            "fingerprint": fingerprint,
            "domain": domain_clean,
            "target": selected_target,
            "geography_label": geo_label,
            "rows_used": raw_row_count,
            "cleaned_rows": len(prep_df),
            "train_rows": len(train_df),
            "validation_rows": len(val_df),
            "test_rows": len(test_df),
            "feature_names": [c for c in train_df.columns if c not in {"timestamp", canonical_target}],
            "suitability": suitability,
            "aggregation_summary": agg_summary,
            "cleaning_summary": cleaning_summary,
        }

    try:
        materialized = materialize_forecasting_features(
            processed,
            domain_clean,
            output_root=output_root,
            artifact_id=artifact_id,
        )
    except TimeSeriesSplitError as err:
        raise DynamicPrepError(f"Geographic subset or dataset too small for time-series split: {err}", 400) from err
    except Exception as err:
        if isinstance(err, DynamicPrepError):
            raise
        raise DynamicPrepError(f"Feature materialization failed: {err}", 400) from err


    return {
        "features_dir": str(materialized["features_dir"]),
        "features_id": materialized["features_id"],
        "fingerprint": fingerprint,
        "domain": domain_clean,
        "target": selected_target,
        "geography_label": geo_label,
        "rows_used": raw_row_count,
        "cleaned_rows": len(prep_df),
        "train_rows": materialized["train_rows"],
        "validation_rows": materialized["validation_rows"],
        "test_rows": materialized["test_rows"],
        "feature_names": materialized["feature_names"],
        "suitability": suitability,
        "aggregation_summary": agg_summary,
        "cleaning_summary": cleaning_summary,
    }

