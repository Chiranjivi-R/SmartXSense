"""Comprehensive test suite for Dataset-Adaptive Geographic Discovery, Aggregation, and Safe Unit Handling.

Verifies:
1. Retail geographic behavior remains 100% unchanged.
2. Wide-format multi-state electricity datasets (e.g. Punjab, Haryana, UP) discover geographic series dynamically.
3. Timestamp columns with generic names like 'Unnamed: 0' are recognized when containing valid dates,
   while sequential integer indices ('0, 1, 2...') are excluded from features and timestamps.
4. Target resolution resolves to actual numeric series or explicit column names; NEVER '__aggregate__'.
5. Safe unit handling: unknown units remain 'Unspecified in source' / 'Electricity Demand (unit unspecified)',
   never assumed to be MW. Verified units (e.g. MW, kWh) are preserved.
6. Ambiguous/conflicting units or extreme scale disparities disable aggregation while allowing individual series.
7. Existing parent-level totals (e.g. Total, All India) and index columns are excluded from component series.
8. Missing observations use bounded interpolation and report coverage transparently.
9. Datasets without geography continue forecasting normally across the full dataset.
10. Cache isolation isolates fingerprints across locations and aggregates.
11. PDF and Dataset Intelligence preserve unit and target variable names accurately.
12. End-to-end simulation of 'india_power_raw (1).csv' if present on the filesystem.
"""

from datetime import timedelta
import io
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app import create_app
from app.extensions import db
from backend.app.services.ml_service import compare_service, forecast_service, MLError
from backend.app.services.dataset_service import get_dataset_geography_service, validate_dataset
from backend.app.services.pdf_report_service import build_pdf_document, _extract_dataset_intelligence
from ml_engine.preprocessing.intelligence import get_geographic_hierarchy, detect_time_column, profile_dataset
from ml_engine.preprocessing.feature_discovery import (
    discover_features_and_resolve_schema,
    detect_target_unit,
    detect_candidate_timestamps,
    adapt_frame,
)
from ml_engine.preprocessing.dynamic_prep import (
    prepare_dynamic_dataset,
    filter_df_by_geography,
    DynamicPrepError,
)


class TestConfig:
    TESTING = True
    SECRET_KEY = "test-secret"
    JWT_SECRET_KEY = "test-jwt-secret-key-that-is-at-least-32-bytes"
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=1)
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CORS_ORIGINS = ["http://localhost:5173"]


@pytest.fixture
def test_client(tmp_path):
    class CustomConfig(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts")
        UPLOAD_FOLDER = str(tmp_path / "uploads")
        PROCESSED_DATA_FOLDER = str(tmp_path / "processed")
        FEATURES_DATA_FOLDER = str(tmp_path / "features")

    app = create_app(CustomConfig)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        yield client


# ===========================================================================
# 1. Retail Geographic Behavior Preserved
# ===========================================================================

def test_retail_geographic_behavior_preserved():
    """Ensure existing Retail row-based location discovery and filtering remain unchanged."""
    demo_retail_dir = ROOT / "ml_engine" / "data" / "features" / "retail" / "features_1c0c92ff-264e-4398-b83c-e0e082b956b4"
    train_df = pd.read_parquet(demo_retail_dir / "train.parquet")
    geo = get_geographic_hierarchy(train_df)

    assert geo["has_geography"] is True
    assert "Ahmedabad" in geo["available_locations"]
    assert "Bangalore" in geo["available_locations"]
    assert geo["dataset_format"] == "row_based"
    assert len(geo["hierarchy_levels"]) > 0

    # Filtering for single location
    filtered, label = filter_df_by_geography(train_df, location="Bangalore")
    assert label == "Bangalore"
    assert len(filtered) > 0
    assert (filtered["city"].str.lower() == "bangalore").all()


# ===========================================================================
# 2. Wide-Format Multi-State Electricity Discovery & Target Resolution
# ===========================================================================

def test_wide_format_multistate_electricity_discovery():
    """Verify that wide-format time series columns are recognized dynamically,
    and the aggregate target resolves to a descriptive label, NEVER '__aggregate__'."""
    n = 60
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Unnamed: 0": range(n),
        "Date": dates,
        "Punjab": np.random.uniform(2500, 3500, n),
        "Haryana": np.random.uniform(2000, 3000, n),
        "Delhi": np.random.uniform(1500, 2500, n),
        "Rajasthan": np.random.uniform(3000, 4500, n),
        "UP": np.random.uniform(4500, 7000, n),
    })

    schema = discover_features_and_resolve_schema(df, domain="electricity")
    assert schema.is_valid is True
    assert schema.is_multiseries is True
    assert schema.dataset_format == "wide_multiseries"
    assert "Unnamed: 0" not in schema.detected_geographic_series
    assert set(schema.detected_geographic_series) == {"Delhi", "Haryana", "Punjab", "Rajasthan", "UP"}
    # Target column must NOT be raw string '__aggregate__'
    assert schema.target_column != "__aggregate__"
    assert "Aggregate" in schema.target_column

    geo = get_geographic_hierarchy(df)
    assert geo["has_geography"] is True
    assert geo["dataset_format"] == "wide_multiseries"
    assert geo["available_locations"] == sorted(["Delhi", "Haryana", "Punjab", "Rajasthan", "UP"])
    assert geo["allow_aggregate"] is True


# ===========================================================================
# 3. Unnamed: 0 as Date vs Sequential Integer Index
# ===========================================================================

def test_unnamed_header_as_valid_timestamp():
    """Generic headers like 'Unnamed: 0' containing valid date strings must be recognized as timestamps."""
    n = 40
    date_strs = pd.date_range("2019-02-01", periods=n, freq="D").strftime("%d/%m/%Y %H:%M:%S").tolist()
    df = pd.DataFrame({
        "Unnamed: 0": date_strs,
        "Punjab": np.random.uniform(200, 400, n),
        "Haryana": np.random.uniform(150, 300, n),
    })

    # Test in detect_time_column
    time_col, candidates, conf = detect_time_column(df)
    assert time_col == "Unnamed: 0"
    assert conf >= 0.70

    # Test in full feature discovery
    schema = discover_features_and_resolve_schema(df, domain="electricity")
    assert schema.time_column == "Unnamed: 0"
    assert schema.is_multiseries is True
    assert set(schema.detected_geographic_series) == {"Haryana", "Punjab"}
    assert "Unnamed: 0" not in schema.detected_geographic_series


def test_sequential_integer_index_excluded_from_timestamps_and_predictors():
    """Sequential integer columns (0, 1, 2...) named 'Unnamed: 0' must NOT be mistaken for timestamps,
    and must be excluded from features."""
    n = 50
    df = pd.DataFrame({
        "Unnamed: 0": list(range(n)),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "Punjab": np.random.uniform(200, 400, n),
        "Haryana": np.random.uniform(150, 300, n),
    })

    time_col, _, _ = detect_time_column(df)
    assert time_col == "timestamp"
    assert time_col != "Unnamed: 0"

    schema = discover_features_and_resolve_schema(df, domain="electricity")
    assert schema.time_column == "timestamp"
    assert "Unnamed: 0" in schema.excluded_columns
    assert "Unnamed: 0" not in schema.detected_geographic_series


# ===========================================================================
# 4. Safe Unit Handling (Preserve Scale, Never Assume MW)
# ===========================================================================

def test_safe_unit_handling_unspecified_vs_verified():
    """Unspecified units must never be assumed as MW, and original scale must be preserved."""
    # 1. Unverified series names
    unit, src, label = detect_target_unit("Punjab", domain="electricity")
    assert unit == "Unspecified in source"
    assert src == "unspecified"
    assert label == "Electricity Demand (unit unspecified)"

    # 2. Verified Megawatts series
    unit_mw, src_mw, label_mw = detect_target_unit("grid_load_mw", domain="electricity")
    assert unit_mw == "MW"
    assert src_mw == "source_metadata"
    assert label_mw == "Demand (MW)"

    # 3. Verified Kilowatt-hours series
    unit_kwh, src_kwh, label_kwh = detect_target_unit("consumption_kwh", domain="electricity")
    assert unit_kwh == "kWh"
    assert src_kwh == "source_metadata"
    assert label_kwh == "Demand (kWh)"

    # 4. Explicit user confirmation
    unit_user, src_user, label_user = detect_target_unit("Punjab", domain="electricity", user_unit="MWh")
    assert unit_user == "MWh"
    assert src_user == "user_confirmed"
    assert label_user == "Demand (MWh)"


def test_dataset_intelligence_records_unspecified_unit():
    """Dataset Intelligence and schema discovery must record 'Unspecified in source'."""
    n = 45
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "Punjab": np.random.uniform(100, 200, n),
        "Haryana": np.random.uniform(80, 160, n),
    })

    schema = discover_features_and_resolve_schema(df, domain="electricity")
    assert schema.unit == "Unspecified in source"
    assert schema.unit_source == "unspecified"
    assert schema.unit_label == "Electricity Demand (unit unspecified)"

    prof = profile_dataset(df, domain="electricity")
    assert prof["unit"] == "Unspecified in source"
    assert prof["unit_source"] == "unspecified"
    assert prof["unit_label"] == "Electricity Demand (unit unspecified)"


# ===========================================================================
# 5. Aggregation Restrictions: Incompatible Units & Scale Disparity
# ===========================================================================

def test_conflicting_units_disables_aggregation():
    """Series with conflicting explicit units (e.g. MW and kWh) must disable aggregation,
    but allow individual series forecasting."""
    n = 40
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "Region_A_MW": np.random.uniform(100, 200, n),
        "Region_B_kWh": np.random.uniform(100000, 200000, n),
    })

    schema = discover_features_and_resolve_schema(df, domain="electricity")
    assert schema.allow_aggregate is False
    assert "Conflicting measurement units" in schema.aggregation_disabled_reason

    # 1. Attempting aggregation when disabled raises ValueError or DynamicPrepError
    schema_agg = discover_features_and_resolve_schema(df, domain="electricity", explicit_target="__aggregate__")
    schema_agg.target_column = None
    with pytest.raises(ValueError, match="Geographic aggregation is unavailable"):
        schema_agg.adapt_frame(df)

    with pytest.raises(DynamicPrepError, match="Geographic aggregation is unavailable"):
        filter_df_by_geography(df, location="__aggregate__")

    # 2. Selecting individual location resolves cleanly and succeeds
    schema_single = discover_features_and_resolve_schema(df, domain="electricity", explicit_target="Region_A_MW")
    adapted = schema_single.adapt_frame(df)
    assert not adapted["demand_mw"].isna().any()
    assert len(adapted) == n


def test_extreme_scale_disparity_disables_aggregation():
    """Extreme scale disparity across components (ratio > 1000) disables aggregation to prevent distortion."""
    n = 40
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "SmallSubstation": np.random.uniform(1, 5, n),
        "MassiveGrid": np.random.uniform(50000, 80000, n),
    })

    schema = discover_features_and_resolve_schema(df, domain="electricity")
    assert schema.allow_aggregate is False
    assert "scale disparity" in schema.aggregation_disabled_reason

    # Individual selection is allowed
    prep = prepare_dynamic_dataset(df, domain="electricity", location="SmallSubstation")
    assert prep["target"] == "SmallSubstation"
    assert prep["train_rows"] > 0


# ===========================================================================
# 6. Parent Totals and Index Columns Excluded
# ===========================================================================

def test_parent_total_and_index_columns_excluded():
    """Verify that existing parent totals (e.g. Total, All) and row indices (Unnamed: 0)
    are detected and excluded from the component series to prevent double-counting."""
    n = 60
    p = np.random.uniform(2000, 4000, n)
    h = np.random.uniform(1500, 3500, n)
    tot = p + h  # Exact sum

    df = pd.DataFrame({
        "Unnamed: 0": range(n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "Punjab": p,
        "Haryana": h,
        "Total": tot,
    })

    schema = discover_features_and_resolve_schema(df, domain="electricity")
    assert "Total" in schema.detected_parent_totals
    assert "Total" not in schema.detected_geographic_series
    assert "Unnamed: 0" not in schema.detected_geographic_series
    assert schema.detected_geographic_series == ["Haryana", "Punjab"]


# ===========================================================================
# 7. Missing Value Handling & Coverage Reporting
# ===========================================================================

def test_transparent_missing_value_handling_and_coverage():
    """Verify missing values use bounded interpolation and are reported transparently in coverage meta."""
    n = 50
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "SeriesAlpha": np.random.uniform(100, 200, n),
        "SeriesBeta": np.random.uniform(150, 250, n),
    })
    # Inject 2 missing values (well within bounded limit=3)
    df.loc[10, "SeriesAlpha"] = np.nan
    df.loc[11, "SeriesAlpha"] = np.nan

    schema = discover_features_and_resolve_schema(df, domain="electricity")
    geo = get_geographic_hierarchy(df)

    assert geo["coverage_meta"]["missing_cells"] == 2
    assert geo["coverage_meta"]["coverage_pct"] == 98.0
    assert "bounded linear interpolation" in geo["coverage_meta"]["assumptions"]

    # Adapt frame performs interpolation without dropping or filling with 0
    adapted = schema.adapt_frame(df)
    assert not adapted["demand_mw"].isna().any()
    # At index 10, value must be interpolated between index 9 and 12, not 0
    assert adapted.loc[10, "demand_mw"] > 0


# ===========================================================================
# 8. Target Resolution: Never String '__aggregate__'
# ===========================================================================

def test_target_resolution_never_passes_aggregate_string():
    """Selecting individual location resolves to actual column; selecting aggregate resolves to descriptive label."""
    n = 50
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "Punjab": np.random.uniform(100, 200, n),
        "Haryana": np.random.uniform(150, 250, n),
    })

    # 1. Aggregate Mode
    prep_agg = prepare_dynamic_dataset(df, domain="electricity", target="__aggregate__")
    assert prep_agg["target"] != "__aggregate__"
    assert "Aggregate" in prep_agg["target"]
    assert prep_agg["geography_label"] == prep_agg["target"]

    # 2. Individual Location Mode
    prep_punjab = prepare_dynamic_dataset(df, domain="electricity", location="Punjab")
    assert prep_punjab["target"] == "Punjab"
    assert prep_punjab["geography_label"] == "Punjab"


# ===========================================================================
# 9. Datasets Without Geography Continue Normally
# ===========================================================================

def test_datasets_without_geography_continue_normally():
    """Ensure datasets without geography continue forecasting across the full dataset without breaking."""
    n = 50
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="D"),
        "temperature": np.random.uniform(15, 30, n),
        "humidity": np.random.uniform(40, 80, n),
        "demand_mw": np.random.uniform(5000, 8000, n),
    })

    schema = discover_features_and_resolve_schema(df, domain="electricity")
    geo = get_geographic_hierarchy(df)

    assert schema.is_multiseries is False
    assert schema.is_valid is True
    assert schema.target_column == "demand_mw"
    assert geo["has_geography"] is False
    assert "Geographic filtering is not available" in geo["message"]

    # Prepares normally
    prep = prepare_dynamic_dataset(df, domain="electricity")
    assert prep["geography_label"] == "All Geographies"
    assert prep["target"] == "demand_mw"
    assert prep["train_rows"] > 0


# ===========================================================================
# 10. Cache Isolation: Aggregate vs Single Location
# ===========================================================================

def test_cache_isolation_aggregate_vs_location():
    """Verify that different locations and the aggregate produce completely isolated cache fingerprints."""
    n = 50
    df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "RegionAlpha": np.random.uniform(1000, 2000, n),
        "RegionBeta": np.random.uniform(3000, 5000, n),
    })

    prep_agg = prepare_dynamic_dataset(df, domain="electricity", dataset_id="ds_test_123")
    prep_alpha = prepare_dynamic_dataset(df, domain="electricity", location="RegionAlpha", dataset_id="ds_test_123")
    prep_beta = prepare_dynamic_dataset(df, domain="electricity", location="RegionBeta", dataset_id="ds_test_123")

    assert prep_agg["fingerprint"] != prep_alpha["fingerprint"]
    assert prep_alpha["fingerprint"] != prep_beta["fingerprint"]
    assert prep_agg["features_dir"] != prep_alpha["features_dir"]
    assert prep_alpha["features_dir"] != prep_beta["features_dir"]


# ===========================================================================
# 11. PDF Report Unit Handling & Target Name
# ===========================================================================

def test_pdf_report_preserves_unspecified_unit_and_target_name():
    """Verify that PDF generation records 'Unspecified in source' and actual target name."""
    mock_report = {
        "run_id": "test_run_geo_safe_001",
        "domain": "electricity",
        "model": "random_forest",
        "target": "Punjab",
        "unit": "Unspecified in source",
        "unit_source": "unspecified",
        "geography_label": "Punjab",
        "dataset_info": {
            "geographic_scope": "Punjab",
            "unit": "Unspecified in source",
            "training_observations": 35,
            "validation_observations": 7,
            "test_observations": 8,
            "total_records": 50,
        },
        "model_info": {
            "selected_model": "Random Forest",
            "selection_criterion": "Lowest Validation RMSE",
        },
        "test_metrics": {
            "rmse": 12.34,
            "mae": 9.87,
            "mape": 4.12,
            "r2": 0.88,
        },
    }

    intel = _extract_dataset_intelligence(mock_report)
    assert intel["unit"] == "Unspecified in source"

    pdf_buffer = build_pdf_document(mock_report)
    assert isinstance(pdf_buffer, io.BytesIO)
    pdf_bytes = pdf_buffer.getvalue()
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")


# ===========================================================================
# 12. Real Pipeline Simulation with India Power Raw (if available)
# ===========================================================================

def test_india_power_raw_pipeline_simulation():
    """If india_power_raw (1).csv exists on the system, run full pipeline simulation."""
    candidate_paths = [
        Path(os.path.expanduser("~")) / "Downloads" / "india_power_raw (1).csv",
        ROOT / "tests" / "fixtures" / "india_power_raw (1).csv",
    ]
    raw_path = next((p for p in candidate_paths if p.exists()), None)
    if not raw_path:
        pytest.skip("india_power_raw (1).csv not found in Downloads or fixtures; skipping live file test.")

    df = pd.read_csv(raw_path)
    assert len(df) == 503
    assert len(df.columns) == 34

    schema = discover_features_and_resolve_schema(df, domain="electricity")
    assert schema.is_multiseries is True
    assert schema.time_column == "Unnamed: 0"
    assert "Unnamed: 0" not in schema.detected_geographic_series
    assert len(schema.detected_geographic_series) >= 30
    assert "Punjab" in schema.detected_geographic_series
    assert schema.unit == "Unspecified in source"
    assert schema.target_column != "__aggregate__"
    assert "Aggregate" in schema.target_column

    # Prepare dynamic dataset for aggregate
    prep_agg = prepare_dynamic_dataset(df, domain="electricity", dataset_id="india_power_raw")
    assert prep_agg["unit"] == "Unspecified in source"
    assert prep_agg["target"] != "__aggregate__"
    assert prep_agg["cleaned_rows"] > 0

    # Prepare dynamic dataset for Punjab
    prep_punjab = prepare_dynamic_dataset(df, domain="electricity", location="Punjab", dataset_id="india_power_raw")
    assert prep_punjab["unit"] == "Unspecified in source"
    assert prep_punjab["target"] == "Punjab"
    assert prep_punjab["fingerprint"] != prep_agg["fingerprint"]
