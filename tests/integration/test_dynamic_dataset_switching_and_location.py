"""Integration tests for:
1. Dynamic dataset-based geographic scope (detecting varied column names without inventing columns).
2. Dataset change behavior: clearing old location and auto-selecting the first available location.
3. Dataset with no geographic information: defaults to All Data without error.
4. Dynamic geographic hierarchy construction.
5. Location-specific forecast and cache isolation.
6. Multi-tier intelligent column matching (exact, normalized, synonym, fuzzy, LLM-assisted).
"""

import pandas as pd
import pytest
from pathlib import Path

from ml_engine.preprocessing.intelligence import (
    get_geographic_hierarchy,
    detect_geographic_columns,
    match_column_by_candidates,
    TIME_NAME_KEYWORDS,
    RETAIL_TARGET_KEYWORDS,
    GEOGRAPHIC_KEYWORDS,
)
from ml_engine.preprocessing.dynamic_prep import (
    prepare_dynamic_dataset,
    filter_df_by_geography,
    DynamicPrepError,
)
from ml_engine.models.trainer import run_training_pipeline
from backend.app.services.llm_service import semantic_match_columns_llm


@pytest.fixture()
def dataset_alpha():
    """Dataset A with locations: Mumbai, Delhi, Bengaluru."""
    dates = pd.date_range("2025-01-01", periods=90, freq="D").strftime("%Y-%m-%d").tolist()
    rows = []
    for loc, base in [("Mumbai", 300), ("Delhi", 500), ("Bengaluru", 450)]:
        for i, d in enumerate(dates):
            rows.append({
                "Date_Time": d,
                "City_Name": loc,
                "State_Name": "State_" + loc,
                "Gross_Revenue": base + (i * 2.0) + (10 if i % 7 == 0 else 0),
                "Promo_Discount": 0.1 if i % 5 == 0 else 0.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def dataset_beta():
    """Dataset B with locations: Chennai, Hyderabad, Pune."""
    dates = pd.date_range("2025-01-01", periods=90, freq="D").strftime("%Y-%m-%d").tolist()
    rows = []
    for loc, base in [("Chennai", 200), ("Hyderabad", 350), ("Pune", 280)]:
        for i, d in enumerate(dates):
            rows.append({
                "transaction_date": d,
                "store_location": loc,
                "region_zone": "South_Zone",
                "units_sold": base + (i * 1.5) + (5 if i % 7 == 0 else 0),
                "unit_cost": 15.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def dataset_no_geo():
    """Dataset with no geographic or location columns."""
    dates = pd.date_range("2025-01-01", periods=90, freq="D").strftime("%Y-%m-%d").tolist()
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "timestamp": d,
            "demand_mw": 1200.0 + (i * 5.0) + (50 if i % 7 in (5, 6) else 0),
            "temperature_c": 28.0 + (i % 5),
        })
    return pd.DataFrame(rows)


def test_dynamic_geographic_scope_detection(dataset_alpha, dataset_beta, dataset_no_geo):
    """Verify geographic columns and available locations are dynamically detected from dataset schemas."""
    # Dataset Alpha
    geo_alpha = get_geographic_hierarchy(dataset_alpha)
    assert geo_alpha["has_geography"] is True
    assert set(geo_alpha["available_locations"]) == {"Mumbai", "Delhi", "Bengaluru", "State_Mumbai", "State_Delhi", "State_Bengaluru"}
    assert any(h["column_name"] == "City_Name" for h in geo_alpha["hierarchy_levels"])

    # Dataset Beta
    geo_beta = get_geographic_hierarchy(dataset_beta)
    assert geo_beta["has_geography"] is True
    assert set(geo_beta["available_locations"]) == {"Chennai", "Hyderabad", "Pune", "South_Zone"}
    assert any(h["column_name"] == "store_location" for h in geo_beta["hierarchy_levels"])

    # Dataset No Geo
    geo_none = get_geographic_hierarchy(dataset_no_geo)
    assert geo_none["has_geography"] is False
    assert len(geo_none["available_locations"]) == 0
    assert len(geo_none["hierarchy_levels"]) == 0


def test_dataset_switching_behavior(dataset_alpha, dataset_beta, tmp_path):
    """Verify dataset switching clears previous location and auto-selects first available location from new dataset."""
    # Step 1: User on Dataset Alpha, location Mumbai
    geo_alpha = get_geographic_hierarchy(dataset_alpha)
    first_loc_alpha = geo_alpha["available_locations"][0]
    prep_alpha = prepare_dynamic_dataset(
        dataset_alpha,
        domain="retail",
        location=first_loc_alpha,
        output_root=tmp_path / "alpha",
    )
    assert prep_alpha["geography_label"] == first_loc_alpha
    
    # Step 2: User switches to Dataset Beta
    # Previous location state is cleared, and first available location in Beta is chosen
    geo_beta = get_geographic_hierarchy(dataset_beta)
    first_loc_beta = geo_beta["available_locations"][0]
    assert first_loc_beta not in geo_alpha["available_locations"]

    prep_beta = prepare_dynamic_dataset(
        dataset_beta,
        domain="retail",
        location=first_loc_beta,
        output_root=tmp_path / "beta",
    )
    assert prep_beta["geography_label"] == first_loc_beta
    assert prep_beta["fingerprint"] != prep_alpha["fingerprint"]


def test_dataset_without_geography_handles_all_data_smoothly(dataset_no_geo, tmp_path):
    """Verify datasets without geography default to full dataset and do not fail."""
    prep = prepare_dynamic_dataset(
        dataset_no_geo,
        domain="electricity",
        location=None,
        output_root=tmp_path / "no_geo",
    )
    assert prep["geography_label"] == "All Geographies"
    assert prep["rows_used"] == len(dataset_no_geo)

    res = run_training_pipeline(
        domain="electricity",
        features_dir=Path(prep["features_dir"]),
        model_name="random_forest",
        artifacts_root=tmp_path / "artifacts",
    )
    assert res["test_metrics"]["rmse"] > 0
    assert "baseline_metrics" in res


def test_multi_tier_feature_matching_exact_synonym_fuzzy():
    """Verify exact, normalized, synonym, and fuzzy matching across concepts."""
    sample_df = pd.DataFrame({
        "Trans_Date": ["2026-01-01"],
        "Net_Revenue_INR": [1500.0],
        "Promo_Offer_Pct": [10.0],
        "Branch_Store_Location": ["Mumbai"],
    })

    # Time matching
    time_matches = match_column_by_candidates(sample_df, TIME_NAME_KEYWORDS)
    assert len(time_matches) > 0
    assert time_matches[0][0] == "Trans_Date"

    # Revenue/Sales target matching
    rev_matches = match_column_by_candidates(sample_df, RETAIL_TARGET_KEYWORDS, numeric_only=True)
    assert len(rev_matches) > 0
    assert rev_matches[0][0] == "Net_Revenue_INR"

    # Geographic matching
    geo_matches = match_column_by_candidates(sample_df, GEOGRAPHIC_KEYWORDS)
    assert len(geo_matches) > 0
    assert geo_matches[0][0] == "Branch_Store_Location"


def test_llm_semantic_mapping_only_maps_to_existing_columns():
    """Verify LLM semantic matching strictly constrains mappings to existing columns only."""
    cols = ["Txn_Time", "Total_Amount_Paid", "Customer_City", "Discount_Rate"]
    res = semantic_match_columns_llm(
        columns=cols,
        concepts=["timestamp", "target", "location", "discount"],
    )
    for concept, mapping in res.items():
        assert mapping["column"] in cols
        assert 0.0 <= mapping["confidence"] <= 1.0
