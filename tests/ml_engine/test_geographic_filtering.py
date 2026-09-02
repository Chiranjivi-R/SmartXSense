"""Unit tests for geographic hierarchy discovery, dynamic dataset preparation, and row-count validation."""

from pathlib import Path
import pandas as pd
import pytest

from ml_engine.preprocessing.dynamic_prep import (
    DynamicPrepError,
    compute_dynamic_fingerprint,
    filter_df_by_geography,
    prepare_dynamic_dataset,
)
from ml_engine.preprocessing.intelligence import get_geographic_hierarchy


@pytest.fixture()
def geo_df():
    dates = [f"2026-01-{i:02d}" for i in range(1, 21)]
    return pd.DataFrame({
        "timestamp": dates,
        "Country": ["India"] * 10 + ["USA"] * 10,
        "State": ["Karnataka"] * 6 + ["Maharashtra"] * 4 + ["California"] * 10,
        "District": ["Bengaluru Urban"] * 4 + ["Mysuru"] * 2 + ["Mumbai"] * 4 + ["LA"] * 10,
        "City": ["Bengaluru"] * 4 + ["Mysuru"] * 2 + ["Mumbai"] * 4 + ["Los Angeles"] * 10,
        "sales": [100 + i * 10 for i in range(20)],
        "promo": [i % 2 for i in range(20)],
    })


def test_geographic_hierarchy_discovery(geo_df):
    geo_meta = get_geographic_hierarchy(geo_df)
    assert geo_meta["has_geography"] is True
    assert set(geo_meta["levels"]["countries"]) == {"India", "USA"}
    assert set(geo_meta["levels"]["states"]) == {"California", "Karnataka", "Maharashtra"}


def test_cascading_geographic_filtering(geo_df):
    # Filter by Country = India
    india_meta = get_geographic_hierarchy(geo_df, {"country": "India"})
    assert set(india_meta["levels"]["states"]) == {"Karnataka", "Maharashtra"}
    assert india_meta["filtered_rows"] == 10

    # Filter by State = Karnataka
    karnataka_meta = get_geographic_hierarchy(geo_df, {"country": "India", "state": "Karnataka"})
    assert set(karnataka_meta["levels"]["cities"]) == {"Bengaluru", "Mysuru"}
    assert karnataka_meta["filtered_rows"] == 6

    # Filter by City = Bengaluru
    bengaluru_meta = get_geographic_hierarchy(geo_df, {"country": "India", "state": "Karnataka", "city": "Bengaluru"})
    assert bengaluru_meta["filtered_rows"] == 4


def test_filter_df_by_geography(geo_df):
    filtered, label = filter_df_by_geography(geo_df, {"country": "India", "state": "Karnataka", "city": "Bengaluru"})
    assert len(filtered) == 4
    assert label == "Bengaluru, Karnataka, India"


def test_empty_geographic_selection_fails_clearly(geo_df):
    with pytest.raises(DynamicPrepError) as exc_info:
        prepare_dynamic_dataset(geo_df, domain="retail", geo_filters={"country": "Japan"})
    assert "0 rows selected" in str(exc_info.value)
    assert exc_info.value.code == 400


def test_tiny_geographic_subset_handled_safely(geo_df):
    with pytest.raises(DynamicPrepError) as exc_info:
        prepare_dynamic_dataset(geo_df, domain="retail", geo_filters={"country": "India", "state": "Karnataka", "city": "Bengaluru"})
    assert "minimum 10 required" in str(exc_info.value)
    assert exc_info.value.code == 400


def test_fingerprint_changes_with_geography_and_target(geo_df):
    fp1 = compute_dynamic_fingerprint(geo_df, "retail", "sales", "All Geographies")
    fp2 = compute_dynamic_fingerprint(geo_df.head(10), "retail", "sales", "India")
    fp3 = compute_dynamic_fingerprint(geo_df, "retail", "promo", "All Geographies")

    assert fp1 != fp2
    assert fp1 != fp3
    assert fp2 != fp3


def test_dataset_without_geography_prepares_normally():
    dates = [f"2026-01-{i:02d}" for i in range(1, 25)]
    no_geo_df = pd.DataFrame({
        "timestamp": dates,
        "sales": [100 + i for i in range(24)],
        "price": [10.0 + i for i in range(24)],
    })
    result = prepare_dynamic_dataset(no_geo_df, domain="retail")
    assert result["geography_label"] == "All Geographies"
    assert result["rows_used"] == 24
    assert Path(result["features_dir"]).exists()
