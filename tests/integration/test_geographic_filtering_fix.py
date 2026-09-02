import pytest
import pandas as pd
from pathlib import Path

from ml_engine.preprocessing.intelligence import get_geographic_hierarchy, _is_no_filter_val
from ml_engine.preprocessing.dynamic_prep import filter_df_by_geography, DynamicPrepError, prepare_dynamic_dataset
from backend.app.services.ml_service import resolve_dynamic_features_and_context, forecast_service, MLError


@pytest.fixture
def retail_train_df():
    features_dir = Path("ml_engine/data/features/retail/features_1c0c92ff-264e-4398-b83c-e0e082b956b4")
    return pd.read_parquet(features_dir / "train.parquet")


def test_is_no_filter_val():
    assert _is_no_filter_val(None) is True
    assert _is_no_filter_val("") is True
    assert _is_no_filter_val("   ") is True
    assert _is_no_filter_val("All") is True
    assert _is_no_filter_val("All Data") is True
    assert _is_no_filter_val("All Geographies") is True
    assert _is_no_filter_val("All Cities") is True
    assert _is_no_filter_val("All States") is True
    assert _is_no_filter_val("All City Tiers") is True
    assert _is_no_filter_val("none") is True
    assert _is_no_filter_val("null") is True
    assert _is_no_filter_val("nan") is True
    assert _is_no_filter_val("Agra") is False
    assert _is_no_filter_val("Bangalore") is False
    assert _is_no_filter_val("Karnataka") is False


def test_get_geographic_hierarchy_with_agra(retail_train_df):
    geo_meta = get_geographic_hierarchy(retail_train_df, {"location": "Agra"})
    assert geo_meta["has_geography"] is True
    assert geo_meta["total_rows"] == len(retail_train_df)
    assert geo_meta["filtered_rows"] == 30
    assert "Agra" in geo_meta["available_locations"]


def test_get_geographic_hierarchy_with_bangalore_and_karnataka(retail_train_df):
    blr_meta = get_geographic_hierarchy(retail_train_df, {"location": "Bangalore"})
    assert blr_meta["filtered_rows"] == 24

    kar_meta = get_geographic_hierarchy(retail_train_df, {"state": "Karnataka"})
    assert kar_meta["filtered_rows"] == 88


def test_get_geographic_hierarchy_with_all_placeholders(retail_train_df):
    for placeholder in ["All Data", "All Cities", "All States", "All City Tiers", "All", "", None]:
        res = get_geographic_hierarchy(retail_train_df, {"location": placeholder})
        assert res["filtered_rows"] == len(retail_train_df)


def test_filter_df_by_geography_valid_locations(retail_train_df):
    filtered_agra, label_agra = filter_df_by_geography(retail_train_df, location="Agra")
    assert len(filtered_agra) == 30
    assert label_agra == "Agra"

    filtered_blr, label_blr = filter_df_by_geography(retail_train_df, location="Bangalore")
    assert len(filtered_blr) == 24
    assert label_blr == "Bangalore"


def test_filter_df_by_geography_invalid_location_raises_400(retail_train_df):
    with pytest.raises(DynamicPrepError) as exc_info:
        filter_df_by_geography(retail_train_df, location="Atlantis")
    assert exc_info.value.code == 400
    assert "Atlantis" in str(exc_info.value)


def test_resolve_dynamic_features_agra():
    res_dir, ctx = resolve_dynamic_features_and_context({
        "dataset_id": "demo_retail",
        "domain": "retail",
        "location": "Agra",
    })
    assert res_dir.exists()
    assert ctx["location"] == "Agra"
    assert ctx["rows_used"] == 30
    assert (res_dir / "train.parquet").exists()


def test_forecast_service_with_agra():
    res = forecast_service({
        "dataset_id": "demo_retail",
        "domain": "retail",
        "model": "random_forest",
        "location": "Agra",
    })
    assert res["model"] == "random_forest"
    assert res["training_rows"] > 0
    assert len(res["predictions"]) > 0
    assert res["test_metrics"]["mae"] > 0


def test_forecast_service_with_invalid_location_returns_400():
    with pytest.raises(MLError) as exc_info:
        forecast_service({
            "dataset_id": "demo_retail",
            "domain": "retail",
            "model": "random_forest",
            "location": "Atlantis",
        })
    assert exc_info.value.code == 400


def test_forecast_service_on_non_geographic_dataset_returns_400_for_location():
    with pytest.raises(MLError) as exc_info:
        forecast_service({
            "dataset_id": "demo_electricity",
            "domain": "electricity",
            "model": "random_forest",
            "location": "Agra",
        })
    assert exc_info.value.code == 400
    assert "Location-specific forecasting is unavailable" in str(exc_info.value)
