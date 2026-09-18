"""Focused regression tests for Dynamic Location Discovery, Selection, Eligibility, and N/A Hardening.

Covers the 15 core requirements:
1. Locations are discovered dynamically from dataset geographic values.
2. Locations with insufficient history (e.g. Ahmedabad) STILL appear in the location list.
3. Ahmedabad (21 total, 14 training) remains selectable and present in available_locations.
4. Random Forest: 14 < 15 -> RF ineligible with explicit N/A criteria.
5. Auto Select with no eligible models returns N/A without silent fallback.
6. Locations not present in dataset do not appear in the discovered location list.
7. API manually requesting nonexistent location returns [NO_RECORDS_FOUND].
8. Location with sufficient history (e.g. Mysuru) appears and forecasts normally.
9. Locations where some models are ineligible (e.g. ARIMA/SARIMA interleaved) remain selectable and Auto Select evaluates remaining eligible models.
10. Location change recalculates applicability and clears stale recommendations.
11. Dataset change recalculates available location list dynamically.
12. Auto Select never benchmarks or trains an inapplicable model.
13. Zero-record requests never trigger model training.
14. Zero test target leakage (validation split strictly drives ranking).
15. Existing Ahmedabad audit behavior preserved.
"""

from datetime import timedelta
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
from backend.app.services.dataset_service import get_dataset_geography_service
from ml_engine.preprocessing.intelligence import get_geographic_hierarchy
from ml_engine.preprocessing.dynamic_prep import prepare_dynamic_dataset, filter_df_by_geography, DynamicPrepError
from ml_engine.models.applicability import assess_model_applicability
from ml_engine.features.artifacts import materialize_forecasting_features


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

    app = create_app(CustomConfig)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        yield client


def test_1_locations_discovered_dynamically_from_dataset():
    """1. Location list is derived dynamically from actual dataset values (not model eligibility)."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=50, freq="D"),
        "sales": np.linspace(10, 50, 50),
        "city": ["LocationAlpha"] * 25 + ["LocationBeta"] * 25,
    })
    geo = get_geographic_hierarchy(df)
    assert geo["has_geography"] is True
    assert "LocationAlpha" in geo["available_locations"]
    assert "LocationBeta" in geo["available_locations"]
    assert "LocationGamma" not in geo["available_locations"]


def test_2_and_3_insufficient_history_location_remains_in_list_and_selectable():
    """2 & 3. A location with insufficient history (e.g. Ahmedabad: 21 rows, 14 train) STILL appears in location list."""
    demo_retail_dir = ROOT / "ml_engine" / "data" / "features" / "retail" / "features_1c0c92ff-264e-4398-b83c-e0e082b956b4"
    train_df = pd.read_parquet(demo_retail_dir / "train.parquet")
    geo = get_geographic_hierarchy(train_df)

    # Ahmedabad has only 21 observations in this dataset
    ahm_rows = len(train_df[train_df["city"].str.lower() == "ahmedabad"])
    assert ahm_rows == 21

    # CRITICAL: Ahmedabad MUST appear in available_locations despite insufficient forecasting history
    assert "Ahmedabad" in geo["available_locations"]

    # When selected, filtering yields exactly 21 total records and 14 training observations
    prep = prepare_dynamic_dataset(train_df, domain="retail", location="Ahmedabad")
    feat_dir = Path(prep["features_dir"])
    tr_df = pd.read_parquet(feat_dir / "train.parquet")
    assert len(tr_df) == 14


def test_4_rf_eligibility_explicit_criteria():
    """4. Random Forest: 14 < 15 -> RF ineligible with explicit criteria message."""
    demo_retail_dir = ROOT / "ml_engine" / "data" / "features" / "retail" / "features_1c0c92ff-264e-4398-b83c-e0e082b956b4"
    train_df = pd.read_parquet(demo_retail_dir / "train.parquet")
    prep = prepare_dynamic_dataset(train_df, domain="retail", location="Ahmedabad")
    app = assess_model_applicability(prep["features_dir"], domain="retail")

    rf_result = app["models"]["random_forest"]
    assert rf_result["eligible"] is False
    assert rf_result["reason_code"] == "INSUFFICIENT_HISTORY"
    assert "N/A — Eligible training observation criterion not met" in rf_result["reason"]
    assert "14 usable training observations available" in rf_result["reason"]
    assert "minimum 15 required" in rf_result["reason"]


def test_5_autoselect_no_eligible_models_returns_na_without_fallback():
    """5. Auto Select: when no eligible models exist, returns structured N/A without fallback."""
    payload = {
        "domain": "retail",
        "dataset_id": "demo_retail",
        "location": "Ahmedabad",
        "models": None,
    }
    with pytest.raises(MLError) as exc_info:
        compare_service(payload)

    assert exc_info.value.code == 400
    assert "N/A — No applicable forecasting model for the selected location" in exc_info.value.message
    assert "No eligible models available for comparison" in exc_info.value.message
    assert "14 usable training observations available" in exc_info.value.message
    assert "21 usable training observations available" not in exc_info.value.message


def test_6_location_not_present_does_not_appear_in_list():
    """6. A location not present in dataset does NOT appear in the location list."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="D"),
        "sales": range(30),
        "city": ["Pune"] * 30,
    })
    geo = get_geographic_hierarchy(df)
    assert "Pune" in geo["available_locations"]
    assert "Agra" not in geo["available_locations"]
    assert "Ahmedabad" not in geo["available_locations"]


def test_7_api_request_for_nonexistent_location_returns_no_records_found(test_client):
    """7. API manually requesting a nonexistent location returns [NO_RECORDS_FOUND] HTTP 400."""
    payload = {
        "domain": "retail",
        "dataset_id": "demo_retail",
        "location": "NonexistentAtlantisCity",
        "model": "random_forest",
    }
    resp = test_client.post("/api/v1/forecasting/forecast", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    err_msg = data.get("error") or data.get("message") or ""
    assert "NO_RECORDS_FOUND" in err_msg
    assert "No records found" in err_msg


def test_8_location_with_sufficient_history_forecasts_normally():
    """8. A location with sufficient history (Mysuru: 25 training obs) appears and forecasts normally."""
    demo_retail_dir = ROOT / "ml_engine" / "data" / "features" / "retail" / "features_1c0c92ff-264e-4398-b83c-e0e082b956b4"
    train_df = pd.read_parquet(demo_retail_dir / "train.parquet")
    geo = get_geographic_hierarchy(train_df)
    assert "Mysuru" in geo["available_locations"]

    res = compare_service({
        "domain": "retail",
        "dataset_id": "demo_retail",
        "location": "Mysuru",
        "models": None,
    })
    assert res["recommended_model"] in ("hybrid", "random_forest", "xgboost", "lstm")
    assert len(res["ranking"]) >= 2


def test_9_some_models_ineligible_location_remains_selectable_and_evaluates_remaining():
    """9. When some models are ineligible (ARIMA/SARIMA interleaved panel), location remains selectable and remaining models are evaluated."""
    res = compare_service({
        "domain": "retail",
        "dataset_id": "demo_retail",
        "location": "Mysuru",
        "models": None,
    })
    ranking_models = [r["model"] for r in res["ranking"]]
    # Ineligible models are excluded
    assert "arima" not in ranking_models
    assert "sarima" not in ranking_models
    # Eligible models are ranked
    assert "hybrid" in ranking_models
    assert "random_forest" in ranking_models
    assert "xgboost" in ranking_models


def test_10_location_change_clears_stale_recommendation():
    """10. Changing from All Geographies -> Ahmedabad recalculates dynamically and does NOT retain stale recommendation."""
    res_all = compare_service({
        "domain": "retail",
        "dataset_id": "demo_retail",
        "location": "All Geographies",
        "models": None,
    })
    assert res_all["recommended_model"] in ("hybrid", "xgboost", "random_forest")

    # Switching to Ahmedabad MUST fail with N/A, not reuse All Geographies recommendation
    with pytest.raises(MLError) as exc:
        compare_service({
            "domain": "retail",
            "dataset_id": "demo_retail",
            "location": "Ahmedabad",
            "models": None,
        })
    assert "N/A — No applicable forecasting model for the selected location" in exc.value.message


def test_11_dataset_change_recalculates_location_list():
    """11. Switching datasets dynamically recalculates available locations from the new dataset."""
    df_a = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"),
        "sales": range(20),
        "city": ["CityA1", "CityA2"] * 10,
    })
    df_b = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"),
        "sales": range(20),
        "city": ["CityB1", "CityB2"] * 10,
    })
    geo_a = get_geographic_hierarchy(df_a)
    geo_b = get_geographic_hierarchy(df_b)

    assert set(geo_a["available_locations"]) == {"CityA1", "CityA2"}
    assert set(geo_b["available_locations"]) == {"CityB1", "CityB2"}
    assert not set(geo_a["available_locations"]).intersection(set(geo_b["available_locations"]))


def test_12_autoselect_never_benchmarks_inapplicable_model(monkeypatch):
    """12. Auto Select candidate filtering strictly precedes any benchmarking or training."""
    from ml_engine.models import trainer

    benchmarked_models = []
    original_run_pipeline = trainer.run_training_pipeline

    def spy_run_pipeline(*args, **kwargs):
        benchmarked_models.append(kwargs.get("model_name"))
        return original_run_pipeline(*args, **kwargs)

    monkeypatch.setattr(trainer, "run_training_pipeline", spy_run_pipeline)

    compare_service({
        "domain": "retail",
        "dataset_id": "demo_retail",
        "location": "Mysuru",
        "models": None,
    })

    assert "arima" not in benchmarked_models
    assert "sarima" not in benchmarked_models


def test_13_zero_record_case_never_triggers_training(monkeypatch):
    """13. Zero-record request immediately aborts with NO_RECORDS_FOUND without triggering model training."""
    from ml_engine.models import trainer

    trained_models = []
    def spy_run_pipeline(*args, **kwargs):
        trained_models.append(kwargs.get("model_name"))
        return original_run_pipeline(*args, **kwargs)

    monkeypatch.setattr(trainer, "run_training_pipeline", spy_run_pipeline)

    with pytest.raises(MLError) as exc:
        forecast_service({
            "domain": "retail",
            "dataset_id": "demo_retail",
            "location": "GhostCityNotInDataset",
            "model": "random_forest",
        })

    assert "NO_RECORDS_FOUND" in exc.value.message
    assert len(trained_models) == 0


def test_14_zero_test_target_leakage():
    """14. Auto Select uses only validation RMSE; held-out test split is strictly reserved."""
    res = compare_service({
        "domain": "retail",
        "dataset_id": "demo_retail",
        "location": "Mysuru",
        "models": None,
    })
    assert res["selection_split"] == "validation"
    assert res["selection_metric"] == "rmse"
    for r in res["ranking"]:
        assert "validation_metrics" in r
        assert "rmse" in r["validation_metrics"]
    val_rmses = [r["validation_metrics"]["rmse"] for r in res["ranking"]]
    assert val_rmses == sorted(val_rmses)


def test_15_existing_ahmedabad_audit_regression_passes(test_client):
    """15. Manual Random Forest on Ahmedabad returns structured HTTP 400 with INSUFFICIENT_HISTORY."""
    payload = {
        "domain": "retail",
        "dataset_id": "demo_retail",
        "location": "Ahmedabad",
        "model": "random_forest",
    }
    resp = test_client.post("/api/v1/forecasting/forecast", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    err_msg = data.get("error") or data.get("message") or ""
    assert "INSUFFICIENT_HISTORY" in err_msg
    assert "14 usable training observations available" in err_msg
    assert "minimum 15 required for Random Forest" in err_msg
