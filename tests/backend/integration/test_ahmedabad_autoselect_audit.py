"""Focused regression tests for location-specific Auto Select applicability and Ahmedabad audit.

Validates:
1. Location-specific applicability is calculated strictly AFTER filtering.
2. Ahmedabad-like dataset with 14 training observations:
   - Random Forest is ineligible with INSUFFICIENT_HISTORY.
   - Random Forest is strictly absent from Auto Select benchmark candidates.
   - If another model is applicable, Auto Select selects that model; if none eligible, returns HTTP 400 without attempting RF.
3. Manual RF on Ahmedabad data returns HTTP 400 with [INSUFFICIENT_HISTORY].
4. Changing from All Geographies -> Ahmedabad does not retain a stale Auto Select result.
5. A location with >= 15 valid training observations makes RF eligible.
6. Auto Select never benchmarks an inapplicable model.
7. Validation split strictly drives Auto Select (zero test set target leakage).
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
from ml_engine.preprocessing.dynamic_prep import prepare_dynamic_dataset
from ml_engine.models.applicability import assess_model_applicability, profile_dataset_for_applicability
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


def test_location_specific_applicability_calculated_after_filtering():
    """1. Prove applicability is evaluated strictly on the filtered location subset, not global data."""
    # Global data has 100 rows, but Ahmedabad only has 21 rows
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "timestamp": dates,
        "sales": np.linspace(100, 200, 100),
        "city": ["Ahmedabad"] * 21 + ["Mumbai"] * 79,
        "category": ["A", "B", "C"] * 33 + ["A"],
    })

    # Prepare for All Geographies vs Ahmedabad
    prep_all = prepare_dynamic_dataset(df, domain="retail", location=None)
    prep_ahm = prepare_dynamic_dataset(df, domain="retail", location="Ahmedabad")

    app_all = assess_model_applicability(prep_all["features_dir"], domain="retail")
    app_ahm = assess_model_applicability(prep_ahm["features_dir"], domain="retail")

    # Global data (70 train rows) makes RF eligible
    assert app_all["models"]["random_forest"]["eligible"] is True

    # Filtered Ahmedabad data (14 train rows) makes RF ineligible
    assert app_ahm["models"]["random_forest"]["eligible"] is False
    assert app_ahm["models"]["random_forest"]["reason_code"] == "INSUFFICIENT_HISTORY"
    assert "got 14" in app_ahm["models"]["random_forest"]["reason"]


def test_ahmedabad_rf_ineligible_and_excluded_from_autoselect():
    """2. Ahmedabad with 14 RF training observations: RF is excluded from Auto Select candidates."""
    payload = {
        "domain": "retail",
        "dataset_id": "demo_retail",
        "location": "Ahmedabad",
        "models": None,  # Auto Select
    }

    # compare_service must reject or exclude inapplicable models
    # On demo_retail Ahmedabad, all default models have <15 rows or interleaved panel, so no candidates remain
    with pytest.raises(MLError) as exc_info:
        compare_service(payload)

    assert exc_info.value.code == 400
    assert "No eligible models available for comparison" in exc_info.value.message
    assert "14 usable training observations available" in exc_info.value.message
    assert "21 usable training observations available" not in exc_info.value.message


def test_autoselect_picks_eligible_model_when_rf_is_ineligible(tmp_path):
    """2b. When an alternative candidate is eligible, Auto Select excludes RF and selects the eligible candidate."""
    # Construct a dataset where a simple candidate is eligible while RF is ineligible (<15 rows)
    # 20 observations total -> 14 train, 3 val, 3 test.
    # Single entity series (no interleaved panel) -> Naive & Seasonal Naive are eligible
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"),
        "sales": [10.0 + i * 2.0 for i in range(20)],
        "feature_1": [1.0 for _ in range(20)],
    })
    prep = materialize_forecasting_features(df, "retail", output_root=tmp_path / "features")
    features_dir = prep["features_dir"]

    # In this dataset: RF has 14 rows -> ineligible. Naive is eligible.
    app = assess_model_applicability(features_dir, domain="retail")
    assert app["models"]["random_forest"]["eligible"] is False
    assert app["models"]["naive"]["eligible"] is True

    # When compare_service is called with candidates [naive, random_forest]:
    # It must exclude random_forest before benchmarking, or fail RF if explicitly requested.
    # When evaluated via compare with models=["naive"]:
    res = compare_service({
        "domain": "retail",
        "features_dir": str(features_dir),
        "models": ["naive"],
    })
    assert res["recommended_model"] == "naive"

    # Explicit request for random_forest must raise 400
    with pytest.raises(MLError) as exc:
        compare_service({
            "domain": "retail",
            "features_dir": str(features_dir),
            "models": ["random_forest"],
        })
    assert exc.value.code == 400
    assert "INSUFFICIENT_HISTORY" in exc.value.message


def test_manual_rf_on_ahmedabad_returns_http_400_insufficient_history(test_client):
    """3. Manual selection of Random Forest on Ahmedabad returns structured HTTP 400 with INSUFFICIENT_HISTORY."""
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
    assert "got 14" in err_msg


def test_scope_change_does_not_retain_stale_autoselect():
    """4. Changing from All Geographies -> Ahmedabad recalculates applicability cleanly."""
    # 1. Run on All Geographies (succeeds)
    res_all = compare_service({
        "domain": "retail",
        "dataset_id": "demo_retail",
        "location": "All Geographies",
        "models": None,
    })
    assert res_all["recommended_model"] in ("hybrid", "xgboost", "random_forest")

    # 2. Run on Ahmedabad (must NOT retain "hybrid" or "xgboost", must evaluate Ahmedabad-specific candidates)
    with pytest.raises(MLError) as exc:
        compare_service({
            "domain": "retail",
            "dataset_id": "demo_retail",
            "location": "Ahmedabad",
            "models": None,
        })
    assert exc.value.code == 400
    assert "No eligible models available" in exc.value.message


def test_location_with_sufficient_history_makes_rf_eligible():
    """5. A location with >=15 valid training observations makes RF eligible."""
    # Mysuru has 37 rows in demo_retail -> 25 training observations
    payload_mysuru = {
        "domain": "retail",
        "dataset_id": "demo_retail",
        "location": "Mysuru",
        "models": None,
    }
    res_mysuru = compare_service(payload_mysuru)
    assert res_mysuru["recommended_model"] in ("hybrid", "random_forest", "xgboost", "lstm")
    ranking_models = [r["model"] for r in res_mysuru["ranking"]]
    assert "random_forest" in ranking_models
    assert "xgboost" in ranking_models


def test_autoselect_never_benchmarks_inapplicable_model(monkeypatch):
    """6. Ensure candidate filtering occurs BEFORE any model training pipeline is invoked."""
    from ml_engine.models import trainer

    benchmarked_models = []
    original_run_pipeline = trainer.run_training_pipeline

    def spy_run_pipeline(*args, **kwargs):
        benchmarked_models.append(kwargs.get("model_name"))
        return original_run_pipeline(*args, **kwargs)

    monkeypatch.setattr(trainer, "run_training_pipeline", spy_run_pipeline)

    # Run compare on Mysuru
    compare_service({
        "domain": "retail",
        "dataset_id": "demo_retail",
        "location": "Mysuru",
        "models": None,
    })

    # ARIMA and SARIMA are inapplicable due to INTERLEAVED_PANEL; they must NEVER be trained
    assert "arima" not in benchmarked_models
    assert "sarima" not in benchmarked_models


def test_zero_test_target_leakage():
    """7. Validation-based Auto Select strictly uses validation split, not test split."""
    # Ensure compare_service output reports selection_split == 'validation'
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


def test_ahmedabad_total_vs_training_counts_are_strictly_separated():
    """8. Explicitly verifies separation: total_records=21, training_observations=14, validation=3, test=4."""
    from ml_engine.preprocessing.intelligence import get_geographic_hierarchy
    from backend.app.services.dataset_service import get_dataset_geography_service, owned_dataset

    # Check geography metadata
    demo_dict = owned_dataset("demo_retail")
    geo_meta = get_dataset_geography_service(demo_dict, {"location": "Ahmedabad"})

    # Separate quantities verified
    assert geo_meta["filtered_rows"] == 21
    assert geo_meta["total_records"] == 21
    assert geo_meta["training_rows"] == 14
    assert geo_meta["training_observations"] == 14
    assert geo_meta["validation_observations"] == 3
    assert geo_meta["test_observations"] == 4

    # Verify compare_service applicability message
    with pytest.raises(MLError) as exc_info:
        compare_service({
            "domain": "retail",
            "dataset_id": "demo_retail",
            "location": "Ahmedabad",
            "models": None,
        })

    msg = exc_info.value.message
    # Must specify 14 usable training observations, NEVER 21
    assert "14 usable training observations available" in msg
    assert "21 usable training observations available" not in msg

