"""Integration tests verifying unified 10-model forecasting portfolio across Retail and Electricity."""

from datetime import timedelta
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app import create_app
from app.extensions import db
from ml_engine.models.applicability import assess_model_applicability
from ml_engine.models.capability_matrix import (
    DEFAULT_COMPARE_MODELS,
    build_model_capability_matrix,
    get_models_catalog_payload,
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


@pytest.fixture()
def client(tmp_path):
    class CustomTestConfig(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts")
        UPLOAD_FOLDER = str(tmp_path / "uploads")
        PROCESSED_DATA_FOLDER = str(tmp_path / "processed")
        FEATURES_DATA_FOLDER = str(tmp_path / "features")

    app = create_app(CustomTestConfig)
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


ALL_10_MODELS = [
    "naive",
    "seasonal_naive",
    "arima",
    "sarima",
    "random_forest",
    "xgboost",
    "trend_xgboost",
    "lstm",
    "cnn_lstm",
    "hybrid",
]


def test_models_catalog_exposes_all_10_models(client):
    """GET /api/v1/dashboard/models returns all 10 models in the unified portfolio."""
    resp = client.get("/api/v1/dashboard/models")
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    model_keys = [item["key"] for item in data["models"]]
    for m in ALL_10_MODELS:
        assert m in model_keys, f"Model '{m}' missing from models catalog"


def test_electricity_portfolio_applicability_and_execution(client):
    """Verify that all 10 models are eligible on electricity and execute successfully."""
    elec_dir = Path("ml_engine/data/features/electricity/features_23889e57-aa6d-4a50-930c-874f6a963a25")
    app_report = assess_model_applicability(elec_dir / "train.parquet", target_column="demand_mw", domain="electricity")
    models_app = app_report["models"]

    # All 10 models must be eligible on canonical electricity data
    for m in ALL_10_MODELS:
        assert models_app[m]["eligible"] is True, f"Electricity model '{m}' should be eligible: {models_app[m]}"

    # Test execution of representative models across families:
    # 1. Baseline
    resp = client.post("/api/v1/forecasting/train", json={"domain": "electricity", "model": "naive", "prediction_limit": 5})
    assert resp.status_code == 200
    assert resp.get_json()["data"]["model"] == "naive"

    # 2. Statistical
    resp = client.post("/api/v1/forecasting/train", json={"domain": "electricity", "model": "arima", "order": [1, 0, 1], "training_window": 168, "prediction_limit": 5})
    assert resp.status_code == 200
    assert resp.get_json()["data"]["model"] == "arima"

    # 3. Tabular ensemble
    resp = client.post("/api/v1/forecasting/train", json={"domain": "electricity", "model": "random_forest", "n_estimators": 5, "prediction_limit": 5})
    assert resp.status_code == 200
    assert resp.get_json()["data"]["model"] == "random_forest"

    # 4. Trend XGBoost
    resp = client.post("/api/v1/forecasting/train", json={"domain": "electricity", "model": "trend_xgboost", "n_estimators": 5, "prediction_limit": 5})
    assert resp.status_code == 200
    assert resp.get_json()["data"]["model"] == "trend_xgboost"

    # 5. Hybrid
    resp = client.post("/api/v1/forecasting/train", json={"domain": "electricity", "model": "hybrid", "components": ["random_forest", "xgboost"], "n_estimators": 5, "prediction_limit": 5})
    assert resp.status_code == 200
    assert resp.get_json()["data"]["model"] == "hybrid"


def test_retail_portfolio_applicability_and_data_driven_rejection(client):
    """Verify that retail evaluates all 10 models and provides data-driven explanations."""
    retail_dir = Path("ml_engine/data/features/retail/features_1c0c92ff-264e-4398-b83c-e0e082b956b4")
    app_report = assess_model_applicability(retail_dir / "train.parquet", target_column="sales", domain="retail")
    models_app = app_report["models"]

    # Check data-driven reasons:
    # arima and sarima ineligible due to interleaved multi-entity panel
    assert models_app["arima"]["eligible"] is False
    assert models_app["arima"]["reason_code"] == "INTERLEAVED_PANEL"
    assert models_app["sarima"]["eligible"] is False
    assert models_app["sarima"]["reason_code"] == "INTERLEAVED_PANEL"

    # trend_xgboost ineligible due to no hourly lag structure (retail has daily lags 1, 7, 14, 28)
    assert models_app["trend_xgboost"]["eligible"] is False
    assert models_app["trend_xgboost"]["reason_code"] == "NO_HOURLY_LAG_STRUCTURE"

    # random_forest, xgboost, hybrid, lstm, cnn_lstm, naive, seasonal_naive are eligible
    for m in ["random_forest", "xgboost", "hybrid", "lstm", "cnn_lstm", "naive", "seasonal_naive"]:
        assert models_app[m]["eligible"] is True, f"Retail model '{m}' should be eligible: {models_app[m]}"

    # Verify manual execution of an ineligible model returns HTTP 400 with structured data
    resp = client.post("/api/v1/forecasting/train", json={"domain": "retail", "model": "arima", "prediction_limit": 5})
    assert resp.status_code == 400
    err_json = resp.get_json()
    assert "INTERLEAVED_PANEL" in err_json["message"]

    resp = client.post("/api/v1/forecasting/train", json={"domain": "retail", "model": "trend_xgboost", "prediction_limit": 5})
    assert resp.status_code == 400
    err_json = resp.get_json()
    assert "NO_HOURLY_LAG_STRUCTURE" in err_json["message"]

    # Verify manual execution of eligible models returns HTTP 200
    resp = client.post("/api/v1/forecasting/train", json={"domain": "retail", "model": "random_forest", "n_estimators": 5, "prediction_limit": 5})
    assert resp.status_code == 200
    assert resp.get_json()["data"]["model"] == "random_forest"

    resp = client.post("/api/v1/forecasting/train", json={"domain": "retail", "model": "naive", "prediction_limit": 5})
    assert resp.status_code == 200
    assert resp.get_json()["data"]["model"] == "naive"


def test_auto_select_strictly_validation_driven(client):
    """Auto Select compares candidate models using validation metrics only."""
    resp = client.post(
        "/api/v1/forecasting/compare",
        json={
            "domain": "electricity",
            "models": ["random_forest", "xgboost"],
            "selection_metric": "rmse",
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["selection_split"] == "validation"
    assert data["selection_metric"] == "rmse"
    assert "recommended_model" in data
    assert data["recommended_model"] in ("random_forest", "xgboost")
    for item in data["ranking"]:
        assert "validation_metrics" in item
        assert "rmse" in item["validation_metrics"]
