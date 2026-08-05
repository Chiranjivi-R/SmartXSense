"""Integration tests for backend forecasting, model comparison, and explainability endpoints."""

from datetime import timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402


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
def client():
    """Flask test client fixture — yields test_client only (no tuple)."""
    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def mock_features_dir(tmp_path):
    """Create minimal feature split artifacts in tmp_path for test use."""
    features_dir = tmp_path / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n = 60
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "demand_mw": np.sin(np.linspace(0, 10, n)) * 100 + 500,
        "sales": np.sin(np.linspace(0, 10, n)) * 10 + 50,
        "feature_lag1": np.roll(np.sin(np.linspace(0, 10, n)), 1),
    })

    df.iloc[:35].to_parquet(features_dir / "train.parquet", index=False)
    df.iloc[35:48].to_parquet(features_dir / "validation.parquet", index=False)
    df.iloc[48:].to_parquet(features_dir / "test.parquet", index=False)

    return features_dir


# ---------------------------------------------------------------------------
# Forecasting endpoint tests
# ---------------------------------------------------------------------------

def test_forecast_endpoint_single_model(client, mock_features_dir):
    """POST /train returns 200 with validation and test metrics for RF."""
    response = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "random_forest",
            "n_estimators": 10,
        },
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["data"]["model"] == "random_forest"
    assert "validation_metrics" in json_data["data"]
    assert "test_metrics" in json_data["data"]


def test_forecast_endpoint_hybrid(client, mock_features_dir):
    """POST /train with hybrid returns 200 with weights and ensemble metrics."""
    response = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "hybrid",
            "components": ["arima", "random_forest"],
            "n_estimators": 10,
        },
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    assert json_data["data"]["model"] == "hybrid"
    assert "weights" in json_data["data"]
    assert "validation_metrics" in json_data["data"]


def test_compare_endpoint(client, mock_features_dir):
    """POST /compare returns ranked models using validation metrics only."""
    # Use a lightweight explicit candidate list for fast test execution
    response = client.post(
        "/api/v1/forecasting/compare",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "models": ["random_forest", "xgboost"],
            "selection_metric": "rmse",
        },
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    data = json_data["data"]
    assert data["selection_metric"] == "rmse"
    assert data["selection_split"] == "validation"
    assert "recommended_model" in data
    assert isinstance(data["ranking"], list)
    assert len(data["ranking"]) >= 2


def test_compare_selection_split_is_validation(client, mock_features_dir):
    """Confirm that comparison output always declares selection_split = 'validation'."""
    response = client.post(
        "/api/v1/forecasting/compare",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "models": ["random_forest", "xgboost"],
            "selection_metric": "rmse",
        },
    )
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["selection_split"] == "validation"
    # Recommended model must appear in ranking
    recommended = data["recommended_model"]
    ranked_models = [r["model"] for r in data["ranking"]]
    assert recommended in ranked_models


# ---------------------------------------------------------------------------
# Path security tests
# ---------------------------------------------------------------------------

def test_path_security_traversal_rejection(client):
    """Paths with '..' must be rejected with 400."""
    response = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "electricity",
            "features_dir": "../../../secret_dir",
            "model": "random_forest",
        },
    )

    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["status"] == "error"
    assert "traversal" in json_data["message"].lower()


def test_path_security_nonexistent_dir(client):
    """Paths pointing to non-existent directories must return 400."""
    response = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "electricity",
            "features_dir": "C:/nonexistent_test_path_abc123",
            "model": "random_forest",
        },
    )
    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["status"] == "error"


# ---------------------------------------------------------------------------
# Invalid input tests
# ---------------------------------------------------------------------------

def test_invalid_domain_or_model(client, mock_features_dir):
    """Invalid domain or unsupported model/domain combo must return 400."""
    res_domain = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "invalid_domain",
            "features_dir": str(mock_features_dir),
            "model": "random_forest",
        },
    )
    assert res_domain.status_code == 400

    res_arima = client.post(
        "/api/v1/forecasting/train",
        json={
            "domain": "retail",
            "features_dir": str(mock_features_dir),
            "model": "arima",
        },
    )
    assert res_arima.status_code == 400


def test_invalid_json_payload(client):
    """Missing or malformed JSON body must return 400."""
    response = client.post(
        "/api/v1/forecasting/train",
        data="not-json",
        content_type="text/plain",
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# Explainability endpoint tests
# ---------------------------------------------------------------------------

def test_explain_endpoint_global_shap(client, mock_features_dir):
    """POST /explain with global type returns feature importances dict."""
    response = client.post(
        "/api/v1/explainability/explain",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "random_forest",
            "explanation_type": "global",
            "method": "shap",
            "n_estimators": 10,
        },
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    data = json_data["data"]
    assert data["explanation_type"] == "global"
    assert data["method"] == "shap"
    assert isinstance(data["feature_importance"], dict)
    assert len(data["feature_importance"]) > 0


def test_explain_endpoint_local_shap(client, mock_features_dir):
    """POST /explain with local type returns per-sample SHAP attribution."""
    response = client.post(
        "/api/v1/explainability/explain",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "xgboost",
            "explanation_type": "local",
            "method": "shap",
            "sample_index": 0,
            "n_estimators": 10,
        },
    )

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["status"] == "success"
    data = json_data["data"]
    assert data["explanation_type"] == "local"
    assert isinstance(data["prediction"], float)
    assert "shap_contributions" in data
    assert "top_positive_contributors" in data
    assert "top_negative_contributors" in data


def test_explain_endpoint_unsupported_model(client, mock_features_dir):
    """Unsupported explainability models must return 400."""
    response = client.post(
        "/api/v1/explainability/explain",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "arima",
            "explanation_type": "global",
        },
    )

    assert response.status_code == 400
    json_data = response.get_json()
    assert json_data["status"] == "error"
    # Error message must indicate explainability limitation
    assert "explainability" in json_data["message"].lower() or "supported" in json_data["message"].lower()


def test_explain_endpoint_lstm_unsupported(client, mock_features_dir):
    """LSTM model must be rejected by explainability endpoint with 400."""
    response = client.post(
        "/api/v1/explainability/explain",
        json={
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "lstm",
            "explanation_type": "global",
        },
    )
    assert response.status_code == 400
