"""Integration tests for dashboard, reports, models, and settings APIs (Phase F)."""

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
def mock_features_dir(tmp_path):
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


def _app(tmp_path):
    class Cfg(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts")
        FORECAST_HISTORY_ROOT = str(tmp_path / "history")

    return create_app(Cfg)


def test_dashboard_summary_empty_history(tmp_path):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        response = client.get("/api/v1/dashboard")
        assert response.status_code == 200
        data = response.get_json()["data"]
        assert data["total_forecast_runs"] == 0
        assert data["recent_runs"] == []
        assert data["latest_forecast"] is None
        assert "retail" in data["available_domains"]
        assert "random_forest" in data["available_models"]
        db.session.remove()
        db.drop_all()


def test_dashboard_summary_after_forecast(tmp_path, mock_features_dir):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        forecast = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
                "n_estimators": 8,
            },
        )
        assert forecast.status_code == 200

        future = client.post(
            "/api/v1/forecasting/future",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
                "n_estimators": 8,
                "horizon": 12,
            },
        )
        assert future.status_code == 200

        response = client.get("/api/v1/dashboard?recent_limit=5")
        assert response.status_code == 200
        data = response.get_json()["data"]
        assert data["total_forecast_runs"] >= 2
        assert data["future_forecast_runs"] >= 1
        assert len(data["recent_runs"]) >= 1
        assert data["latest_model"] is not None
        assert data["latest_domain"] == "electricity"
        db.session.remove()
        db.drop_all()


def test_reports_list_and_get(tmp_path, mock_features_dir):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        forecast = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
                "n_estimators": 8,
            },
        )
        assert forecast.status_code == 200
        run_id = forecast.get_json()["data"]["history"]["run_id"]

        listed = client.get("/api/v1/reports?limit=10")
        assert listed.status_code == 200
        reports = listed.get_json()["data"]["reports"]
        assert any(item["run_id"] == run_id for item in reports)
        assert all("predictions" not in item for item in reports)

        detail = client.get(f"/api/v1/reports/{run_id}")
        assert detail.status_code == 200
        body = detail.get_json()["data"]
        assert body["run_id"] == run_id
        assert "predictions" not in body

        missing = client.get("/api/v1/reports/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        assert missing.status_code == 404
        db.session.remove()
        db.drop_all()


def test_models_catalog(tmp_path):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        response = client.get("/api/v1/dashboard/models")
        assert response.status_code == 200
        data = response.get_json()["data"]
        keys = {item["key"] for item in data["models"]}
        for expected in (
            "naive",
            "seasonal_naive",
            "arima",
            "random_forest",
            "xgboost",
            "lstm",
            "cnn_lstm",
            "hybrid",
        ):
            assert expected in keys
        rf = next(item for item in data["models"] if item["key"] == "random_forest")
        assert rf["persistence_supported"] is True
        assert rf["explainability_supported"] is True
        assert rf["future_forecast_supported"] is True
        assert "accuracy" not in rf
        hybrid = next(item for item in data["models"] if item["key"] == "hybrid")
        assert hybrid["persistence_supported"] is False
        db.session.remove()
        db.drop_all()


def test_settings_system_info(tmp_path):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        response = client.get("/api/v1/settings")
        assert response.status_code == 200
        data = response.get_json()["data"]
        assert data["backend_status"] == "ok"
        assert "retail" in data["supported_domains"]
        assert data["apis"]["forecasting"] is True
        assert data["apis"]["history"] is True
        db.session.remove()
        db.drop_all()
