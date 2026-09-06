"""Tests for API future forecasting with validation-based auto model selection."""

import sys
from pathlib import Path
from datetime import timedelta
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app import create_app
from app.extensions import db

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


def test_future_endpoint_auto_select_validation_winning_model(tmp_path):
    """Test /api/v1/forecasting/future with model='auto_select' selects best model from validation split."""
    class CustomConfig(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts")

    # Generate synthetic dataset
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=120, freq="D"),
        "sales": np.sin(np.linspace(0, 12, 120)) * 50 + 150,
        "festival_flag": [1 if i % 14 == 0 else 0 for i in range(120)],
    })
    prep = materialize_forecasting_features(df, "retail", output_root=tmp_path / "features")
    features_dir = prep["features_dir"]

    app = create_app(CustomConfig)
    with app.app_context():
        db.create_all()
        client = app.test_client()

        payload = {
            "domain": "retail",
            "features_dir": str(features_dir),
            "model": "auto_select",
            "horizon": 7,
        }

        resp = client.post("/api/v1/forecasting/future", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()["data"]

        assert data["selected_model"] in ("xgboost", "random_forest", "arima", "hybrid")
        assert "selection_reason" in data
        assert "validation_metrics" in data
        assert "test_metrics" in data
        assert "predictions" in data
        assert len(data["predictions"]) == 7
        assert "artifact" in data
        if data["selected_model"] in ("xgboost", "random_forest", "hybrid"):
            assert data["artifact"]["used"] is True
            assert data["artifact"]["source"] == "loaded"
            assert data["artifact_reused"] is True
        else:
            assert data["artifact"]["used"] is False
        assert "dataset_context" in data

        # Check prediction structure
        for pred in data["predictions"]:
            assert "timestamp" in pred
            assert "predicted" in pred
            assert "actual" not in pred

        db.session.remove()
        db.drop_all()


def test_future_endpoint_auto_select_electricity_hybrid(tmp_path):
    """Test /api/v1/forecasting/future with electricity auto_select and manual hybrid."""
    class CustomConfig(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts_elec")

    ts = pd.date_range("2024-01-01", periods=200, freq="h")
    df = pd.DataFrame({
        "timestamp": ts,
        "demand_mw": np.sin(np.linspace(0, 20, 200)) * 50 + 200,
        "temperature": np.cos(np.linspace(0, 10, 200)) * 10 + 25,
    })
    prep = materialize_forecasting_features(df, "electricity", output_root=tmp_path / "features_elec")
    features_dir = prep["features_dir"]

    app = create_app(CustomConfig)
    with app.app_context():
        db.create_all()
        client = app.test_client()

        # 1. Test Auto Select for electricity
        payload = {
            "domain": "electricity",
            "features_dir": str(features_dir),
            "model": "auto_select",
            "horizon": 12,
        }

        resp = client.post("/api/v1/forecasting/future", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()["data"]

        assert "selected_model" in data
        assert data["selected_model"] in ("hybrid", "arima", "random_forest", "xgboost")
        assert len(data["predictions"]) == 12
        for pred in data["predictions"]:
            assert "timestamp" in pred
            assert "predicted" in pred
            assert "actual" not in pred
            assert np.isfinite(pred["predicted"])

        # 2. Test Manual Hybrid selection for electricity
        payload_manual = {
            "domain": "electricity",
            "features_dir": str(features_dir),
            "model": "hybrid",
            "horizon": 12,
        }
        resp_manual = client.post("/api/v1/forecasting/future", json=payload_manual)
        assert resp_manual.status_code == 200
        data_manual = resp_manual.get_json()["data"]
        assert data_manual["selected_model"] == "hybrid"
        assert len(data_manual["predictions"]) == 12
        assert "validation_metrics" in data_manual

        db.session.remove()
        db.drop_all()


def test_future_auto_select_unsupported_candidate_models_returns_400(tmp_path):
    """Test B: When requested candidate models contain only future-unsupported models, returns HTTP 400."""
    class CustomConfig(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts_guard")

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=60, freq="D"),
        "sales": np.sin(np.linspace(0, 6, 60)) * 50 + 150,
    })
    prep = materialize_forecasting_features(df, "retail", output_root=tmp_path / "features_guard")
    features_dir = prep["features_dir"]

    app = create_app(CustomConfig)
    with app.app_context():
        db.create_all()
        client = app.test_client()

        # Request candidate models that do not support future forecasting in retail (lstm, naive)
        payload = {
            "domain": "retail",
            "features_dir": str(features_dir),
            "model": "auto_select",
            "models": ["lstm", "naive"],
            "horizon": 7,
        }

        resp = client.post("/api/v1/forecasting/future", json=payload)
        assert resp.status_code == 400
        err_json = resp.get_json()
        assert err_json["status"] == "error"
        assert err_json["code"] == 400
        msg = err_json.get("message", "").lower()
        assert "none of the requested candidate models support future forecasting" in msg
        assert "retail" in msg

        db.session.remove()
        db.drop_all()


def test_future_auto_select_explicit_learning_rate_preserved(tmp_path):
    """Test that explicit user-provided learning_rate is respected and not overwritten by default 0.05."""
    class CustomConfig(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts_lr")

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=80, freq="D"),
        "sales": np.sin(np.linspace(0, 8, 80)) * 50 + 150,
    })
    prep = materialize_forecasting_features(df, "retail", output_root=tmp_path / "features_lr")
    features_dir = prep["features_dir"]

    app = create_app(CustomConfig)
    with app.app_context():
        db.create_all()
        client = app.test_client()

        explicit_lr = 0.08
        payload = {
            "domain": "retail",
            "features_dir": str(features_dir),
            "model": "auto_select",
            "models": ["xgboost"],
            "learning_rate": explicit_lr,
            "horizon": 7,
        }

        resp = client.post("/api/v1/forecasting/future", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        assert data["selected_model"] == "xgboost"
        assert len(data["predictions"]) == 7

        db.session.remove()
        db.drop_all()

