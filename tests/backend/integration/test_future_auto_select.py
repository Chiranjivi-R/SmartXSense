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

        assert "selected_model" in data
        assert data["selected_model"] in ("xgboost", "random_forest", "naive")
        assert "selection_reason" in data
        assert "validation_metrics" in data
        assert "test_metrics" in data
        assert "predictions" in data
        assert len(data["predictions"]) == 7
        assert "artifact" in data
        assert "dataset_context" in data

        # Check prediction structure
        for pred in data["predictions"]:
            assert "timestamp" in pred
            assert "predicted" in pred
            assert "actual" not in pred

        db.session.remove()
        db.drop_all()
