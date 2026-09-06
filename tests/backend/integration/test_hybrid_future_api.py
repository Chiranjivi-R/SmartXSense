"""End-to-end integration tests for True Hybrid Future Forecasting via Flask API."""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "backend") not in sys.path:
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


@pytest.fixture
def test_client(tmp_path: Path):
    class CustomConfig(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts_hybrid_api")

    app = create_app(CustomConfig)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        yield client
        db.session.remove()
        db.drop_all()


def test_retail_hybrid_future_forecast_7d_14d_30d(tmp_path: Path, test_client):
    """Test A & H: Manual retail hybrid future forecast succeeds for 7d, 14d, 30d."""
    ts = pd.date_range("2024-01-01", periods=120, freq="D")
    df = pd.DataFrame({
        "timestamp": ts,
        "sales": np.sin(np.linspace(0, 10, 120)) * 50 + 200,
        "on_promotion": np.random.choice([0, 1], size=120),
    })
    prep = materialize_forecasting_features(df, "retail", output_root=tmp_path / "features_retail")
    features_dir = prep["features_dir"]

    for h in (7, 14, 30):
        payload = {
            "domain": "retail",
            "features_dir": str(features_dir),
            "model": "hybrid",
            "horizon": h,
        }
        resp = test_client.post("/api/v1/forecasting/future", json=payload)
        assert resp.status_code == 200, f"Failed at horizon {h}: {resp.get_json()}"
        data = resp.get_json()["data"]

        assert data["selected_model"] == "hybrid"
        assert data["domain"] == "retail"
        assert data["horizon"] == h
        assert len(data["predictions"]) == h
        assert "validation_metrics" in data

        # Verify future timestamps are strictly after historical timestamps
        history_last = pd.Timestamp(ts[-1]).tz_localize(None)
        pred_timestamps = [pd.Timestamp(p["timestamp"]).tz_localize(None) for p in data["predictions"]]
        assert pred_timestamps[0] > history_last
        assert pred_timestamps == sorted(pred_timestamps)

        # Verify no actuals fabricated
        for pred in data["predictions"]:
            assert "timestamp" in pred
            assert "predicted" in pred
            assert "actual" not in pred
            assert np.isfinite(pred["predicted"])


def test_electricity_hybrid_future_forecast_horizon_filtering(tmp_path: Path, test_client):
    """Test B: Horizon-aware component filtering at h=24 and h=720 on electricity."""
    ts = pd.date_range("2024-01-01", periods=300, freq="h")
    df = pd.DataFrame({
        "timestamp": ts,
        "demand_mw": np.sin(np.linspace(0, 25, 300)) * 50 + 300,
        "temperature": np.cos(np.linspace(0, 12, 300)) * 10 + 20,
    })
    prep = materialize_forecasting_features(df, "electricity", output_root=tmp_path / "features_elec")
    features_dir = prep["features_dir"]

    # At h=24: Hybrid with components ["random_forest", "xgboost", "sarima"]
    payload_24 = {
        "domain": "electricity",
        "features_dir": str(features_dir),
        "model": "hybrid",
        "components": ["random_forest", "xgboost", "sarima"],
        "horizon": 24,
    }
    resp_24 = test_client.post("/api/v1/forecasting/future", json=payload_24)
    assert resp_24.status_code == 200, resp_24.get_json()
    data_24 = resp_24.get_json()["data"]
    assert data_24["selected_model"] == "hybrid"
    assert len(data_24["predictions"]) == 24

    # At h=720: SARIMA is excluded because h=720 > 168, but RF + XGB continue
    payload_720 = {
        "domain": "electricity",
        "features_dir": str(features_dir),
        "model": "hybrid",
        "components": ["random_forest", "xgboost", "sarima"],
        "horizon": 720,
    }
    resp_720 = test_client.post("/api/v1/forecasting/future", json=payload_720)
    assert resp_720.status_code == 200, resp_720.get_json()
    data_720 = resp_720.get_json()["data"]
    assert data_720["selected_model"] == "hybrid"
    assert len(data_720["predictions"]) == 720


def test_hybrid_future_forecast_fails_when_fewer_than_two_compatible(tmp_path: Path, test_client):
    """Test A5: Structured failure when fewer than two components remain compatible."""
    ts = pd.date_range("2024-01-01", periods=150, freq="h")
    df = pd.DataFrame({
        "timestamp": ts,
        "demand_mw": np.sin(np.linspace(0, 15, 150)) * 50 + 300,
        "temperature": np.cos(np.linspace(0, 8, 150)) * 10 + 20,
    })
    prep = materialize_forecasting_features(df, "electricity", output_root=tmp_path / "features_few")
    features_dir = prep["features_dir"]

    # Pass 1 compatible (RF) and 1 capped at h=168 (SARIMA), request h=720 -> only 1 remains
    payload = {
        "domain": "electricity",
        "features_dir": str(features_dir),
        "model": "hybrid",
        "components": ["random_forest", "sarima"],
        "horizon": 720,
    }
    resp = test_client.post("/api/v1/forecasting/future", json=payload)
    assert resp.status_code == 400
    err = resp.get_json()
    assert err["status"] == "error"
    assert "Hybrid requires at least two future-compatible component models" in err["message"]


def test_hybrid_future_artifact_reuse_on_repeated_requests(tmp_path: Path, test_client):
    """Test I: Repeated identical hybrid future requests reuse component artifacts."""
    ts = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "timestamp": ts,
        "sales": np.sin(np.linspace(0, 8, 100)) * 40 + 150,
    })
    prep = materialize_forecasting_features(df, "retail", output_root=tmp_path / "features_reuse")
    features_dir = prep["features_dir"]

    payload = {
        "domain": "retail",
        "features_dir": str(features_dir),
        "model": "hybrid",
        "horizon": 7,
    }

    # First request: trains components and persists them
    resp1 = test_client.post("/api/v1/forecasting/future", json=payload)
    assert resp1.status_code == 200
    data1 = resp1.get_json()["data"]

    # Second identical request: reuses component artifacts from disk
    resp2 = test_client.post("/api/v1/forecasting/future", json=payload)
    assert resp2.status_code == 200
    data2 = resp2.get_json()["data"]

    assert data2["artifact"]["used"] is True
    assert data2["artifact"]["source"] == "loaded"
    assert data2["artifact_reused"] is True
    # Predictions match
    preds1 = [p["predicted"] for p in data1["predictions"]]
    preds2 = [p["predicted"] for p in data2["predictions"]]
    np.testing.assert_allclose(preds1, preds2, rtol=1e-5)


def test_auto_select_retains_validation_best_behavior(tmp_path: Path, test_client):
    """Test E: Auto select chooses validation-best without hardcoded bias."""
    ts = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "timestamp": ts,
        "sales": np.sin(np.linspace(0, 8, 100)) * 40 + 150,
    })
    prep = materialize_forecasting_features(df, "retail", output_root=tmp_path / "features_auto")
    features_dir = prep["features_dir"]

    payload = {
        "domain": "retail",
        "features_dir": str(features_dir),
        "model": "auto_select",
        "horizon": 7,
    }
    resp = test_client.post("/api/v1/forecasting/future", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["selected_model"] in ("random_forest", "xgboost", "hybrid")
    assert "selection_reason" in data
