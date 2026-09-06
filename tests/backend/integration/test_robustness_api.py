"""Phase G robustness tests: health, validation, path safety, history resilience."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402


class BaseConfig:
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
    np.random.seed(7)
    n = 60
    dates = pd.date_range("2026-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "timestamp": dates,
        "demand_mw": np.sin(np.linspace(0, 8, n)) * 80 + 400,
        "sales": np.sin(np.linspace(0, 8, n)) * 8 + 40,
        "feature_lag1": np.roll(np.sin(np.linspace(0, 8, n)), 1),
    })
    df.iloc[:35].to_parquet(features_dir / "train.parquet", index=False)
    df.iloc[35:48].to_parquet(features_dir / "validation.parquet", index=False)
    df.iloc[48:].to_parquet(features_dir / "test.parquet", index=False)
    return features_dir


def _app(tmp_path):
    class Cfg(BaseConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts")
        FORECAST_HISTORY_ROOT = str(tmp_path / "history")

    return create_app(Cfg)


def test_health_endpoint_ok(tmp_path):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "success"
        data = body["data"]
        assert data["service"] == "SmartXSense"
        assert data["backend"] == "ok"
        assert data["ml_engine"] == "available"
        assert data["history_store"] == "available"
        assert "database" in data
        assert "overall" in data
        # Never expose absolute paths
        blob = json.dumps(data)
        assert ":\\\\" not in blob
        assert "/Users/" not in blob
        db.session.remove()
        db.drop_all()


def test_forecast_rejects_malformed_json_and_unknown_model(tmp_path, mock_features_dir):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()

        bad_json = client.post(
            "/api/v1/forecasting/forecast",
            data="{not-json",
            content_type="application/json",
        )
        assert bad_json.status_code == 400

        unknown = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "totally_fake_model",
            },
        )
        assert unknown.status_code == 400
        assert "Unsupported model" in unknown.get_json()["message"]
        db.session.remove()
        db.drop_all()


def test_path_traversal_rejected_without_leaking_path(tmp_path):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        response = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "electricity",
                "features_dir": "../../../etc/passwd",
                "model": "random_forest",
            },
        )
        assert response.status_code == 400
        message = response.get_json()["message"]
        assert "traversal" in message.lower() or "access denied" in message.lower() or "does not exist" in message.lower()
        assert "passwd" not in message
        db.session.remove()
        db.drop_all()


def test_compare_rejects_invalid_metric_and_too_many_models(tmp_path, mock_features_dir):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        bad_metric = client.post(
            "/api/v1/forecasting/compare",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "models": ["random_forest"],
                "selection_metric": "accuracy",
            },
        )
        assert bad_metric.status_code == 400

        too_many = client.post(
            "/api/v1/forecasting/compare",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "models": [
                    "naive",
                    "seasonal_naive",
                    "arima",
                    "random_forest",
                    "xgboost",
                    "lstm",
                    "hybrid",
                ],
            },
        )
        assert too_many.status_code == 400
        assert "at most" in too_many.get_json()["message"]
        db.session.remove()
        db.drop_all()


def test_history_keeps_valid_entries_when_index_has_junk(tmp_path, mock_features_dir):
    app = _app(tmp_path)
    history_root = Path(tmp_path / "history")
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

        index_path = history_root / "index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index.insert(0, {"run_id": "../bad", "domain": "x"})
        index.insert(1, {"run_id": "deadbeefdeadbeefdeadbeefdeadbeef", "predictions": [1, 2, 3]})
        index.append("not-an-object")
        index_path.write_text(json.dumps(index), encoding="utf-8")

        listed = client.get("/api/v1/forecasting/history?limit=20")
        assert listed.status_code == 200
        runs = listed.get_json()["data"]["runs"]
        assert any(item["run_id"] == run_id for item in runs)
        assert all("predictions" not in item for item in runs)
        assert all(item.get("run_id") != "../bad" for item in runs)
        db.session.remove()
        db.drop_all()


def test_artifact_corruption_falls_back_to_training(tmp_path, mock_features_dir):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        payload = {
            "domain": "electricity",
            "features_dir": str(mock_features_dir),
            "model": "random_forest",
            "n_estimators": 8,
            "prediction_limit": 10,
        }
        first = client.post("/api/v1/forecasting/forecast", json=payload)
        assert first.status_code == 200
        key = first.get_json()["data"]["artifact"]["key"]
        assert key

        # Corrupt the persisted pickle under the controlled artifacts root.
        artifact_dir = Path(tmp_path / "artifacts") / key
        model_file = artifact_dir / "model.pkl"
        assert model_file.is_file()
        model_file.write_bytes(b"not-a-pickle")

        second = client.post("/api/v1/forecasting/forecast", json=payload)
        assert second.status_code == 200
        data = second.get_json()["data"]
        # Safe fallback: train again rather than crash.
        assert data["artifact"]["source"] == "trained"
        assert "predictions" in data
        db.session.remove()
        db.drop_all()


def test_invalid_domain_and_prediction_limit(tmp_path, mock_features_dir):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()

        bad_domain = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "spaceships",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
            },
        )
        assert bad_domain.status_code == 400

        for bad_limit in (0, -5, True, 5000, "abc"):
            response = client.post(
                "/api/v1/forecasting/forecast",
                json={
                    "domain": "electricity",
                    "features_dir": str(mock_features_dir),
                    "model": "random_forest",
                    "prediction_limit": bad_limit,
                },
            )
            assert response.status_code == 400, f"limit={bad_limit}"
        db.session.remove()
        db.drop_all()


def test_future_rejects_invalid_horizon_and_unsupported_combo(tmp_path, mock_features_dir):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()

        for bad_horizon in (0, -1, True, 9999):
            response = client.post(
                "/api/v1/forecasting/future",
                json={
                    "domain": "electricity",
                    "features_dir": str(mock_features_dir),
                    "model": "random_forest",
                    "horizon": bad_horizon,
                },
            )
            assert response.status_code == 400, f"horizon={bad_horizon}"

        # Inapplicable model request on fixture (lacks hourly lag architecture) must return 400
        unsupported_trend = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "retail",
                "features_dir": str(mock_features_dir),
                "model": "trend_xgboost",
            },
        )
        assert unsupported_trend.status_code == 400
        trend_err = unsupported_trend.get_json()
        assert "not applicable" in trend_err.get("message", "").lower()

        # Exp34: clean single-series retail fixture dynamically allows ARIMA (HTTP 200)
        retail_arima = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "retail",
                "features_dir": str(mock_features_dir),
                "model": "arima",
            },
        )
        assert retail_arima.status_code == 200

        unsupported_future = client.post(
            "/api/v1/forecasting/future",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "lstm",
                "horizon": 24,
            },
        )
        assert unsupported_future.status_code == 400
        db.session.remove()
        db.drop_all()


def test_arbitrary_feature_path_and_history_404(tmp_path):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()

        # Absolute path outside allowed roots (when not under temp/project).
        # Use a clearly invalid non-existent absolute path.
        response = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "electricity",
                "features_dir": "C:/nonexistent_smartxsense_features_xyz",
                "model": "random_forest",
            },
        )
        assert response.status_code == 400
        message = response.get_json()["message"].lower()
        assert "does not exist" in message or "access denied" in message
        assert "nonexistent_smartxsense_features_xyz" not in response.get_json()["message"]

        missing = client.get("/api/v1/forecasting/history/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        assert missing.status_code == 404

        bad_id = client.get("/api/v1/forecasting/history/../etc/passwd")
        assert bad_id.status_code in (404, 400)
        db.session.remove()
        db.drop_all()


def test_corrupted_history_index_does_not_crash(tmp_path):
    app = _app(tmp_path)
    history_root = Path(tmp_path / "history")
    history_root.mkdir(parents=True, exist_ok=True)
    (history_root / "index.json").write_text("{not-json", encoding="utf-8")

    with app.app_context():
        db.create_all()
        client = app.test_client()
        response = client.get("/api/v1/forecasting/history")
        assert response.status_code == 200
        assert response.get_json()["data"]["runs"] == []

        dash = client.get("/api/v1/dashboard")
        assert dash.status_code == 200
        assert dash.get_json()["data"]["total_forecast_runs"] == 0
        db.session.remove()
        db.drop_all()


def test_explain_rejects_invalid_inputs(tmp_path, mock_features_dir):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()

        bad_model = client.post(
            "/api/v1/explainability/explain",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "invalid_model_xyz",
                "explanation_type": "global",
                "method": "shap",
            },
        )
        assert bad_model.status_code == 400

        bad_domain = client.post(
            "/api/v1/explainability/explain",
            json={
                "domain": "weather",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
            },
        )
        assert bad_domain.status_code == 400
        db.session.remove()
        db.drop_all()


def test_compare_rejects_invalid_selection_split(tmp_path, mock_features_dir):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        response = client.post(
            "/api/v1/forecasting/compare",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "models": ["random_forest"],
                "selection_split": "test",
            },
        )
        assert response.status_code == 400
        db.session.remove()
        db.drop_all()
