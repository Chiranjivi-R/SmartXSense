"""Focused tests for client-facing error sanitization (no path/traceback leakage)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402
from app.services.client_error_sanitize import sanitize_client_error_message  # noqa: E402
from app.services.ml_service import MLError  # noqa: E402


class TestConfig:
    TESTING = True
    DEBUG = False
    SECRET_KEY = "test-secret"
    JWT_SECRET_KEY = "test-jwt-secret-key-that-is-at-least-32-bytes"
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=1)
    SQLALCHEMY_DATABASE_URI = "sqlite://"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CORS_ORIGINS = ["http://localhost:5173"]


def test_sanitize_redacts_windows_absolute_paths():
    raw = r"Forecasting pipeline error: cannot open C:\Users\Chiranjivi R\Desktop\secret\train.parquet"
    cleaned = sanitize_client_error_message(raw)
    assert "Chiranjivi" not in cleaned
    assert "Desktop" not in cleaned
    assert "<path>" in cleaned
    assert "Forecasting pipeline error" in cleaned


def test_sanitize_redacts_unix_absolute_paths():
    raw = "Features unavailable under /home/ubuntu/project/ml_engine/data/features/x"
    cleaned = sanitize_client_error_message(raw)
    assert "/home/ubuntu" not in cleaned
    assert "<path>" in cleaned


def test_sanitize_blocks_traceback_and_pickle_leaks():
    assert sanitize_client_error_message('Traceback (most recent call last):\nFile "/tmp/x.py", line 1') == (
        "Request could not be processed"
    )
    assert sanitize_client_error_message("joblib.load failed: UnpicklingError") == (
        "Model artifact could not be processed"
    )


def test_sanitize_keeps_useful_validation_messages():
    msg = "Parameter 'prediction_limit' must be between 1 and 1000"
    assert sanitize_client_error_message(msg) == msg
    assert sanitize_client_error_message("Path traversal detected in features_dir") == (
        "Path traversal detected in features_dir"
    )


def test_mlerror_applies_sanitization():
    err = MLError(r"Forecasting pipeline error: missing C:\Users\secret\data.parquet", code=400)
    assert err.code == 400
    assert "Users" not in err.message
    assert "secret" not in err.message
    assert "<path>" in err.message


def test_forecast_api_400_does_not_echo_absolute_path(tmp_path):
    class Cfg(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts")
        FORECAST_HISTORY_ROOT = str(tmp_path / "history")

    app = create_app(Cfg)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        # Absolute path outside project should fail validation without echoing the full path body carelessly
        abs_path = str(tmp_path / "outside" / "features")
        response = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "retail",
                "features_dir": abs_path,
                "model": "random_forest",
            },
        )
        assert response.status_code == 400
        body = response.get_json()
        assert body["status"] == "error"
        assert body["code"] == 400
        message = body["message"]
        # Must not leak the absolute temp path string
        assert abs_path not in message
        assert "Traceback" not in message
        db.session.remove()
        db.drop_all()


def test_run_py_disables_werkzeug_debugger():
    run_py = (ROOT / "backend" / "run.py").read_text(encoding="utf-8")
    assert "use_debugger=False" in run_py
    assert "debug=False" in run_py
    assert "app.run(" in run_py
