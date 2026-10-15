"""Integration tests for PDF generation and CSV/JSON report exports."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

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


def _app(tmp_path: Path):
    app = create_app(TestConfig)
    app.config["FORECAST_HISTORY_ROOT"] = str(tmp_path / "history")
    app.config["MODEL_ARTIFACTS_ROOT"] = str(tmp_path / "artifacts")
    return app


@pytest.fixture()
def mock_features_dir(tmp_path: Path) -> Path:
    fdir = tmp_path / "features"
    fdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 200
    ts = pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC")
    target = 100 + rng.normal(0, 5, n)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "demand_mw": target,
            "year": ts.year,
            "month": ts.month,
            "day": ts.day,
            "hour": ts.hour,
            "day_of_week": ts.dayofweek,
            "quarter": ts.quarter,
            "is_weekend": (ts.dayofweek >= 5).astype(int),
            "lag_1": pd.Series(target).shift(1),
            "lag_24": pd.Series(target).shift(24),
            "lag_168": pd.Series(target).shift(168),
            "rolling_mean_24": pd.Series(target).shift(1).rolling(24).mean(),
            "rolling_mean_168": pd.Series(target).shift(1).rolling(168).mean(),
        }
    ).dropna().reset_index(drop=True)

    n_clean = len(df)
    t_end = int(n_clean * 0.7)
    v_end = int(n_clean * 0.85)

    df.iloc[:t_end].to_parquet(fdir / "train.parquet", index=False)
    df.iloc[t_end:v_end].to_parquet(fdir / "validation.parquet", index=False)
    df.iloc[v_end:].to_parquet(fdir / "test.parquet", index=False)
    return fdir


def test_pdf_and_exports_for_test_evaluation_and_future(tmp_path, mock_features_dir):
    app = _app(tmp_path)
    with app.app_context():
        db.create_all()
        client = app.test_client()

        # 1. Execute a test evaluation forecast
        resp = client.post(
            "/api/v1/forecasting/forecast",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
                "n_estimators": 8,
            },
        )
        assert resp.status_code == 200
        run_id = resp.get_json()["data"]["history"]["run_id"]

        # 2. Get report detail
        detail = client.get(f"/api/v1/reports/{run_id}")
        assert detail.status_code == 200
        data = detail.get_json()["data"]
        assert data["run_id"] == run_id
        assert "dataset_info" in data
        assert "validation_benchmark" in data
        assert "forecast_data" in data
        assert "predictions" not in data  # Contract: legacy raw predictions key must not be present at root

        # 3. Download PDF report
        pdf_resp = client.get(f"/api/v1/reports/{run_id}/pdf")
        assert pdf_resp.status_code == 200
        assert pdf_resp.content_type == "application/pdf"
        assert pdf_resp.data.startswith(b"%PDF-")
        assert len(pdf_resp.data) > 1000

        # 4. Export CSV
        csv_resp = client.get(f"/api/v1/reports/{run_id}/export/csv")
        assert csv_resp.status_code == 200
        assert "text/csv" in csv_resp.content_type
        csv_lines = csv_resp.data.decode("utf-8").strip().splitlines()
        assert len(csv_lines) > 1
        assert "timestamp,predicted,actual,residual" in csv_lines[0]

        # 5. Export JSON
        json_resp = client.get(f"/api/v1/reports/{run_id}/export/json")
        assert json_resp.status_code == 200
        json_data = json.loads(json_resp.data.decode("utf-8"))
        assert json_data["run_id"] == run_id

        # 6. Execute a future forecast
        fut_resp = client.post(
            "/api/v1/forecasting/future",
            json={
                "domain": "electricity",
                "features_dir": str(mock_features_dir),
                "model": "random_forest",
                "n_estimators": 8,
                "horizon": 12,
            },
        )
        assert fut_resp.status_code == 200
        fut_run_id = fut_resp.get_json()["data"]["history"]["run_id"]

        # 7. Future PDF download
        fut_pdf = client.get(f"/api/v1/reports/{fut_run_id}/pdf")
        assert fut_pdf.status_code == 200
        assert fut_pdf.data.startswith(b"%PDF-")

        # 8. Future CSV export (Must NOT have actuals column!)
        fut_csv = client.get(f"/api/v1/reports/{fut_run_id}/export/csv")
        assert fut_csv.status_code == 200
        fut_csv_lines = fut_csv.data.decode("utf-8").strip().splitlines()
        assert "timestamp,predicted" == fut_csv_lines[0]
        assert "actual" not in fut_csv_lines[0]

        # 9. Nonexistent run 404
        missing_pdf = client.get("/api/v1/reports/nonexistent_12345678/pdf")
        assert missing_pdf.status_code == 404

        db.session.remove()
        db.drop_all()
