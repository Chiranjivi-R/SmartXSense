"""Integration tests verifying geographic isolation and dynamic applicability for newly uploaded datasets.

Verifies:
1. Dynamic location discovery: newly uploaded datasets discover locations strictly from their own data.
2. No location, artifact, or metadata leakage between demo datasets and uploaded datasets.
3. Insufficient history on uploaded dataset (e.g. France with 1 timestamp) yields structured N/A.
4. Auto Select on uploaded dataset without eligible models returns NO_APPLICABLE_MODEL (never fallback).
5. All Geographies on uploaded dataset uses the full uploaded dataset.
6. Fingerprint isolation: SHA-256 fingerprint includes dataset_id so caches cannot collide.
7. Switching between datasets completely isolates locations, features, models, and forecasts.
"""

from datetime import timedelta
from io import BytesIO
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app import create_app
from app.extensions import db
from app.services.ml_service import compare_service, forecast_service, MLError
from app.services.dataset_service import get_dataset_geography_service
from ml_engine.preprocessing.dynamic_prep import (
    compute_dynamic_fingerprint,
    prepare_dynamic_dataset,
    filter_df_by_geography,
    DynamicPrepError,
)
from ml_engine.models.applicability import assess_model_applicability


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
def test_app_and_client(tmp_path):
    class CustomConfig(TestConfig):
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts")
        UPLOAD_FOLDER = str(tmp_path / "uploads")
        PROCESSED_DATA_FOLDER = str(tmp_path / "processed")
        FEATURES_DATA_FOLDER = str(tmp_path / "features")

    app = create_app(CustomConfig)
    with app.app_context():
        db.create_all()
        client = app.test_client()
        yield app, client
        db.session.remove()
        db.drop_all()


def _upload_dataset(client, csv_bytes, filename, domain="retail"):
    """Helper to register user and upload dataset via API."""
    client.post(
        "/api/v1/auth/register",
        json={
            "email": f"test_{filename}@example.com",
            "password": "SecurePassword123!",
            "full_name": "Test Analyst",
            "role": "analyst",
        },
    )
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": f"test_{filename}@example.com", "password": "SecurePassword123!"},
    )
    token = login_resp.get_json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    upload_resp = client.post(
        "/api/v1/datasets/upload",
        headers=headers,
        data={"file": (BytesIO(csv_bytes), filename, "text/csv"), "domain": domain},
        content_type="multipart/form-data",
    )
    assert upload_resp.status_code == 201, f"Upload failed: {upload_resp.get_json()}"
    return upload_resp.get_json()["data"]["dataset_id"], headers


def test_fingerprint_isolation_by_dataset_id():
    """Verify compute_dynamic_fingerprint produces distinct digests for different dataset_ids."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"),
        "sales": np.arange(20, dtype=float),
        "city": ["Austin"] * 20,
    })

    fp1 = compute_dynamic_fingerprint(df, "sales", "timestamp", "Austin", dataset_id="dataset_alpha")
    fp2 = compute_dynamic_fingerprint(df, "sales", "timestamp", "Austin", dataset_id="dataset_beta")
    fp_none = compute_dynamic_fingerprint(df, "sales", "timestamp", "Austin", dataset_id=None)

    assert fp1 != fp2, "Fingerprints for distinct dataset_ids must not collide"
    assert fp1 != fp_none, "Fingerprint with dataset_id must differ from None"


def test_dynamic_location_discovery_strictly_from_uploaded_dataset(test_app_and_client):
    """Verify that uploading a dataset discovers only its own locations, not demo locations."""
    app, client = test_app_and_client

    # Dataset 1: Texas cities
    texas_csv = (
        "Date,City,Sales\n"
        + "\n".join([f"2024-01-{(i%28)+1:02d},Austin,{100+i}" for i in range(30)])
        + "\n"
        + "\n".join([f"2024-01-{(i%28)+1:02d},Dallas,{200+i}" for i in range(30)])
        + "\n"
    ).encode("utf-8")

    ds1_id, h1 = _upload_dataset(client, texas_csv, "texas_retail.csv", domain="retail")

    geo_resp1 = client.get(f"/api/v1/datasets/{ds1_id}/geography", headers=h1)
    assert geo_resp1.status_code == 200
    locations1 = set(geo_resp1.get_json()["data"]["available_locations"])
    assert "Austin" in locations1
    assert "Dallas" in locations1
    assert "Houston" not in locations1
    assert "Mysuru" not in locations1
    assert "Ahmedabad" not in locations1
    assert "California" not in locations1

    # Dataset 2: European countries
    euro_csv = (
        "Date,Country,Sales\n"
        + "\n".join([f"2024-02-{(i%28)+1:02d},France,{50+i}" for i in range(25)])
        + "\n"
        + "\n".join([f"2024-02-{(i%28)+1:02d},Germany,{80+i}" for i in range(25)])
        + "\n"
    ).encode("utf-8")

    ds2_id, h2 = _upload_dataset(client, euro_csv, "euro_retail.csv", domain="retail")

    geo_resp2 = client.get(f"/api/v1/datasets/{ds2_id}/geography", headers=h2)
    assert geo_resp2.status_code == 200
    locations2 = set(geo_resp2.get_json()["data"]["available_locations"])
    assert "France" in locations2
    assert "Germany" in locations2
    # Texas locations must NOT appear in euro dataset
    assert "Austin" not in locations2
    assert "Dallas" not in locations2


def test_uploaded_dataset_low_history_location_returns_structured_na(test_app_and_client):
    """Verify that a location with only 1 unique timestamp in an uploaded dataset returns structured N/A."""
    app, client = test_app_and_client

    csv_bytes = (
        "Date,City,Sales\n"
        + "\n".join([f"2024-01-{(i%28)+1:02d},London,{100+i}" for i in range(40)])
        + "\n"
        + "\n".join([f"2024-01-01,Paris,{10+i}" for i in range(20)])
        + "\n"
    ).encode("utf-8")

    ds_id, headers = _upload_dataset(client, csv_bytes, "mixed_history.csv", domain="retail")

    # Calling compare on Paris (1 unique date) must return structured N/A error
    resp = client.post(
        "/api/v1/forecasting/compare",
        headers=headers,
        json={
            "dataset_id": ds_id,
            "domain": "retail",
            "location": "Paris",
        },
    )

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["status"] == "error"
    assert "data" in body
    assert body["data"].get("status") == "N/A"
    assert body["data"].get("reason_code") == "INSUFFICIENT_HISTORY"
    assert "too small for time-series split" in body["message"] or "unique timestamps" in body["message"]


def test_uploaded_dataset_autoselect_insufficient_history_no_fallback(test_app_and_client):
    """Verify that Auto Select on an uploaded dataset with insufficient history returns N/A, never fallback to RF."""
    app, client = test_app_and_client

    # 10 days of data for city Bristol -> 70% train = 7 training obs (< 15 required for RF/XGBoost, < 30 for ARIMA)
    csv_bytes = (
        "Date,City,Sales\n"
        + "\n".join([f"2024-01-{i:02d},Bristol,{50+i}" for i in range(1, 11)])
        + "\n"
    ).encode("utf-8")

    ds_id, headers = _upload_dataset(client, csv_bytes, "short_history.csv", domain="retail")

    # compare endpoint must report NO_APPLICABLE_MODEL
    compare_resp = client.post(
        "/api/v1/forecasting/compare",
        headers=headers,
        json={
            "dataset_id": ds_id,
            "domain": "retail",
            "location": "Bristol",
        },
    )

    assert compare_resp.status_code == 400
    comp_body = compare_resp.get_json()
    assert comp_body.get("data", {}).get("status") == "N/A"
    assert comp_body.get("data", {}).get("reason_code") == "NO_APPLICABLE_MODEL"
    assert "No applicable forecasting model" in comp_body.get("message", "")

    # forecast endpoint with auto_select must NOT fall back to Random Forest
    forecast_resp = client.post(
        "/api/v1/forecasting/forecast",
        headers=headers,
        json={
            "dataset_id": ds_id,
            "domain": "retail",
            "location": "Bristol",
            "model": "auto_select",
        },
    )

    assert forecast_resp.status_code == 400
    fc_body = forecast_resp.get_json()
    assert fc_body.get("data", {}).get("status") == "N/A"
    assert fc_body.get("data", {}).get("reason_code") == "NO_APPLICABLE_MODEL"


def test_uploaded_dataset_all_geographies_uses_full_dataset(test_app_and_client):
    """Verify that all geography uses the full uploaded dataset."""
    app, client = test_app_and_client

    rows = ["Date,City,Sales"]
    for day in range(1, 61):
        city = "CityA" if day <= 20 else ("CityB" if day <= 40 else "CityC")
        month = 1 if day <= 28 else 2
        d = (day % 28) + 1
        rows.append(f"2024-{month:02d}-{d:02d},{city},{100 + day}")

    csv_bytes = "\n".join(rows).encode("utf-8")
    ds_id, headers = _upload_dataset(client, csv_bytes, "multi_city.csv", domain="retail")

    resp = client.post(
        "/api/v1/forecasting/compare",
        headers=headers,
        json={
            "dataset_id": ds_id,
            "domain": "retail",
            "location": "all",
        },
    )

    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert "recommended_model" in data
    assert isinstance(data["ranking"], list)
    assert len(data["ranking"]) > 0
    assert data["dataset_context"]["location"] in ("All Geographies", "all", None)


def test_dataset_switching_isolates_features_and_recommendations(test_app_and_client):
    """Verify switching between two uploaded datasets completely isolates their feature caches and forecasts."""
    app, client = test_app_and_client

    header_alpha = "Date,store_id,City,Sales\n" + "\n".join([f"2024-{1 if i<=28 else 2:02d}-{(i%28)+1:02d},S01,AlphaCity,{100 + i}" for i in range(60)])
    ds_alpha_id, h_alpha = _upload_dataset(client, header_alpha.encode("utf-8"), "ds_alpha.csv")

    header_beta = "Date,store_id,City,Sales\n" + "\n".join([f"2024-{1 if i<=28 else 2:02d}-{(i%28)+1:02d},S02,BetaCity,{5000 + i*10}" for i in range(60)])
    ds_beta_id, h_beta = _upload_dataset(client, header_beta.encode("utf-8"), "ds_beta.csv")

    resp_alpha = client.post(
        "/api/v1/forecasting/forecast",
        headers=h_alpha,
        json={
            "dataset_id": ds_alpha_id,
            "domain": "retail",
            "location": "AlphaCity",
            "model": "auto_select",
        },
    )
    assert resp_alpha.status_code == 200
    res_alpha = resp_alpha.get_json()["data"]
    assert res_alpha["dataset_context"]["location"] == "AlphaCity"
    assert res_alpha["model"] is not None
    preds_alpha = [p["predicted"] for p in res_alpha["predictions"]]
    assert np.mean(preds_alpha) < 1000

    resp_beta = client.post(
        "/api/v1/forecasting/forecast",
        headers=h_beta,
        json={
            "dataset_id": ds_beta_id,
            "domain": "retail",
            "location": "BetaCity",
            "model": "auto_select",
        },
    )
    assert resp_beta.status_code == 200
    res_beta = resp_beta.get_json()["data"]
    assert res_beta["dataset_context"]["location"] == "BetaCity"
    assert res_beta["model"] is not None
    preds_beta = [p["predicted"] for p in res_beta["predictions"]]
    assert np.mean(preds_beta) > 3000
