"""End-to-end integration tests for dataset-adaptive feature discovery and forecasting pipeline."""

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


@pytest.fixture()
def client(tmp_path):
    class TestConfig:
        TESTING = True
        SECRET_KEY = "test-secret"
        JWT_SECRET_KEY = "test-jwt-secret-key-that-is-at-least-32-bytes"
        JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
        JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=1)
        SQLALCHEMY_DATABASE_URI = "sqlite://"
        SQLALCHEMY_TRACK_MODIFICATIONS = False
        CORS_ORIGINS = ["http://localhost:5173"]
        UPLOAD_FOLDER = str(tmp_path / "uploads")
        PROCESSED_DATA_FOLDER = str(tmp_path / "processed")
        FEATURES_DATA_FOLDER = str(tmp_path / "features")
        ML_ARTIFACTS_ROOT = str(tmp_path / "artifacts")
        MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()
        yield app.test_client(), app
        db.session.remove()
        db.drop_all()


def auth_header(client, email="adaptive_analyst@example.com"):
    client.post("/api/v1/auth/register", json={"email": email, "password": "SecurePassword123!", "full_name": "Adaptive Analyst", "role": "analyst"})
    response = client.post("/api/v1/auth/login", json={"email": email, "password": "SecurePassword123!"})
    return {"Authorization": f"Bearer {response.get_json()['data']['access_token']}"}


def upload_csv(client, headers, df: pd.DataFrame, filename: str, domain: str):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return client.post(
        "/api/v1/datasets/upload",
        headers=headers,
        data={"file": (BytesIO(csv_bytes), filename, "text/csv"), "domain": domain},
        content_type="multipart/form-data",
    )


def test_aep_hourly_end_to_end_adaptive_pipeline(client):
    """Scenario 1 & 2: AEP_hourly.csv (Datetime, AEP_MW) uploads, validates, and preprocesses successfully."""
    c, app = client
    headers = auth_header(c)

    # 1. Synthesize 250 rows of AEP hourly electricity data
    n = 250
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "Datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "AEP_MW": 14000.0 + 3000.0 * np.sin(np.linspace(0, 15, n)) + np.random.normal(0, 100, n),
    })

    # Upload
    up_resp = upload_csv(c, headers, df, "AEP_hourly.csv", "electricity")
    assert up_resp.status_code == 201
    dataset_id = up_resp.get_json()["data"]["dataset_id"]

    # Validate
    val_resp = c.post(f"/api/v1/datasets/{dataset_id}/validate", headers=headers)
    assert val_resp.status_code == 200
    val_data = val_resp.get_json()["data"]
    assert val_data["is_valid"] is True
    assert val_data["target"] == "AEP_MW"
    assert val_data["time_column"] == "Datetime"
    assert val_data["canonical_target"] == "demand_mw"

    # Preprocess without needing to specify target or rename columns
    prep_resp = c.post(f"/api/v1/datasets/{dataset_id}/preprocess", headers=headers, json={"generate_forecast_features": True})
    assert prep_resp.status_code == 200
    prep_data = prep_resp.get_json()["data"]
    assert prep_data["features_ready"] is True
    assert prep_data["target"] == "AEP_MW"
    assert prep_data["canonical_target"] == "demand_mw"
    assert prep_data["time_column"] == "Datetime"

    # Forecast with trained model using the preprocessed custom dataset
    fc_resp = c.post("/api/v1/forecasting/train", headers=headers, json={
        "dataset_id": dataset_id,
        "domain": "electricity",
        "model": "random_forest",
        "n_estimators": 5,
        "prediction_limit": 10,
    })
    assert fc_resp.status_code == 200
    fc_data = fc_resp.get_json()["data"]
    assert fc_data["model"] == "random_forest"
    assert fc_data["target"] == "AEP_MW"


def test_ambiguous_target_requires_explicit_choice(client):
    """Scenario 5: Multiple candidate targets prompt ambiguity unless explicit target is given."""
    c, app = client
    headers = auth_header(c, "retail_analyst@example.com")

    n = 150
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "TransactionDate": dates.strftime("%Y-%m-%d"),
        "Sales": 5000.0 + np.random.normal(0, 200, n),
        "Quantity": np.random.randint(10, 100, n).astype(float),
        "Store": ["Store_A"] * n,
    })

    up_resp = upload_csv(c, headers, df, "ambiguous_retail.csv", "retail")
    assert up_resp.status_code == 201
    dataset_id = up_resp.get_json()["data"]["dataset_id"]

    # Validate should detect ambiguity
    val_resp = c.post(f"/api/v1/datasets/{dataset_id}/validate", headers=headers)
    assert val_resp.status_code == 200
    val_data = val_resp.get_json()["data"]
    assert val_data["is_ambiguous_target"] is True
    assert "Sales" in val_data["ambiguous_targets"]
    assert "Quantity" in val_data["ambiguous_targets"]

    # Preprocessing without target must return 400
    prep_fail = c.post(f"/api/v1/datasets/{dataset_id}/preprocess", headers=headers, json={})
    assert prep_fail.status_code == 400
    assert "ambiguous" in prep_fail.get_json()["message"].lower()

    # Preprocessing with explicit target="Quantity" succeeds
    prep_ok = c.post(f"/api/v1/datasets/{dataset_id}/preprocess", headers=headers, json={
        "target": "Quantity",
        "generate_forecast_features": True,
    })
    assert prep_ok.status_code == 200
    prep_data = prep_ok.get_json()["data"]
    assert prep_data["target"] == "Quantity"
    assert prep_data["canonical_target"] == "sales"


def test_invalid_datasets_rejection(client):
    """Scenario 6: Constant target (zero variance) and missing timestamps return clear 400 errors."""
    c, app = client
    headers = auth_header(c, "invalid_analyst@example.com")

    # Case A: Constant target
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    df_const = pd.DataFrame({
        "Datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "AEP_MW": [5000.0] * n,
    })
    up_const = upload_csv(c, headers, df_const, "const_power.csv", "electricity")
    ds_const_id = up_const.get_json()["data"]["dataset_id"]

    val_const = c.post(f"/api/v1/datasets/{ds_const_id}/validate", headers=headers)
    assert val_const.status_code == 422

    prep_const = c.post(f"/api/v1/datasets/{ds_const_id}/preprocess", headers=headers)
    assert prep_const.status_code == 400
    assert "constant" in prep_const.get_json()["message"].lower() or "variance" in prep_const.get_json()["message"].lower()

    # Case B: Missing timestamps
    df_no_time = pd.DataFrame({
        "Store": ["Store_1"] * n,
        "Demand": np.random.normal(500, 50, n),
    })
    up_no_time = upload_csv(c, headers, df_no_time, "no_time.csv", "electricity")
    ds_no_time_id = up_no_time.get_json()["data"]["dataset_id"]

    prep_no_time = c.post(f"/api/v1/datasets/{ds_no_time_id}/preprocess", headers=headers)
    assert prep_no_time.status_code == 400
    assert "timestamp" in prep_no_time.get_json()["message"].lower() or "time" in prep_no_time.get_json()["message"].lower()


def test_explicit_column_mapping_precedence(client):
    """Scenario 7: Explicit user mappings override automatic heuristics."""
    c, app = client
    headers = auth_header(c, "mapping_analyst@example.com")

    n = 120
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "MyDate": dates.strftime("%Y-%m-%d"),
        "AlternativeDate": dates.strftime("%Y-%m-%d"),
        "PrimaryMetric": np.random.uniform(10, 50, n),
        "TargetMetric": 100.0 + 10.0 * np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 2, n),
    })

    up = upload_csv(c, headers, df, "custom_mapped.csv", "retail")
    ds_id = up.get_json()["data"]["dataset_id"]

    prep_resp = c.post(f"/api/v1/datasets/{ds_id}/preprocess", headers=headers, json={
        "target": "TargetMetric",
        "time_column": "AlternativeDate",
        "generate_forecast_features": True,
    })
    assert prep_resp.status_code == 200
    prep_data = prep_resp.get_json()["data"]
    assert prep_data["target"] == "TargetMetric"
    assert prep_data["time_column"] == "AlternativeDate"
    assert prep_data["canonical_target"] == "sales"


def test_target_switch_recalculates_fingerprint(client):
    """Scenario 8: Switching target recalculates fingerprint and generates distinct feature directories."""
    c, app = client
    headers = auth_header(c, "switch_analyst@example.com")

    n = 120
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "MetricOne": 100.0 + np.random.normal(0, 5, n),
        "MetricTwo": 5000.0 + np.random.normal(0, 100, n),
    })

    up = upload_csv(c, headers, df, "switch_targets.csv", "retail")
    ds_id = up.get_json()["data"]["dataset_id"]

    # Preprocess with MetricOne
    prep1 = c.post(f"/api/v1/datasets/{ds_id}/preprocess", headers=headers, json={"target": "MetricOne", "generate_forecast_features": True})
    assert prep1.status_code == 200
    p1_data = prep1.get_json()["data"]
    dir1 = p1_data["features_dir"]

    # Preprocess with MetricTwo
    prep2 = c.post(f"/api/v1/datasets/{ds_id}/preprocess", headers=headers, json={"target": "MetricTwo", "generate_forecast_features": True})
    assert prep2.status_code == 200
    p2_data = prep2.get_json()["data"]
    dir2 = p2_data["features_dir"]

    # Must be distinct feature directories
    assert dir1 != dir2
    assert p1_data["target"] == "MetricOne"
    assert p2_data["target"] == "MetricTwo"


def test_electricity_auto_select_evaluates_sarima_and_validation_rmse(client):
    """Scenario 10 & 11: SARIMA participates in electricity Auto Select and uses validation RMSE."""
    c, app = client
    headers = auth_header(c, "autoselect_analyst@example.com")

    # Hourly electricity series with 100 observations
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    df = pd.DataFrame({
        "Datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "Power_MW": 12000.0 + 2000.0 * np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 50, n),
    })

    up = upload_csv(c, headers, df, "autoselect_elec.csv", "electricity")
    ds_id = up.get_json()["data"]["dataset_id"]

    prep = c.post(f"/api/v1/datasets/{ds_id}/preprocess", headers=headers, json={"generate_forecast_features": True})
    assert prep.status_code == 200
    f_dir = prep.get_json()["data"]["features_dir"]

    # Compare models with default candidates (Auto Select candidate pool)
    comp_resp = c.post("/api/v1/forecasting/compare", headers=headers, json={
        "domain": "electricity",
        "dataset_id": ds_id,
        "selection_metric": "rmse",
    })
    assert comp_resp.status_code == 200
    comp_data = comp_resp.get_json()["data"]
    assert comp_data["selection_split"] == "validation"
    assert comp_data["selection_metric"] == "rmse"

    ranked_models = [r["model"] for r in comp_data["ranking"]]
    # SARIMA must participate in candidate evaluation when eligible
    assert "sarima" in ranked_models
    assert "random_forest" in ranked_models

    for item in comp_data["ranking"]:
        assert "validation_metrics" in item
        assert "rmse" in item["validation_metrics"]


def test_geographic_isolation_intact(client):
    """Scenario 9: Geographic hierarchy filtering maintains strict location isolation."""
    c, app = client
    headers = auth_header(c, "geo_analyst@example.com")

    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    df_loc_a = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Location": ["Alpha"] * 60,
        "Sales": 100.0 + np.random.normal(0, 5, 60),
    })
    df_loc_b = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Location": ["Beta"] * 60,
        "Sales": 500.0 + np.random.normal(0, 10, 60),
    })
    combined_df = pd.concat([df_loc_a, df_loc_b], ignore_index=True)

    up = upload_csv(c, headers, combined_df, "multi_loc.csv", "retail")
    ds_id = up.get_json()["data"]["dataset_id"]

    # Preprocess
    prep = c.post(f"/api/v1/datasets/{ds_id}/preprocess", headers=headers, json={"generate_forecast_features": True})
    assert prep.status_code == 200

    # Geography endpoint returns locations
    geo_resp = c.post(f"/api/v1/datasets/{ds_id}/geography", headers=headers, json={})
    assert geo_resp.status_code == 200
    geo_data = geo_resp.get_json()["data"]
    assert "Alpha" in geo_data["available_locations"]
    assert "Beta" in geo_data["available_locations"]

    # Forecasting for Location "Alpha" succeeds
    fc_alpha = c.post("/api/v1/forecasting/train", headers=headers, json={
        "dataset_id": ds_id,
        "domain": "retail",
        "location": "Alpha",
        "model": "random_forest",
        "n_estimators": 5,
        "prediction_limit": 5,
    })
    assert fc_alpha.status_code == 200
    assert fc_alpha.get_json()["data"]["dataset_context"]["location"] == "Alpha"

