"""Integration tests for authenticated dataset-management endpoints."""

from datetime import timedelta
from io import BytesIO
from pathlib import Path
import sys

import pytest
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))

from app import create_app  # noqa: E402
from app.extensions import db  # noqa: E402


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
        MAX_CONTENT_LENGTH = 1024 * 1024

    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()
        yield app.test_client(), app
        db.session.remove()
        db.drop_all()


def auth_header(client, email="analyst@example.com"):
    client.post("/api/v1/auth/register", json={"email": email, "password": "SecurePassword123!", "full_name": "Dataset Analyst", "role": "analyst"})
    response = client.post("/api/v1/auth/login", json={"email": email, "password": "SecurePassword123!"})
    return {"Authorization": f"Bearer {response.get_json()['data']['access_token']}"}


def upload(client, headers, contents, filename, domain, content_type="text/csv"):
    return client.post(
        "/api/v1/datasets/upload",
        headers=headers,
        data={"file": (BytesIO(contents), filename, content_type), "domain": domain},
        content_type="multipart/form-data",
    )


RETAIL_CSV = b"timestamp,store_id,item_id,sales,promo\n2026-01-01,S01,I01,420,1\n2026-01-02,S01,I01,390,0\n"
ELECTRICITY_CSV = b"timestamp,grid_load_mw\n2026-01-01T00:00:00Z,2050\n2026-01-01T01:00:00Z,1980\n"
RETAIL_PREPROCESS_CSV = (
    b"Date,City,Category,Units_Sold,Base_Price_INR,Holiday_Flag\n"
    b"2026-07-01,Delhi,Food,120,100,0\n"
    b"2026-07-01,Delhi,Food,120,100,0\n"
    b"2026-07-02,Mumbai,Home,invalid,250,1\n"
    b"2026-07-03,Mumbai,Home,135,250,1\n"
)
ELECTRICITY_PREPROCESS_CSV = (
    b"Year,Date/time,Hourly Demand Met (MW),Region\n"
    b"2026,07-01 00:00,2050,West\n"
    b"2026,07-01 00:00,2051,West\n"
    b"2026,07-01 01:00,1980,West\n"
)


def test_upload_without_jwt(client):
    test_client, _ = client
    response = upload(test_client, {}, RETAIL_CSV, "retail.csv", "retail")
    assert response.status_code == 401


def test_successful_retail_dataset_upload(client):
    test_client, _ = client
    response = upload(test_client, auth_header(test_client), RETAIL_CSV, "../../retail.csv", "retail")
    body = response.get_json()
    assert response.status_code == 201
    assert body["data"]["filename"] == "retail.csv"
    assert body["data"]["row_count"] == 2
    assert body["data"]["column_count"] == 5


def test_successful_electricity_dataset_upload(client):
    test_client, _ = client
    response = upload(test_client, auth_header(test_client), ELECTRICITY_CSV, "load.csv", "electricity")
    assert response.status_code == 201
    assert response.get_json()["data"]["domain"] == "electricity"


def test_rejects_unsupported_file_type_and_invalid_domain(client):
    test_client, _ = client
    headers = auth_header(test_client)
    wrong_type = upload(test_client, headers, b"not a dataset", "data.txt", "retail", "text/plain")
    invalid_domain = upload(test_client, headers, RETAIL_CSV, "retail.csv", "weather")
    assert wrong_type.status_code == 415
    assert invalid_domain.status_code == 400


def test_list_and_retrieve_owned_dataset(client):
    test_client, _ = client
    headers = auth_header(test_client)
    dataset_id = upload(test_client, headers, RETAIL_CSV, "retail.csv", "retail").get_json()["data"]["dataset_id"]
    listing = test_client.get("/api/v1/datasets", headers=headers)
    detail = test_client.get(f"/api/v1/datasets/{dataset_id}?preview_rows=1", headers=headers)
    assert listing.status_code == 200
    # Includes 2 demo datasets plus 1 uploaded dataset
    assert listing.get_json()["data"]["total"] == 3
    assert detail.status_code == 200
    assert detail.get_json()["data"]["preview"] == [{"timestamp": "2026-01-01", "store_id": "S01", "item_id": "I01", "sales": 420, "promo": 1}]
    assert "file_path" not in detail.get_json()["data"]


def test_dataset_registry_demo_and_domain_filtering(client):
    test_client, _ = client
    headers = auth_header(test_client, "registry-filter@example.com")
    upload(test_client, headers, RETAIL_CSV, "my_retail.csv", "retail")
    upload(test_client, headers, ELECTRICITY_CSV, "my_load.csv", "electricity")

    # Domain filter = retail
    retail_list = test_client.get("/api/v1/datasets?domain=retail", headers=headers).get_json()["data"]["datasets"]
    assert len(retail_list) == 2
    assert any(d["dataset_id"] == "demo_retail" for d in retail_list)
    assert any(d["filename"] == "my_retail.csv" for d in retail_list)

    # Domain filter = electricity
    elec_list = test_client.get("/api/v1/datasets?domain=electricity", headers=headers).get_json()["data"]["datasets"]
    assert len(elec_list) == 2
    assert any(d["dataset_id"] == "demo_electricity" for d in elec_list)
    assert any(d["filename"] == "my_load.csv" for d in elec_list)


def test_inaccessible_and_nonexistent_dataset_return_not_found(client):
    test_client, _ = client
    owner_headers = auth_header(test_client, "owner@example.com")
    other_headers = auth_header(test_client, "other@example.com")
    dataset_id = upload(test_client, owner_headers, RETAIL_CSV, "retail.csv", "retail").get_json()["data"]["dataset_id"]
    assert test_client.get(f"/api/v1/datasets/{dataset_id}", headers=other_headers).status_code == 404
    assert test_client.get("/api/v1/datasets/not-a-real-dataset", headers=owner_headers).status_code == 404


def test_delete_dataset_removes_owned_upload(client):
    test_client, app = client
    headers = auth_header(test_client)
    dataset_id = upload(test_client, headers, RETAIL_CSV, "retail.csv", "retail").get_json()["data"]["dataset_id"]
    response = test_client.delete(f"/api/v1/datasets/{dataset_id}", headers=headers)
    assert response.status_code == 200
    assert not list(Path(app.config["UPLOAD_FOLDER"]).glob("*"))
    assert test_client.get(f"/api/v1/datasets/{dataset_id}", headers=headers).status_code == 404


def test_delete_dataset_removes_processed_output(client):
    test_client, app = client
    headers = auth_header(test_client, "delete-processed@example.com")
    dataset_id = upload(test_client, headers, RETAIL_PREPROCESS_CSV, "retail.csv", "retail").get_json()["data"]["dataset_id"]
    assert test_client.post(f"/api/v1/datasets/{dataset_id}/preprocess", headers=headers, json={"generate_lags": False}).status_code == 200
    assert list(Path(app.config["PROCESSED_DATA_FOLDER"]).rglob("*.parquet"))
    assert test_client.delete(f"/api/v1/datasets/{dataset_id}", headers=headers).status_code == 200
    assert not list(Path(app.config["PROCESSED_DATA_FOLDER"]).rglob("*.parquet"))


def test_valid_dataset_validation(client):
    test_client, _ = client
    headers = auth_header(test_client)
    dataset_id = upload(test_client, headers, RETAIL_CSV, "retail.csv", "retail").get_json()["data"]["dataset_id"]
    response = test_client.post(f"/api/v1/datasets/{dataset_id}/validate", headers=headers, json={"schema_type": "retail_v1", "strict_mode": True})
    assert response.status_code == 200
    assert response.get_json()["data"]["is_valid"] is True


@pytest.mark.parametrize(
    ("contents", "filename"),
    [
        (b"Date,Product_Name,Units_Sold\n2026-07-01,Product A,120\n", "retail-variant-a.csv"),
        (b"Invoice_Date,Category,Brand,Units\n2026-07-01,Food,Brand A,120\n", "retail-variant-b.csv"),
        (RETAIL_CSV, "retail-variant-c.csv"),
    ],
)
def test_validation_accepts_supported_retail_schema_aliases(client, contents, filename):
    test_client, _ = client
    headers = auth_header(test_client, f"{filename}@example.com")
    dataset_id = upload(test_client, headers, contents, filename, "retail").get_json()["data"]["dataset_id"]
    response = test_client.post(f"/api/v1/datasets/{dataset_id}/validate", headers=headers, json={"schema_type": "retail_v1", "strict_mode": True})
    assert response.status_code == 200
    assert response.get_json()["data"]["is_valid"] is True


def test_validation_accepts_indian_hourly_electricity_aliases(client):
    test_client, _ = client
    headers = auth_header(test_client, "hourly-demand-validation@example.com")
    dataset_id = upload(test_client, headers, ELECTRICITY_PREPROCESS_CSV, "hourly-demand.csv", "electricity").get_json()["data"]["dataset_id"]
    response = test_client.post(f"/api/v1/datasets/{dataset_id}/validate", headers=headers, json={"schema_type": "electricity_v1", "strict_mode": True})
    assert response.status_code == 200
    assert response.get_json()["data"]["is_valid"] is True


def test_invalid_dataset_validation_is_persisted(client):
    test_client, _ = client
    headers = auth_header(test_client)
    invalid_csv = b"timestamp,description\n2026-01-01,missing retail measures\n"
    dataset_id = upload(test_client, headers, invalid_csv, "invalid.csv", "retail").get_json()["data"]["dataset_id"]
    response = test_client.post(f"/api/v1/datasets/{dataset_id}/validate", headers=headers, json={"schema_type": "retail_v1", "strict_mode": True})
    assert response.status_code == 422
    with test_client.application.app_context():
        from app.models import DatasetValidation
        assert db.session.scalar(db.select(DatasetValidation).where(DatasetValidation.dataset_id == dataset_id)).is_valid is False


def test_preprocess_without_jwt_or_access_to_another_users_dataset(client):
    test_client, _ = client
    owner_headers = auth_header(test_client, "preprocess-owner@example.com")
    other_headers = auth_header(test_client, "preprocess-other@example.com")
    dataset_id = upload(test_client, owner_headers, RETAIL_PREPROCESS_CSV, "retail.csv", "retail").get_json()["data"]["dataset_id"]
    assert test_client.post(f"/api/v1/datasets/{dataset_id}/preprocess").status_code == 401
    assert test_client.post(f"/api/v1/datasets/{dataset_id}/preprocess", headers=other_headers).status_code == 404


def test_preprocess_retail_normalizes_dates_targets_and_preserves_features(client):
    test_client, _ = client
    headers = auth_header(test_client, "retail-preprocess@example.com")
    dataset_id = upload(test_client, headers, RETAIL_PREPROCESS_CSV, "retail.csv", "retail").get_json()["data"]["dataset_id"]
    response = test_client.post(f"/api/v1/datasets/{dataset_id}/preprocess", headers=headers, json={"generate_lags": False})
    body = response.get_json()
    assert response.status_code == 200
    assert body["data"]["processed_rows"] == 2
    assert {"year", "month", "day_of_week", "is_weekend"} <= set(body["data"]["features_generated"])
    with test_client.application.app_context():
        from app.models import DatasetPreprocessing
        record = db.session.get(DatasetPreprocessing, body["data"]["processed_dataset_id"])
        processed = pd.read_parquet(record.processed_file_path)
    assert {"timestamp", "sales", "city", "category", "price", "holiday_flag"} <= set(processed.columns)
    assert processed["sales"].tolist() == [120, 135]
    assert processed["timestamp"].tolist() == ["2026-07-01T00:00:00Z", "2026-07-03T00:00:00Z"]


def test_preprocess_electricity_combines_timestamp_and_removes_duplicate_timestamps(client):
    test_client, _ = client
    headers = auth_header(test_client, "electricity-preprocess@example.com")
    dataset_id = upload(test_client, headers, ELECTRICITY_PREPROCESS_CSV, "demand.csv", "electricity").get_json()["data"]["dataset_id"]
    response = test_client.post(f"/api/v1/datasets/{dataset_id}/preprocess", headers=headers, json={"generate_lags": False})
    body = response.get_json()
    assert response.status_code == 200
    assert body["data"]["processed_rows"] == 2
    assert "hour" in body["data"]["features_generated"]
    with test_client.application.app_context():
        from app.models import DatasetPreprocessing
        record = db.session.get(DatasetPreprocessing, body["data"]["processed_dataset_id"])
        processed = pd.read_parquet(record.processed_file_path)
    assert processed["demand_mw"].tolist() == [2050, 1980]
    assert processed["timestamp"].tolist() == ["2026-07-01T00:00:00Z", "2026-07-01T01:00:00Z"]


@pytest.mark.parametrize(
    ("domain", "contents", "filename"),
    [
        ("retail", b"Date,City\n2026-07-01,Delhi\n", "invalid-retail.csv"),
        ("electricity", b"Date/time,Region\n2026-07-01 00:00,West\n", "invalid-electricity.csv"),
    ],
)
def test_preprocess_invalid_domain_schema(client, domain, contents, filename):
    test_client, _ = client
    headers = auth_header(test_client, f"invalid-{domain}@example.com")
    dataset_id = upload(test_client, headers, contents, filename, domain).get_json()["data"]["dataset_id"]
    response = test_client.post(f"/api/v1/datasets/{dataset_id}/preprocess", headers=headers, json={"generate_lags": False})
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


def _retail_series_csv(n_days: int = 60) -> bytes:
    lines = ["timestamp,sales,price,city"]
    for i in range(n_days):
        day = 1 + (i % 28)
        month = 1 + (i // 28)
        lines.append(f"2024-{month:02d}-{day:02d},{50 + i % 17},100,Delhi")
    return ("\n".join(lines) + "\n").encode("utf-8")


def test_upload_rejects_empty_file(client):
    test_client, _ = client
    headers = auth_header(test_client, "empty-file@example.com")
    response = upload(test_client, headers, b"", "empty.csv", "retail")
    assert response.status_code == 400


def test_upload_rejects_malformed_csv(client):
    test_client, _ = client
    headers = auth_header(test_client, "malformed@example.com")
    # Unreadable as tabular CSV with no columns
    response = upload(test_client, headers, b"\n\n\n", "bad.csv", "retail")
    assert response.status_code == 400


def test_upload_excel_when_supported(client):
    test_client, _ = client
    headers = auth_header(test_client, "excel-upload@example.com")
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=40, freq="D"),
            "sales": list(range(40)),
            "price": [10] * 40,
        }
    )
    buffer = BytesIO()
    frame.to_excel(buffer, index=False)
    contents = buffer.getvalue()
    response = upload(
        test_client,
        headers,
        contents,
        "retail.xlsx",
        "retail",
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    assert response.status_code == 201
    assert response.get_json()["data"]["row_count"] == 40


def test_upload_oversized_file_rejected(client):
    test_client, app = client
    headers = auth_header(test_client, "oversized@example.com")
    # Exceed TestConfig MAX_CONTENT_LENGTH (1 MiB)
    big = b"timestamp,sales\n" + b"2024-01-01,1\n" * 200000
    assert len(big) > app.config["MAX_CONTENT_LENGTH"]
    response = upload(test_client, headers, big, "huge.csv", "retail")
    assert response.status_code in {413, 400}


def test_preprocess_generates_forecast_feature_splits(client):
    test_client, app = client
    headers = auth_header(test_client, "features-ready@example.com")
    dataset_id = upload(test_client, headers, _retail_series_csv(60), "retail-series.csv", "retail").get_json()["data"]["dataset_id"]
    validate = test_client.post(
        f"/api/v1/datasets/{dataset_id}/validate",
        headers=headers,
        json={"schema_type": "retail_v1", "strict_mode": True},
    )
    assert validate.status_code == 200
    response = test_client.post(
        f"/api/v1/datasets/{dataset_id}/preprocess",
        headers=headers,
        json={"generate_lags": False, "generate_forecast_features": True},
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["data"]["features_ready"] is True
    assert body["data"]["features_dir"]
    assert body["data"]["train_rows"] > 0
    assert body["data"]["validation_rows"] > 0
    assert body["data"]["test_rows"] > 0
    features_dir = Path(app.config["FEATURES_DATA_FOLDER"]) / "retail" / Path(body["data"]["features_dir"]).name
    # relative path may include full relative chain — resolve via returned path under configured root
    candidate = Path(app.config["FEATURES_DATA_FOLDER"]) / "retail"
    matches = list(candidate.glob("features_*"))
    assert matches
    feature_path = matches[-1]
    assert (feature_path / "train.parquet").exists()
    assert (feature_path / "validation.parquet").exists()
    assert (feature_path / "test.parquet").exists()
    assert (feature_path / "metadata.json").exists()


def test_preprocess_small_dataset_reports_features_not_ready(client):
    test_client, _ = client
    headers = auth_header(test_client, "too-small@example.com")
    dataset_id = upload(test_client, headers, RETAIL_PREPROCESS_CSV, "retail.csv", "retail").get_json()["data"]["dataset_id"]
    response = test_client.post(
        f"/api/v1/datasets/{dataset_id}/preprocess",
        headers=headers,
        json={"generate_lags": False, "generate_forecast_features": True},
    )
    body = response.get_json()
    assert response.status_code == 200
    assert body["data"]["features_ready"] is False
    assert "features_error" in body["data"]
    assert body["data"]["processed_rows"] == 2
