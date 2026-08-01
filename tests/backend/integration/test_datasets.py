"""Integration tests for authenticated dataset-management endpoints."""

from datetime import timedelta
from io import BytesIO
from pathlib import Path
import sys

import pytest

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
    assert listing.get_json()["data"]["total"] == 1
    assert detail.status_code == 200
    assert detail.get_json()["data"]["preview"] == [{"timestamp": "2026-01-01", "store_id": "S01", "item_id": "I01", "sales": 420, "promo": 1}]
    assert "file_path" not in detail.get_json()["data"]


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


def test_valid_dataset_validation(client):
    test_client, _ = client
    headers = auth_header(test_client)
    dataset_id = upload(test_client, headers, RETAIL_CSV, "retail.csv", "retail").get_json()["data"]["dataset_id"]
    response = test_client.post(f"/api/v1/datasets/{dataset_id}/validate", headers=headers, json={"schema_type": "retail_v1", "strict_mode": True})
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
