"""Integration tests for the documented authentication endpoints."""

from datetime import timedelta
from pathlib import Path
import sys

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


@pytest.fixture()
def client():
    app = create_app(TestConfig)
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


def registration_payload(**overrides):
    data = {"email": "Analyst@SmartXSense.ai", "password": "SecurePassword123!", "full_name": "Data Analyst", "role": "analyst"}
    data.update(overrides)
    return data


def register(client):
    return client.post("/api/v1/auth/register", json=registration_payload())


def test_successful_registration(client):
    response = register(client)
    body = response.get_json()
    assert response.status_code == 201
    assert body["data"]["email"] == "analyst@smartxsense.ai"
    assert body["data"]["created_at"].endswith("Z")
    assert "password_hash" not in body["data"]


def test_duplicate_registration(client):
    register(client)
    response = register(client)
    assert response.status_code == 409
    assert response.get_json()["status"] == "error"


def test_successful_login(client):
    register(client)
    response = client.post("/api/v1/auth/login", json={"email": "ANALYST@SMARTXSENSE.AI", "password": "SecurePassword123!"})
    body = response.get_json()
    assert response.status_code == 200
    assert body["data"]["access_token"]
    assert body["data"]["refresh_token"]
    assert body["data"]["expires_in"] == 3600


def test_incorrect_password(client):
    register(client)
    response = client.post("/api/v1/auth/login", json={"email": "analyst@smartxsense.ai", "password": "WrongPassword123!"})
    assert response.status_code == 401
    assert response.get_json()["message"] == "Invalid email or password credentials"


def test_refresh_token(client):
    register(client)
    login = client.post("/api/v1/auth/login", json={"email": "analyst@smartxsense.ai", "password": "SecurePassword123!"}).get_json()
    response = client.post("/api/v1/auth/refresh", json={"refresh_token": login["data"]["refresh_token"]})
    assert response.status_code == 200
    assert response.get_json()["data"]["access_token"]


def test_me_without_token(client):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401
    assert response.get_json()["status"] == "error"


def test_me_with_valid_token(client):
    register(client)
    login = client.post("/api/v1/auth/login", json={"email": "analyst@smartxsense.ai", "password": "SecurePassword123!"}).get_json()
    response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {login['data']['access_token']}"})
    body = response.get_json()
    assert response.status_code == 200
    assert body["data"]["email"] == "analyst@smartxsense.ai"
    assert "password_hash" not in body["data"]
