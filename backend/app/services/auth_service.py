"""Authentication business logic and safe response serialization."""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import UTC, datetime

from flask import current_app
from flask_jwt_extended import create_access_token, create_refresh_token, decode_token
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.security import check_password_hash, generate_password_hash

from ..extensions import db
from ..models import User, UserSession


EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
ALLOWED_ROLES = {"admin", "analyst", "viewer"}


class AuthError(Exception):
    """Expected authentication or request-validation failure."""

    def __init__(self, message: str, code: int = 400, errors: dict | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.errors = errors


def error_payload(error: AuthError) -> dict:
    """Return the API contract's standard error envelope."""
    payload = {"status": "error", "code": error.code, "message": error.message}
    if error.errors:
        payload["errors"] = error.errors
    return payload


def _require_json(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise AuthError("A JSON request body is required")
    return payload


def _required_string(payload: dict, field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise AuthError("Validation failed", errors={field: ["This field is required."]})
    return value.strip()


def _normalise_email(value: str) -> str:
    email = value.strip().lower()
    if len(email) > 255 or not EMAIL_PATTERN.fullmatch(email):
        raise AuthError("Validation failed", errors={"email": ["Enter a valid email address."]})
    return email


def _validate_password(password: str) -> None:
    if len(password) < 8 or not all((re.search(r"[a-z]", password), re.search(r"[A-Z]", password), re.search(r"\d", password), re.search(r"[^\w\s]", password))):
        raise AuthError(
            "Validation failed",
            errors={"password": ["Password must be at least 8 characters and include upper-case, lower-case, numeric, and special characters."]},
        )


def serialize_user(user: User, *, include_preferences: bool = False) -> dict:
    """Serialize only fields permitted by the public authentication API."""
    data = {
        "user_id": user.user_id,
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
    }
    if include_preferences:
        # Preferences are not represented by default rows in the schema.  Return
        # the contract's safe defaults until user settings functionality is added.
        data["preferences"] = {"theme": "dark", "notifications": True}
    return data


def api_timestamp(value: str) -> str:
    """Render SQLite's UTC timestamp in the API contract's ISO-8601 form."""
    if "T" in value:
        return value if value.endswith(("Z", "+00:00")) else f"{value}Z"
    return f"{value.replace(' ', 'T')}Z"


def register_user(payload: object) -> User:
    """Validate and persist a newly registered user."""
    body = _require_json(payload)
    email = _normalise_email(_required_string(body, "email"))
    password = _required_string(body, "password")
    full_name = _required_string(body, "full_name")
    role = _required_string(body, "role").lower()

    if len(full_name) > 255:
        raise AuthError("Validation failed", errors={"full_name": ["Must not exceed 255 characters."]})
    if role not in ALLOWED_ROLES:
        raise AuthError("Validation failed", errors={"role": ["Must be one of: admin, analyst, viewer."]})
    _validate_password(password)

    try:
        if db.session.scalar(db.select(User).where(User.email == email)) is not None:
            raise AuthError("Email already registered", 409)
        user = User(
            user_id=str(uuid.uuid4()),
            email=email,
            password_hash=generate_password_hash(password),
            full_name=full_name,
            role=role,
        )
        db.session.add(user)
        db.session.commit()
        return user
    except AuthError:
        raise
    except SQLAlchemyError as exc:
        db.session.rollback()
        # A uniqueness race is intentionally reported as the documented conflict.
        if "unique" in str(exc).lower():
            raise AuthError("Email already registered", 409) from exc
        raise AuthError("Unable to register user", 500) from exc


def _access_expiry_seconds() -> int:
    expiry = current_app.config["JWT_ACCESS_TOKEN_EXPIRES"]
    return int(expiry.total_seconds())


def _refresh_token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def login_user(payload: object, *, ip_address: str | None, user_agent: str | None) -> tuple[User, str, str]:
    """Authenticate a user and create a persisted refresh-token session."""
    body = _require_json(payload)
    email = _normalise_email(_required_string(body, "email"))
    password = _required_string(body, "password")

    try:
        user = db.session.scalar(db.select(User).where(User.email == email))
        if user is None or not user.is_active or not check_password_hash(user.password_hash, password):
            raise AuthError("Invalid email or password credentials", 401)

        access_token = create_access_token(identity=user.user_id)
        refresh_token = create_refresh_token(identity=user.user_id)
        refresh_claims = decode_token(refresh_token)
        expires_at = datetime.fromtimestamp(refresh_claims["exp"], UTC).isoformat().replace("+00:00", "Z")
        session = UserSession(
            session_id=str(uuid.uuid4()),
            user_id=user.user_id,
            refresh_token=_refresh_token_digest(refresh_token),
            ip_address=(ip_address or None)[:45],
            user_agent=(user_agent or None)[:512],
            expires_at=expires_at,
        )
        db.session.add(session)
        db.session.commit()
        return user, access_token, refresh_token
    except AuthError:
        raise
    except SQLAlchemyError as exc:
        db.session.rollback()
        raise AuthError("Unable to authenticate user", 500) from exc


def refresh_access_token(refresh_token: str) -> str:
    """Validate a persisted refresh token and issue a replacement access token."""
    try:
        claims = decode_token(refresh_token)
    except Exception as exc:  # Flask-JWT-Extended translates JWT decode failures here.
        raise AuthError("Invalid or expired refresh token", 401) from exc

    if claims.get("type") != "refresh" or not isinstance(claims.get("sub"), str):
        raise AuthError("Invalid or expired refresh token", 401)

    try:
        session = db.session.scalar(
            db.select(UserSession).where(
                UserSession.user_id == claims["sub"],
                UserSession.refresh_token == _refresh_token_digest(refresh_token),
                UserSession.is_revoked.is_(False),
            )
        )
        user = db.session.get(User, claims["sub"])
        if session is None or user is None or not user.is_active:
            raise AuthError("Invalid or expired refresh token", 401)
        return create_access_token(identity=user.user_id)
    except AuthError:
        raise
    except SQLAlchemyError as exc:
        raise AuthError("Unable to refresh access token", 500) from exc


def current_user(user_id: str) -> User:
    """Load an active user represented by an access-token identity."""
    try:
        user = db.session.get(User, user_id)
    except SQLAlchemyError as exc:
        raise AuthError("Unable to retrieve user profile", 500) from exc
    if user is None or not user.is_active:
        raise AuthError("User not found", 401)
    return user


def access_expiry_seconds() -> int:
    """Expose access-token expiry in the API response's documented units."""
    return _access_expiry_seconds()
