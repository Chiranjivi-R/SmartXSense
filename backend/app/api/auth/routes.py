"""JWT authentication endpoints."""

from flask import jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from ...extensions import jwt
from ...services.auth_service import (
    AuthError,
    access_expiry_seconds,
    api_timestamp,
    current_user,
    error_payload,
    login_user,
    refresh_access_token,
    register_user,
    serialize_user,
)
from . import auth_bp


def _error_response(error: AuthError):
    return jsonify(error_payload(error)), error.code


def _request_json():
    return request.get_json(silent=True) if request.is_json else None


def _refresh_token_from_request() -> str | None:
    """Accept the contract's JSON token and standard Bearer transport."""
    body = _request_json()
    if isinstance(body, dict) and isinstance(body.get("refresh_token"), str):
        return body["refresh_token"]
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    return token if scheme.lower() == "bearer" and token else None


@auth_bp.post("/register")
def register():
    try:
        user = register_user(_request_json())
        data = serialize_user(user)
        data["created_at"] = api_timestamp(user.created_at)
        return jsonify({"status": "success", "code": 201, "message": "User registered successfully", "data": data}), 201
    except AuthError as error:
        return _error_response(error)


@auth_bp.post("/login")
def login():
    try:
        user, access_token, refresh_token = login_user(
            _request_json(), ip_address=request.remote_addr, user_agent=request.user_agent.string
        )
        return jsonify({
            "status": "success",
            "code": 200,
            "data": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_in": access_expiry_seconds(),
                "user": serialize_user(user),
            },
        })
    except AuthError as error:
        return _error_response(error)


@auth_bp.post("/refresh")
def refresh():
    token = _refresh_token_from_request()
    if not token:
        return _error_response(AuthError("Invalid or expired refresh token", 401))
    try:
        return jsonify({
            "status": "success",
            "code": 200,
            "data": {"access_token": refresh_access_token(token), "expires_in": access_expiry_seconds()},
        })
    except AuthError as error:
        return _error_response(error)


@auth_bp.get("/me")
@jwt_required()
def me():
    try:
        return jsonify({"status": "success", "code": 200, "data": serialize_user(current_user(get_jwt_identity()), include_preferences=True)})
    except AuthError as error:
        return _error_response(error)


@jwt.unauthorized_loader
def missing_jwt(reason: str):
    return _error_response(AuthError("Missing or invalid authorization token", 401))


@jwt.invalid_token_loader
def invalid_jwt(reason: str):
    return _error_response(AuthError("Missing or invalid authorization token", 401))


@jwt.expired_token_loader
def expired_jwt(jwt_header: dict, jwt_payload: dict):
    return _error_response(AuthError("Authorization token has expired", 401))
