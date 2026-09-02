"""Authenticated dataset-management endpoints."""

from flask import jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from ...services.auth_service import AuthError, current_user
from ...services.dataset_service import (
    DatasetError,
    dataset_detail,
    delete_dataset,
    error_payload,
    get_dataset_geography_service,
    list_datasets,
    list_item,
    owned_dataset,
    profile_dataset_service,
    upload_dataset,
    upload_response,
    validate_dataset,
)
from ...services.preprocessing_service import preprocess_dataset
from . import datasets_bp


def _error_response(error: DatasetError):
    return jsonify(error_payload(error)), error.code


def _authenticated_user():
    """Ensure a valid access token still represents an active user."""
    try:
        return current_user(get_jwt_identity())
    except AuthError as error:
        raise DatasetError(error.message, error.code) from error


def _require_dataset_editor():
    """Return a user allowed to change datasets, or a deliberate 403.

    Dataset mutation is protected on the server as well as in the UI.  The
    legacy ``analyst`` role remains the data-scientist role for backwards
    compatibility with existing accounts.
    """
    user = _authenticated_user()
    if user.role not in {"admin", "analyst", "data_scientist"}:
        raise DatasetError("Insufficient permissions to modify datasets", 403)
    return user


@datasets_bp.post("/upload")
@jwt_required()
def upload():
    try:
        # Multipart fields are populated in request.form. Keep the documented
        # query-string form as a compatibility fallback for existing clients.
        domain = request.form.get("domain") or request.args.get("domain")
        dataset = upload_dataset(request.files.get("file"), domain=domain, user_id=_require_dataset_editor().user_id)
        return jsonify({"status": "success", "code": 201, "message": "Dataset uploaded successfully", "data": upload_response(dataset)}), 201
    except DatasetError as error:
        return _error_response(error)


@datasets_bp.get("")
@jwt_required(optional=True)
def list_all():
    try:
        limit = int(request.args.get("limit", 20))
        offset = int(request.args.get("offset", 0))
        user_id = None
        identity = get_jwt_identity()
        if identity:
            try:
                user_id = current_user(identity).user_id
            except AuthError:
                pass
        total, datasets = list_datasets(user_id, domain=request.args.get("domain"), limit=limit, offset=offset)
        return jsonify({"status": "success", "code": 200, "data": {"total": total, "datasets": datasets}})
    except ValueError:
        return _error_response(DatasetError("Invalid pagination parameters"))
    except DatasetError as error:
        return _error_response(error)


@datasets_bp.get("/<dataset_id>")
@jwt_required()
def details(dataset_id: str):
    try:
        preview_rows = int(request.args.get("preview_rows", 10))
        return jsonify({"status": "success", "code": 200, "data": dataset_detail(owned_dataset(dataset_id, _authenticated_user().user_id), preview_rows)})
    except ValueError:
        return _error_response(DatasetError("Invalid preview_rows parameter"))
    except DatasetError as error:
        return _error_response(error)


@datasets_bp.delete("/<dataset_id>")
@jwt_required()
def delete(dataset_id: str):
    try:
        user = _authenticated_user()
        if user.role not in {"admin", "analyst", "data_scientist"}:
            raise DatasetError("Insufficient permissions to delete datasets", 403)
        dataset = owned_dataset(dataset_id, user.user_id)
        delete_dataset(dataset)
        return jsonify({"status": "success", "code": 200, "message": f"Dataset {dataset_id} deleted successfully"})
    except DatasetError as error:
        return _error_response(error)


@datasets_bp.post("/<dataset_id>/validate")
@jwt_required()
def validate(dataset_id: str):
    try:
        dataset = owned_dataset(dataset_id, _require_dataset_editor().user_id)
        _, result = validate_dataset(dataset, request.get_json(silent=True))
        if not result["is_valid"]:
            return _error_response(DatasetError("Dataset validation failed", 422, {"schema": result["validation_logs"]}))
        return jsonify({"status": "success", "code": 200, "data": result})
    except DatasetError as error:
        return _error_response(error)


@datasets_bp.post("/<dataset_id>/preprocess")
@jwt_required()
def preprocess(dataset_id: str):
    try:
        dataset = owned_dataset(dataset_id, _require_dataset_editor().user_id)
        _, result = preprocess_dataset(dataset, request.get_json(silent=True))
        return jsonify({
            "status": "success",
            "code": 200,
            "message": "Preprocessing and feature engineering completed",
            "data": result,
        })
    except DatasetError as error:
        return _error_response(error)


@datasets_bp.route("/<dataset_id>/profile", methods=["GET", "POST"])
@jwt_required(optional=True)
def profile(dataset_id: str):
    try:
        user_id = None
        identity = get_jwt_identity()
        if identity:
            try:
                user_id = current_user(identity).user_id
            except AuthError:
                pass

        dataset = owned_dataset(dataset_id, user_id or "demo_user")
        body = request.get_json(silent=True) if request.method == "POST" else {}
        domain = request.args.get("domain") or (body.get("domain") if isinstance(body, dict) else None)
        target = request.args.get("target") or (body.get("target") if isinstance(body, dict) else None)

        result = profile_dataset_service(dataset, domain=domain, requested_target=target)
        return jsonify({"status": "success", "code": 200, "data": result})
    except DatasetError as error:
        return _error_response(error)


@datasets_bp.route("/<dataset_id>/geography", methods=["GET", "POST"])
@jwt_required(optional=True)
def geography(dataset_id: str):
    try:
        user_id = None
        identity = get_jwt_identity()
        if identity:
            try:
                user_id = current_user(identity).user_id
            except AuthError:
                pass

        dataset = owned_dataset(dataset_id, user_id or "demo_user")
        body = request.get_json(silent=True) if request.method == "POST" else {}
        if not isinstance(body, dict):
            body = {}

        nested_geo = body.get("geo_filters") if isinstance(body.get("geo_filters"), dict) else {}
        geo_filters = {
            "country": request.args.get("country") or body.get("country") or nested_geo.get("country"),
            "state": request.args.get("state") or body.get("state") or nested_geo.get("state"),
            "district": request.args.get("district") or body.get("district") or nested_geo.get("district"),
            "region": request.args.get("region") or body.get("region") or nested_geo.get("region"),
            "city": request.args.get("city") or body.get("city") or nested_geo.get("city"),
            "city_tier": request.args.get("city_tier") or body.get("city_tier") or nested_geo.get("city_tier"),
            "location": request.args.get("location") or body.get("location") or nested_geo.get("location"),
        }

        result = get_dataset_geography_service(dataset, geo_filters=geo_filters)
        return jsonify({"status": "success", "code": 200, "data": result})
    except DatasetError as error:
        return _error_response(error)

