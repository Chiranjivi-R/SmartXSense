"""Explainability API routes for feature importance and local attribution explanations."""

from flask import jsonify, request

from ...services.ml_service import MLError, explain_service
from . import explainability_bp


def _error_response(err: MLError):
    return jsonify({"status": "error", "code": err.code, "message": err.message}), err.code


@explainability_bp.get("")
def module_status():
    return jsonify({"status": "success", "code": 200, "message": "Explainability module initialized"})


@explainability_bp.post("/explain")
def explain_route():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            raise MLError("Invalid or missing JSON payload")

        result = explain_service(payload)
        return jsonify({"status": "success", "code": 200, "data": result}), 200
    except MLError as err:
        return _error_response(err)
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": "Internal server error"}), 500
