"""Forecasting API routes for training, prediction, and validation-based model comparison."""

from flask import jsonify, request

from ...services.ml_service import MLError, compare_service, forecast_service
from . import forecasting_bp


def _error_response(err: MLError):
    return jsonify({"status": "error", "code": err.code, "message": err.message}), err.code


@forecasting_bp.get("")
def module_status():
    return jsonify({"status": "success", "code": 200, "message": "Forecasting module initialized"})


@forecasting_bp.post("/train")
@forecasting_bp.post("/forecast")
def forecast_route():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            raise MLError("Invalid or missing JSON payload")

        result = forecast_service(payload)
        return jsonify({"status": "success", "code": 200, "data": result}), 200
    except MLError as err:
        return _error_response(err)
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": f"Internal server error: {err}"}), 500


@forecasting_bp.post("/compare")
def compare_route():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            raise MLError("Invalid or missing JSON payload")

        result = compare_service(payload)
        return jsonify({"status": "success", "code": 200, "data": result}), 200
    except MLError as err:
        return _error_response(err)
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": f"Internal server error: {err}"}), 500
