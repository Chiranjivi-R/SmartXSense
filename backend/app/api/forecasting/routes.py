"""Forecasting API routes for training, prediction, and validation-based model comparison."""

from flask import jsonify, request

from ...services.ml_service import (
    MLError,
    compare_service,
    delete_history_run_service,
    forecast_service,
    future_forecast_service,
    get_history_run_service,
    list_history_service,
)
from . import forecasting_bp


def _error_response(err: MLError):
    return jsonify({"status": "error", "code": err.code, "message": err.message}), err.code


def _internal_error():
    return jsonify({"status": "error", "code": 500, "message": "Internal server error"}), 500


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
    except Exception:
        return _internal_error()


@forecasting_bp.post("/future")
def future_forecast_route():
    try:
        payload = request.get_json(silent=True)
        if payload is None:
            raise MLError("Invalid or missing JSON payload")

        result = future_forecast_service(payload)
        return jsonify({"status": "success", "code": 200, "data": result}), 200
    except MLError as err:
        return _error_response(err)
    except Exception:
        return _internal_error()


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
    except Exception:
        return _internal_error()


@forecasting_bp.get("/history")
def history_list_route():
    try:
        result = list_history_service(
            domain=request.args.get("domain"),
            model=request.args.get("model"),
            limit=request.args.get("limit", 20),
        )
        return jsonify({"status": "success", "code": 200, "data": result}), 200
    except MLError as err:
        return _error_response(err)
    except Exception:
        return _internal_error()


@forecasting_bp.get("/history/<run_id>")
def history_get_route(run_id: str):
    try:
        result = get_history_run_service(run_id)
        return jsonify({"status": "success", "code": 200, "data": result}), 200
    except MLError as err:
        return _error_response(err)
    except Exception:
        return _internal_error()


@forecasting_bp.delete("/history/<run_id>")
def history_delete_route(run_id: str):
    try:
        result = delete_history_run_service(run_id)
        return jsonify({"status": "success", "code": 200, "data": result}), 200
    except MLError as err:
        return _error_response(err)
    except Exception:
        return _internal_error()
