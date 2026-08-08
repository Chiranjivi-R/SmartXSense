"""Settings routes exposing read-only system information."""

from flask import jsonify

from ...services.app_summary_service import AppSummaryError, get_models_catalog, get_system_info
from . import settings_bp


def _error_response(err: AppSummaryError):
    return jsonify({"status": "error", "code": err.code, "message": err.message}), err.code


@settings_bp.get("")
def settings_system():
    try:
        data = get_system_info()
        return jsonify({"status": "success", "code": 200, "data": data}), 200
    except AppSummaryError as err:
        return _error_response(err)
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": "Internal server error"}), 500


@settings_bp.get("/models")
def settings_models():
    try:
        data = get_models_catalog()
        return jsonify({"status": "success", "code": 200, "data": data}), 200
    except AppSummaryError as err:
        return _error_response(err)
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": "Internal server error"}), 500
