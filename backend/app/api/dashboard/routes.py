"""Dashboard summary routes backed by forecast history and capability matrices."""

from flask import jsonify, request

from ...services.app_summary_service import AppSummaryError, get_dashboard_summary, get_models_catalog
from . import dashboard_bp


def _error_response(err: AppSummaryError):
    return jsonify({"status": "error", "code": err.code, "message": err.message}), err.code


@dashboard_bp.get("")
def dashboard_summary():
    try:
        recent_limit_raw = request.args.get("recent_limit", 10)
        try:
            recent_limit = int(recent_limit_raw)
        except (TypeError, ValueError) as err:
            raise AppSummaryError("Parameter 'recent_limit' must be an integer between 1 and 50") from err
        data = get_dashboard_summary(recent_limit=recent_limit)
        return jsonify({"status": "success", "code": 200, "data": data}), 200
    except AppSummaryError as err:
        return _error_response(err)
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": "Internal server error"}), 500


@dashboard_bp.get("/models")
def dashboard_models():
    try:
        data = get_models_catalog()
        return jsonify({"status": "success", "code": 200, "data": data}), 200
    except AppSummaryError as err:
        return _error_response(err)
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": "Internal server error"}), 500
