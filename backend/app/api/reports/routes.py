"""Reports routes backed by forecast history metadata."""

from flask import jsonify, request

from ...services.app_summary_service import AppSummaryError, get_report, list_reports
from . import reports_bp


def _error_response(err: AppSummaryError):
    return jsonify({"status": "error", "code": err.code, "message": err.message}), err.code


@reports_bp.get("")
def reports_list():
    try:
        limit_raw = request.args.get("limit", 20)
        try:
            limit = int(limit_raw)
        except (TypeError, ValueError) as err:
            raise AppSummaryError("Parameter 'limit' must be an integer between 1 and 100") from err

        data = list_reports(domain=request.args.get("domain"), limit=limit)
        return jsonify({"status": "success", "code": 200, "data": data}), 200
    except AppSummaryError as err:
        return _error_response(err)
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": "Internal server error"}), 500


@reports_bp.get("/<run_id>")
def reports_get(run_id: str):
    try:
        data = get_report(run_id)
        return jsonify({"status": "success", "code": 200, "data": data}), 200
    except AppSummaryError as err:
        return _error_response(err)
    except Exception as err:
        return jsonify({"status": "error", "code": 500, "message": "Internal server error"}), 500
