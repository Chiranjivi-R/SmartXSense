"""Health check routes."""

from flask import jsonify

from ...services.health_service import get_health_status
from . import health_bp


@health_bp.get("")
def health_check():
    try:
        data = get_health_status()
        code = 200 if data.get("overall") in {"ok", "degraded"} else 503
        return jsonify({"status": "success" if code == 200 else "error", "code": code, "data": data}), code
    except Exception:
        return jsonify({
            "status": "error",
            "code": 503,
            "message": "Health check failed",
            "data": {
                "service": "SmartXSense",
                "backend": "error",
                "ml_engine": "unavailable",
                "history_store": "unavailable",
                "overall": "error",
            },
        }), 503
