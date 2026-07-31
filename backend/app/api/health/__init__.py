"""Health API blueprint."""

from flask import Blueprint

health_bp = Blueprint("health", __name__, url_prefix="/api/v1/health")

from . import routes  # noqa: E402, F401
