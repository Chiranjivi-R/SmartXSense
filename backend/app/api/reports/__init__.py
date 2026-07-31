"""Reports API blueprint."""

from flask import Blueprint

reports_bp = Blueprint("reports", __name__, url_prefix="/api/v1/reports")

from . import routes  # noqa: E402, F401
