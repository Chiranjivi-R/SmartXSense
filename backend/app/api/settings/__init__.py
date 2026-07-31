"""Settings API blueprint."""

from flask import Blueprint

settings_bp = Blueprint("settings", __name__, url_prefix="/api/v1/settings")

from . import routes  # noqa: E402, F401
