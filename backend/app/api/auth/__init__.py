"""Authentication API blueprint."""

from flask import Blueprint

auth_bp = Blueprint("auth", __name__, url_prefix="/api/v1/auth")

from . import routes  # noqa: E402, F401
