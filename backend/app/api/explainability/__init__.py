"""Explainability API blueprint."""

from flask import Blueprint

explainability_bp = Blueprint("explainability", __name__, url_prefix="/api/v1/explainability")

from . import routes  # noqa: E402, F401
