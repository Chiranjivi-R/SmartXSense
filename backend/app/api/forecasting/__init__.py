"""Forecasting API blueprint."""

from flask import Blueprint

forecasting_bp = Blueprint("forecasting", __name__, url_prefix="/api/v1/forecasting")

from . import routes  # noqa: E402, F401
