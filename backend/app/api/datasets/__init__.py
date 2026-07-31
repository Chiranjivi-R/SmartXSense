"""Datasets API blueprint."""

from flask import Blueprint

datasets_bp = Blueprint("datasets", __name__, url_prefix="/api/v1/datasets")

from . import routes  # noqa: E402, F401
