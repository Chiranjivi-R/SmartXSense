"""Placeholder routes for the forecasting module."""

from flask import jsonify

from . import forecasting_bp


@forecasting_bp.get("")
def module_status():
    return jsonify({"status": "Module initialized"})
