"""Placeholder routes for the health module."""

from flask import jsonify

from . import health_bp


@health_bp.get("")
def module_status():
    return jsonify({"status": "Module initialized"})
