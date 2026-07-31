"""Placeholder routes for the dashboard module."""

from flask import jsonify

from . import dashboard_bp


@dashboard_bp.get("")
def module_status():
    return jsonify({"status": "Module initialized"})
