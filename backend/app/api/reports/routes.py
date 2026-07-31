"""Placeholder routes for the reports module."""

from flask import jsonify

from . import reports_bp


@reports_bp.get("")
def module_status():
    return jsonify({"status": "Module initialized"})
