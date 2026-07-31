"""Placeholder routes for the settings module."""

from flask import jsonify

from . import settings_bp


@settings_bp.get("")
def module_status():
    return jsonify({"status": "Module initialized"})
