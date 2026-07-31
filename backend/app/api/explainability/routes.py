"""Placeholder routes for the explainability module."""

from flask import jsonify

from . import explainability_bp


@explainability_bp.get("")
def module_status():
    return jsonify({"status": "Module initialized"})
