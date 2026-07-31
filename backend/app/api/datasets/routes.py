"""Placeholder routes for the datasets module."""

from flask import jsonify

from . import datasets_bp


@datasets_bp.get("")
def module_status():
    return jsonify({"status": "Module initialized"})
