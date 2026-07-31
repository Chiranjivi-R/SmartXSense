"""Placeholder routes for the authentication module."""

from flask import jsonify

from . import auth_bp


@auth_bp.get("")
def module_status():
    return jsonify({"status": "Module initialized"})
