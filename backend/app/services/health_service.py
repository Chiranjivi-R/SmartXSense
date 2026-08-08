"""Lightweight application health checks (no ML training/inference)."""

from __future__ import annotations

from typing import Any

from ml_engine.history.forecast_history import list_forecast_history, resolve_history_root
from ml_engine.models.trainer import DOMAIN_TARGET_MAP


def _check_database() -> str:
    """Optional DB ping. Auth/datasets use SQLAlchemy; ML core does not require DB."""
    try:
        from flask import has_app_context

        if not has_app_context():
            return "skipped"
        from sqlalchemy import text

        from ..extensions import db

        db.session.execute(text("SELECT 1"))
        return "available"
    except Exception:
        return "unavailable"


def get_health_status() -> dict[str, Any]:
    """Return subsystem availability without expensive ML work."""
    backend = "ok"
    ml_engine = "unavailable"
    history_store = "unavailable"
    database = "unavailable"

    try:
        if DOMAIN_TARGET_MAP:
            ml_engine = "available"
    except Exception:
        ml_engine = "unavailable"

    try:
        root = resolve_history_root()
        root.mkdir(parents=True, exist_ok=True)
        list_forecast_history(limit=1, history_root=root)
        history_store = "available"
    except Exception:
        history_store = "degraded"

    database = _check_database()

    # ML forecasting can operate even if DB is down (file-based artifacts/history).
    overall = "ok"
    if ml_engine != "available":
        overall = "error"
    elif history_store == "unavailable":
        overall = "degraded"
    elif database == "unavailable":
        overall = "degraded"

    return {
        "service": "SmartXSense",
        "backend": backend,
        "ml_engine": ml_engine,
        "history_store": history_store,
        "database": database,
        "overall": overall,
        "supported_domains": sorted(DOMAIN_TARGET_MAP.keys()),
        "notes": [
            "Health does not run model training or inference.",
            "Database supports auth/datasets; forecasting remains usable when database is unavailable.",
        ],
    }
