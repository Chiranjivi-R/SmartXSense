"""Application summary services for Dashboard, Reports, Models, and Settings.

Derives read-only facts from forecast history and the authoritative capability matrix.
Does not invent metrics or expose arbitrary filesystem paths.
"""

from __future__ import annotations

from typing import Any

from ml_engine.history.forecast_history import (
    ForecastHistoryError,
    get_forecast_run,
    list_forecast_history,
)
from ml_engine.models.capability_matrix import (
    DEFAULT_COMPARE_MODELS,
    SUPPORTED_DOMAINS,
    get_models_catalog_payload,
)
from ml_engine.models.trainer import DOMAIN_TARGET_MAP


class AppSummaryError(ValueError):
    def __init__(self, message: str, code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def _safe_history_runs(limit: int = 100) -> list[dict[str, Any]]:
    try:
        return list_forecast_history(limit=limit)
    except ForecastHistoryError:
        return []
    except Exception:
        return []


def _domain_distribution(runs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {domain: 0 for domain in SUPPORTED_DOMAINS}
    for run in runs:
        domain = str(run.get("domain") or "").lower()
        if domain in counts:
            counts[domain] += 1
        elif domain:
            counts[domain] = counts.get(domain, 0) + 1
    return counts


def _model_distribution(runs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        model = str(run.get("model") or "").lower()
        if not model:
            continue
        counts[model] = counts.get(model, 0) + 1
    return counts


def get_dashboard_summary(*, recent_limit: int = 10) -> dict[str, Any]:
    """Build a real dashboard summary from forecast history metadata."""
    if isinstance(recent_limit, bool) or not isinstance(recent_limit, int):
        raise AppSummaryError("Parameter 'recent_limit' must be an integer between 1 and 50")
    if recent_limit < 1 or recent_limit > 50:
        raise AppSummaryError("Parameter 'recent_limit' must be between 1 and 50")

    runs = _safe_history_runs(limit=100)
    total_runs = len(runs)
    future_count = sum(1 for run in runs if run.get("forecast_type") == "future")
    test_count = sum(1 for run in runs if run.get("forecast_type") == "test_evaluation")

    latest = runs[0] if runs else None
    latest_metrics = None
    if latest and isinstance(latest.get("metrics"), dict):
        latest_metrics = {
            key: latest["metrics"].get(key)
            for key in ("mae", "rmse", "mape", "r2")
            if latest["metrics"].get(key) is not None
        }

    catalog = get_models_catalog_payload()
    return {
        "total_forecast_runs": total_runs,
        "test_evaluation_runs": test_count,
        "future_forecast_runs": future_count,
        "available_domains": SUPPORTED_DOMAINS,
        "available_models": [item["key"] for item in catalog["models"]],
        "domain_distribution": _domain_distribution(runs),
        "model_distribution": _model_distribution(runs),
        "latest_forecast": latest,
        "latest_domain": latest.get("domain") if latest else None,
        "latest_model": latest.get("model") if latest else None,
        "latest_metrics": latest_metrics,
        "recent_runs": runs[:recent_limit],
        "default_compare_models": DEFAULT_COMPARE_MODELS,
    }


def list_reports(*, domain: str | None = None, limit: int = 20) -> dict[str, Any]:
    """List report summaries from forecast history (metadata only)."""
    domain_clean = None
    if domain is not None and str(domain).strip():
        domain_clean = str(domain).lower().strip()
        if domain_clean not in DOMAIN_TARGET_MAP:
            raise AppSummaryError(
                f"Invalid domain '{domain_clean}'. Supported: {SUPPORTED_DOMAINS}"
            )

    if isinstance(limit, bool) or not isinstance(limit, int):
        raise AppSummaryError("Parameter 'limit' must be an integer between 1 and 100")
    if limit < 1 or limit > 100:
        raise AppSummaryError("Parameter 'limit' must be between 1 and 100")

    try:
        runs = list_forecast_history(domain=domain_clean, limit=limit)
    except ForecastHistoryError as err:
        raise AppSummaryError(err.message, code=err.code) from err
    except Exception:
        runs = []

    reports = []
    for run in runs:
        reports.append(
            {
                "report_id": run.get("run_id"),
                "run_id": run.get("run_id"),
                "timestamp": run.get("timestamp"),
                "domain": run.get("domain"),
                "model": run.get("model"),
                "target": run.get("target"),
                "forecast_type": run.get("forecast_type"),
                "metrics": run.get("metrics"),
                "prediction_count": run.get("prediction_count"),
                "horizon": run.get("horizon"),
                "artifact_source": run.get("artifact_source"),
                "features_id": run.get("features_id"),
            }
        )
    return {"reports": reports, "count": len(reports)}


def get_report(run_id: str) -> dict[str, Any]:
    try:
        record = get_forecast_run(run_id)
    except ForecastHistoryError as err:
        raise AppSummaryError(err.message, code=err.code) from err
    except Exception as err:
        raise AppSummaryError("Report not found", code=404) from err

    clean = {key: value for key, value in record.items() if key != "predictions"}
    clean["report_id"] = clean.get("run_id")
    return clean


def get_models_catalog() -> dict[str, Any]:
    return get_models_catalog_payload()


def get_system_info() -> dict[str, Any]:
    """Read-only system information for Settings."""
    catalog = get_models_catalog()
    return {
        "application": "SmartXSense",
        "frontend_version": "1.0.0",
        "api_version": "v1",
        "backend_status": "ok",
        "supported_domains": SUPPORTED_DOMAINS,
        "supported_models": [item["key"] for item in catalog["models"]],
        "apis": {
            "forecasting": True,
            "future_forecasting": True,
            "explainability": True,
            "datasets": True,
            "history": True,
            "dashboard": True,
            "reports": True,
            "settings": True,
            "health": True,
        },
        "capabilities": {
            "model_persistence": True,
            "forecast_history": True,
            "client_side_export": True,
            "dataset_upload": True,
        },
        "notes": [
            "Settings are read-only system facts derived from the live application.",
            "User preference toggles are not persisted in this academic build.",
            "Forecast history and model artifacts use controlled file-based storage.",
        ],
    }
