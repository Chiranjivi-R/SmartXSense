"""Authoritative SmartXSense model capability matrix.

This module is the single source of truth for what each model family supports.
Do not invent unsupported capabilities.
"""

from __future__ import annotations

from typing import Any

from ml_engine.explainability import SUPPORTED_EXPLAINABLE_MODELS
from ml_engine.models.artifact_store import is_persistence_supported
from ml_engine.models.future_forecasting import (
    FUTURE_SUPPORTED_MODELS,
    is_future_forecast_supported,
)
from ml_engine.models.trainer import DOMAIN_TARGET_MAP

SUPPORTED_DOMAINS = sorted(DOMAIN_TARGET_MAP.keys())

# Models included in default Compare / Auto Select candidate lists.
DEFAULT_COMPARE_MODELS: dict[str, list[str]] = {
    "electricity": ["arima", "random_forest", "xgboost", "trend_xgboost", "lstm", "hybrid"],
    "retail": ["naive", "random_forest", "xgboost", "lstm", "hybrid"],
}

# Base catalog entries (domains + type). Runtime flags are derived.
_MODEL_BASE: list[dict[str, Any]] = [
    {
        "key": "naive",
        "name": "Naive",
        "model_type": "baseline",
        "domains": ["retail", "electricity"],
        "test_forecast": True,
        "compare_default": False,
        "notes": "Simple baseline for comparison; not used for future forecasting.",
    },
    {
        "key": "seasonal_naive",
        "name": "Seasonal Naive",
        "model_type": "baseline",
        "domains": ["retail", "electricity"],
        "test_forecast": True,
        "compare_default": False,
        "notes": "Seasonal baseline; not used for future forecasting.",
    },
    {
        "key": "arima",
        "name": "ARIMA",
        "model_type": "statistical",
        "domains": ["electricity"],
        "test_forecast": True,
        "compare_default": True,
        "notes": "ARIMA is electricity-only for test and future forecasting.",
    },
    {
        "key": "random_forest",
        "name": "Random Forest",
        "model_type": "tabular_ensemble",
        "domains": ["retail", "electricity"],
        "test_forecast": True,
        "compare_default": True,
        "notes": None,
    },
    {
        "key": "xgboost",
        "name": "XGBoost",
        "model_type": "tabular_ensemble",
        "domains": ["retail", "electricity"],
        "test_forecast": True,
        "compare_default": True,
        "notes": None,
    },
    {
        "key": "trend_xgboost",
        "name": "Trend XGBoost",
        "model_type": "tabular_ensemble",
        "domains": ["electricity"],
        "test_forecast": True,
        "compare_default": True,
        "notes": "Detrended residual XGBoost with conservative spike weighting for non-stationary electricity demand.",
    },
    {
        "key": "lstm",
        "name": "LSTM",
        "model_type": "deep_sequence",
        "domains": ["retail", "electricity"],
        "test_forecast": True,
        "compare_default": True,
        "notes": "Supported for held-out test evaluation.",
    },
    {
        "key": "cnn_lstm",
        "name": "CNN-LSTM",
        "model_type": "deep_sequence",
        "domains": ["retail", "electricity"],
        "test_forecast": True,
        "compare_default": False,
        "notes": "Supported for held-out test evaluation; future forecasting is not enabled.",
    },
    {
        "key": "hybrid",
        "name": "Hybrid",
        "model_type": "ensemble",
        "domains": ["retail", "electricity"],
        "test_forecast": True,
        "compare_default": True,
        "notes": (
            "Hybrid supports validation-weighted test evaluation. "
            "Future forecasting is supported for electricity via future-capable components (ARIMA, RF, XGBoost)."
        ),
    },
]


def build_model_capability_matrix() -> list[dict[str, Any]]:
    """Return enriched capability rows for APIs and documentation."""
    rows: list[dict[str, Any]] = []
    for item in _MODEL_BASE:
        key = item["key"]
        domains = list(item["domains"])
        future_domains = [
            domain for domain in domains if is_future_forecast_supported(domain, key)
        ]
        rows.append(
            {
                "key": key,
                "name": item["name"],
                "model_type": item["model_type"],
                "domains": domains,
                "retail": "retail" in domains,
                "electricity": "electricity" in domains,
                "test_forecast": bool(item["test_forecast"]),
                "future_forecast_supported": bool(future_domains),
                "future_forecast_domains": future_domains,
                "persistence_supported": is_persistence_supported(key),
                "explainability_supported": key in SUPPORTED_EXPLAINABLE_MODELS,
                "compare_default": bool(item["compare_default"]),
                "auto_select_eligible": bool(item["compare_default"]),
                "evaluation_note": "Run forecast to evaluate.",
                "notes": item.get("notes"),
            }
        )
    return rows


def capability_matrix_text() -> str:
    """Human-readable matrix for documentation."""
    header = (
        "| Model | Retail | Electricity | Test Forecast | Future | Persistence | "
        "Explainability | Compare/Auto Select |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for row in build_model_capability_matrix():
        future = (
            ", ".join(row["future_forecast_domains"])
            if row["future_forecast_supported"]
            else "no"
        )
        lines.append(
            "| {name} | {retail} | {elec} | {test} | {future} | {persist} | {xai} | {compare} |".format(
                name=row["name"],
                retail="yes" if row["retail"] else "no",
                elec="yes" if row["electricity"] else "no",
                test="yes" if row["test_forecast"] else "no",
                future=future,
                persist="yes" if row["persistence_supported"] else "no",
                xai="yes" if row["explainability_supported"] else "no",
                compare="yes" if row["compare_default"] else "optional",
            )
        )
    return "\n".join(lines)


def get_models_catalog_payload() -> dict[str, Any]:
    models = build_model_capability_matrix()
    return {
        "domains": SUPPORTED_DOMAINS,
        "models": models,
        "default_compare_models": DEFAULT_COMPARE_MODELS,
        "future_supported_by_domain": {
            domain: sorted(model_set)
            for domain, model_set in FUTURE_SUPPORTED_MODELS.items()
        },
        "explainable_models": list(SUPPORTED_EXPLAINABLE_MODELS),
        "matrix_markdown": capability_matrix_text(),
    }
