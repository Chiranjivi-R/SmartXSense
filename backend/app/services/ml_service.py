"""Backend ML orchestration service for forecasting, model comparison, and XAI explanations.

Strict Security & Integration Rules:
1. Path validation prevents path traversal and enforces feature directory constraints.
2. Model comparison strictly relies on VALIDATION performance.
3. Lightweight candidates are used for quick comparison (deep learning models excluded by default).
4. Direct Python service calls (no subprocess CLI calls).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml_engine.evaluation import AdaptiveModelSelector, compare_models
from ml_engine.explainability import ForecastExplainer, SUPPORTED_EXPLAINABLE_MODELS
from ml_engine.models.random_forest import RandomForestForecaster
from ml_engine.models.trainer import DOMAIN_TARGET_MAP, TrainerError, run_training_pipeline
from ml_engine.models.xgboost_model import XGBoostForecaster


class MLError(ValueError):
    """Raised when an ML backend service operation fails."""

    def __init__(self, message: str, code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


ALLOWED_ROOT_PATHS = (
    Path(__file__).resolve().parents[3] / "ml_engine" / "data" / "features",  # Constrained feature artifact root
    Path(__file__).resolve().parents[3],                                       # Repository root
    Path("C:/tmp"),                                                            # Temporary test directories
    Path("/tmp"),
)

DEFAULT_COMPARISON_MODELS = {
    "electricity": ["arima", "random_forest", "xgboost", "hybrid"],
    "retail": ["random_forest", "xgboost", "hybrid"],
}


def validate_features_directory(features_dir_input: str | Path) -> Path:
    """Validate that features_dir is secure, exists, and contains required split artifacts.

    Security checks:
    - Path traversal attempt (e.g. '../')
    - Nonexistent path
    - Path outside allowed workspace roots
    - Missing train.parquet, validation.parquet, or test.parquet
    """
    if not features_dir_input:
        raise MLError("Missing required parameter 'features_dir'")

    raw_str = str(features_dir_input).strip()
    if ".." in raw_str:
        raise MLError("Path traversal detected in features_dir", code=400)

    try:
        resolved_path = Path(raw_str).resolve()
    except Exception as err:
        raise MLError(f"Invalid features_dir path format: {err}", code=400) from err

    if not resolved_path.exists() or not resolved_path.is_dir():
        raise MLError(f"Features directory does not exist or is not a directory: {resolved_path.as_posix()}", code=400)

    # Check allowed root containment
    is_allowed = any(
        resolved_path == root or root in resolved_path.parents
        for root in ALLOWED_ROOT_PATHS
    )
    if not is_allowed:
        raise MLError(f"Access denied to features directory outside allowed project scope: {resolved_path.as_posix()}", code=400)

    for artifact in ("train.parquet", "validation.parquet", "test.parquet"):
        if not (resolved_path / artifact).exists():
            raise MLError(f"Required feature artifact '{artifact}' missing from directory: {resolved_path.as_posix()}", code=400)

    return resolved_path


def forecast_service(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute model training and prediction pipeline for a single model or hybrid ensemble."""
    if not isinstance(payload, dict):
        raise MLError("Invalid JSON request body")

    domain = payload.get("domain", "").lower().strip()
    if not domain or domain not in DOMAIN_TARGET_MAP:
        raise MLError(f"Invalid or missing domain '{domain}'. Supported: {list(DOMAIN_TARGET_MAP.keys())}")

    model_name = payload.get("model", "").lower().strip()
    if not model_name:
        raise MLError("Missing required parameter 'model'")

    if model_name == "baseline":
        model_name = "naive"

    if model_name == "arima" and domain == "retail":
        raise MLError("ARIMA model is currently supported only for the electricity domain")

    features_dir = validate_features_directory(payload.get("features_dir"))

    prediction_limit_raw = payload.get("prediction_limit", 100)
    if isinstance(prediction_limit_raw, bool):
        raise MLError("Parameter 'prediction_limit' must be an integer between 1 and 1000", code=400)
    try:
        prediction_limit = int(prediction_limit_raw)
    except (ValueError, TypeError) as err:
        raise MLError("Parameter 'prediction_limit' must be an integer between 1 and 1000", code=400) from err

    if prediction_limit < 1 or prediction_limit > 1000:
        raise MLError("Parameter 'prediction_limit' must be between 1 and 1000", code=400)

    components = payload.get("components")
    hybrid_weighting = payload.get("hybrid_weighting", "inverse_rmse")
    order = tuple(payload.get("order", [1, 1, 1]))
    n_estimators = int(payload.get("n_estimators", 200))
    max_depth = payload.get("max_depth")
    if max_depth is not None:
        max_depth = int(max_depth)
    learning_rate = float(payload.get("learning_rate", 0.05))

    try:
        res = run_training_pipeline(
            domain=domain,
            features_dir=features_dir,
            model_name=model_name,
            components=components,
            hybrid_weighting=hybrid_weighting,
            order=order,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            prediction_limit=prediction_limit,
        )
        return res
    except TrainerError as err:
        raise MLError(f"Forecasting pipeline error: {err}", code=400) from err
    except Exception as err:
        raise MLError(f"Internal ML forecasting error: {err}", code=500) from err


def compare_service(payload: dict[str, Any]) -> dict[str, Any]:
    """Compare candidate forecasting models based strictly on validation metrics."""
    if not isinstance(payload, dict):
        raise MLError("Invalid JSON request body")

    domain = payload.get("domain", "").lower().strip()
    if not domain or domain not in DOMAIN_TARGET_MAP:
        raise MLError(f"Invalid or missing domain '{domain}'. Supported: {list(DOMAIN_TARGET_MAP.keys())}")

    features_dir = validate_features_directory(payload.get("features_dir"))
    selection_metric = payload.get("selection_metric", "rmse").lower().strip()

    models_input = payload.get("models")
    if models_input is None:
        candidate_models = DEFAULT_COMPARISON_MODELS.get(domain, ["random_forest", "xgboost", "hybrid"])
    else:
        if not isinstance(models_input, list) or not models_input:
            raise MLError("Parameter 'models' must be a non-empty list of model names")
        candidate_models = [m.lower().strip() for m in models_input]

    # Reject ARIMA for retail domain
    if domain == "retail" and "arima" in candidate_models:
        raise MLError("ARIMA model is not supported for retail domain comparison")

    reports = []
    for m in candidate_models:
        try:
            report = run_training_pipeline(
                domain=domain,
                features_dir=features_dir,
                model_name=m,
            )
            reports.append(report)
        except Exception as err:
            raise MLError(f"Failed to evaluate candidate model '{m}': {err}", code=400) from err

    try:
        comparison_res = compare_models(reports, selection_metric=selection_metric)
        selector = AdaptiveModelSelector(selection_metric=selection_metric)
        selection_res = selector.select_best_model(reports, selection_metric=selection_metric)
    except Exception as err:
        raise MLError(f"Model comparison calculation failed: {err}", code=400) from err

    return {
        "domain": domain,
        "selection_metric": selection_metric,
        "selection_split": "validation",
        "ranking": comparison_res["ranking"],
        "recommended_model": comparison_res["recommended_model"],
        "adaptive_selection": selection_res,
    }


def explain_service(payload: dict[str, Any]) -> dict[str, Any]:
    """Generate SHAP or LIME explanations for tree-based forecasting models."""
    if not isinstance(payload, dict):
        raise MLError("Invalid JSON request body")

    domain = payload.get("domain", "").lower().strip()
    if not domain or domain not in DOMAIN_TARGET_MAP:
        raise MLError(f"Invalid or missing domain '{domain}'. Supported: {list(DOMAIN_TARGET_MAP.keys())}")

    model_name = payload.get("model", "").lower().strip()
    if not model_name:
        raise MLError("Missing required parameter 'model'")

    if model_name not in SUPPORTED_EXPLAINABLE_MODELS:
        raise MLError(
            "Explainability is currently supported for Random Forest and XGBoost.",
            code=400,
        )

    features_dir = validate_features_directory(payload.get("features_dir"))

    explanation_type = payload.get("explanation_type", "global").lower().strip()
    if explanation_type not in ("global", "local"):
        raise MLError(f"Unsupported explanation_type '{explanation_type}'. Supported: 'global', 'local'")

    sample_index = int(payload.get("sample_index", 0))
    method = payload.get("method", "shap").lower().strip()

    target_column = DOMAIN_TARGET_MAP[domain]
    train_df = pd.read_parquet(features_dir / "train.parquet")
    val_df = pd.read_parquet(features_dir / "validation.parquet")

    # Fit target model
    if model_name == "random_forest":
        forecaster = RandomForestForecaster(
            n_estimators=int(payload.get("n_estimators", 200)),
            random_state=42,
        )
    elif model_name == "xgboost":
        forecaster = XGBoostForecaster(
            n_estimators=int(payload.get("n_estimators", 200)),
            random_state=42,
        )
    else:
        raise MLError(f"Model '{model_name}' does not support explainability")

    forecaster.fit(train_df, target_column=target_column)
    explainer = ForecastExplainer(forecaster, model_name=model_name)

    try:
        if explanation_type == "global":
            explanation = explainer.explain_global(val_df, method=method)
        else:
            explanation = explainer.explain_local(
                val_df,
                sample_index=sample_index,
                method=method,
                train_df=train_df,
            )
        explanation["domain"] = domain
        return explanation
    except Exception as err:
        raise MLError(f"Explanation generation failed: {err}", code=400) from err
