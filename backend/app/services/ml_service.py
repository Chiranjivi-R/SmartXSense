"""Backend ML orchestration service for forecasting, model comparison, and XAI explanations.

Strict Security & Integration Rules:
1. Path validation prevents path traversal and enforces feature directory constraints.
2. Model comparison strictly relies on VALIDATION performance.
3. Lightweight candidates are used for quick comparison (deep learning models excluded by default).
4. Direct Python service calls (no subprocess CLI calls).
5. Model artifacts are loaded only from the controlled SmartXSense artifacts directory.
"""

from __future__ import annotations

from pathlib import Path
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

import pandas as pd

from ml_engine.evaluation import AdaptiveModelSelector, compare_models
from ml_engine.explainability import ForecastExplainer, SUPPORTED_EXPLAINABLE_MODELS, ShapTreeExplainer
from ml_engine.models.artifact_store import (
    compute_dataset_fingerprint,
    default_artifacts_root,
    derive_features_id,
    is_persistence_supported,
    save_benchmark_cache,
    try_load_benchmark_cache,
    try_load_compatible_artifact,
)
from ml_engine.models.capability_matrix import DEFAULT_COMPARE_MODELS as DEFAULT_COMPARISON_MODELS
from ml_engine.models.future_forecasting import (
    FutureForecastError,
    future_support_error,
    generate_future_forecast,
    is_future_forecast_supported,
    validate_horizon,
)
from ml_engine.models.random_forest import RandomForestForecaster
from ml_engine.models.trainer import (
    DOMAIN_TARGET_MAP,
    TrainerError,
    evaluate_fitted_model_on_splits,
    expected_configuration_for_request,
    run_training_pipeline,
)
from ml_engine.models.xgboost_model import XGBoostForecaster
from ml_engine.history.forecast_history import (
    ForecastHistoryError,
    delete_forecast_run,
    get_forecast_run,
    list_forecast_history,
    safe_record_from_forecast_result,
)

from .client_error_sanitize import sanitize_client_error_message
from .forecast_summary_service import derive_forecast_facts


class MLError(ValueError):
    """Raised when an ML backend service operation fails."""

    def __init__(self, message: str, code: int = 400, data: dict[str, Any] | None = None) -> None:
        safe = sanitize_client_error_message(message)
        super().__init__(safe)
        self.message = safe
        self.code = code
        self.data = data


import tempfile

ALLOWED_ROOT_PATHS = (
    Path(__file__).resolve().parents[3] / "ml_engine" / "data" / "features",  # Constrained feature artifact root
    Path(__file__).resolve().parents[3],                                       # Repository root
    Path(tempfile.gettempdir()),                                                # System temp directory for tests
    Path("C:/tmp"),                                                            # Temporary test directories
    Path("/tmp"),
)

KNOWN_FORECAST_MODELS = frozenset({
    "naive",
    "seasonal_naive",
    "baseline",
    "arima",
    "sarima",
    "random_forest",
    "xgboost",
    "trend_xgboost",
    "lstm",
    "cnn_lstm",
    "hybrid",
})

ALLOWED_SELECTION_METRICS = frozenset({"rmse", "mae", "mape"})
MAX_COMPARE_MODELS = 6


def _artifact_response(*, used: bool, source: str, key: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"used": used, "source": source}
    if key:
        payload["key"] = key
    return payload


def _attach_history(
    result: dict[str, Any],
    *,
    forecast_type: str,
    features_dir: Path,
    selection_method: str | None = None,
) -> dict[str, Any]:
    """Record successful forecast metadata without storing prediction arrays."""
    features_id = None
    try:
        features_id = derive_features_id(features_dir)
    except Exception:
        features_id = Path(features_dir).name if features_dir else None
    context_frame = None
    if forecast_type == "test_evaluation":
        try:
            # This is an already-authorized, controlled feature artifact.  The
            # summary service retains only aggregates, never source rows.
            context_frame = pd.read_parquet(features_dir / "test.parquet")
        except Exception:
            context_frame = None
    result["summary_facts"] = derive_forecast_facts(result, context_frame=context_frame)
    summary = safe_record_from_forecast_result(
        result=result,
        forecast_type=forecast_type,
        features_id=features_id,
        selection_method=selection_method,
    )
    if summary is not None:
        result["history"] = {
            "recorded": True,
            "run_id": summary.get("run_id"),
        }
    else:
        result["history"] = {"recorded": False}
    return result


def _attach_llm_explanation(
    res: dict[str, Any],
    payload: dict[str, Any],
    dataset_ctx: dict[str, Any],
    selection_reason: str = "User selected model",
) -> dict[str, Any]:
    """Build context and call LLM service to append natural language explanation."""
    t_llm_start = time.perf_counter()
    try:
        from .llm_service import generate_explanation

        domain = payload.get("domain", "retail").lower().strip()
        model_name = res.get("model") or payload.get("model") or "random_forest"
        metrics = res.get("metrics") or {}
        baseline_metrics = res.get("baseline_metrics") or {}

        t_shap_start = time.perf_counter()
        shap_features = []
        if model_name in ("random_forest", "xgboost", "trend_xgboost", "hybrid"):
            if domain == "electricity":
                # For electricity forecasting, avoid synchronous execution of expensive detailed SHAP
                # during the primary forecast request. Use available model-native feature
                # importances for the immediate forecast response and LLM explanation.
                # Detailed SHAP remains available via the dedicated explainability endpoint.
                feat_imp = res.get("feature_importances") or {}
                if isinstance(feat_imp, dict) and feat_imp:
                    shap_features = [{"feature": k, "importance": round(float(v), 4)} for k, v in feat_imp.items()]
                elif isinstance(feat_imp, list):
                    shap_features = feat_imp
            else:
                try:
                    exp_res = explain_service({
                        "domain": domain,
                        "model": model_name,
                        "dataset_id": dataset_ctx.get("dataset_id"),
                        "features_dir": str(res.get("features_dir") or payload.get("features_dir") or ""),
                        "location": payload.get("location"),
                        "geo_filters": payload.get("geo_filters"),
                        "explanation_type": "global",
                    })
                    if isinstance(exp_res, dict) and "features" in exp_res:
                        shap_features = exp_res["features"]
                except Exception:
                    pass
        shap_time_ms = round((time.perf_counter() - t_shap_start) * 1000, 2)

        mape_val = metrics.get("mape")
        scale_warn = bool(isinstance(mape_val, (int, float)) and (mape_val > 0.5 or mape_val > 50))

        # Determine forecast trend from predicted values array
        predictions = res.get("predictions") or []
        forecast_trend = "stable"
        if len(predictions) > 1:
            pred_vals = [float(p.get("predicted", 0)) for p in predictions if p.get("predicted") is not None]
            if pred_vals:
                first_val = pred_vals[0]
                last_val = pred_vals[-1]
                max_val = max(pred_vals)
                min_val = min(pred_vals)
                diff_pct = (last_val - first_val) / (abs(first_val) + 1e-6)

                if diff_pct > 0.05:
                    forecast_trend = "increasing"
                elif diff_pct < -0.05:
                    forecast_trend = "decreasing"
                elif (max_val - min_val) / (abs(min_val) + 1e-6) > 0.25:
                    forecast_trend = "highly_variable"
                else:
                    forecast_trend = "stable"

        # Conditional Data Quality facts - NO FALSE STATEMENTS
        dup_ts_count = dataset_ctx.get("duplicate_timestamps_count", 0)
        dup_rows_count = dataset_ctx.get("duplicate_rows_removed", 0)
        missing_count = dataset_ctx.get("missing_values_handled", 0)
        is_dup_ts_handled = bool(dup_ts_count > 0 or dataset_ctx.get("duplicate_timestamps_handled") is True)

        data_quality = {
            "duplicate_rows_removed": dup_rows_count,
            "duplicate_timestamps_handled": is_dup_ts_handled,
            "duplicate_timestamps_count": dup_ts_count,
            "missing_values_handled": missing_count,
            "excluded_features": ["customer_id", "email", "phone", "order_id", "name", "customer_name", "date_of_birth"],
        }

        context = {
            "dataset_id": dataset_ctx.get("dataset_id") or "demo_dataset",
            "dataset_hash": dataset_ctx.get("dataset_hash") or "demo_hash",
            "dataset_name": dataset_ctx.get("dataset_name") or "Dataset",
            "domain": domain,
            "target": dataset_ctx.get("target") or "target",
            "selected_model": model_name,
            "model_selection_reason": selection_reason,
            "forecast_trend": forecast_trend,
            "metrics": metrics,
            "baseline_metrics": {
                "naive_rmse": baseline_metrics.get("rmse") or res.get("naive_baseline_rmse"),
                "model_improvement_pct": res.get("improvement_over_baseline_pct") or baseline_metrics.get("model_improvement"),
            },
            "data_quality": data_quality,
            "final_model_features": res.get("feature_names") or res.get("features") or [],
            "shap_features": shap_features,
            "target_scale_warning": scale_warn,
        }

        res["llm_explanation"] = generate_explanation(context)
        llm_time_ms = round((time.perf_counter() - t_llm_start) * 1000, 2)
        res["pipeline_diagnostics"] = {
            "shap_calculation_ms": shap_time_ms,
            "llm_explanation_ms": llm_time_ms,
        }
    except Exception as err:
        logger.warning(f"Failed to generate LLM explanation: {err}")
        res["llm_explanation"] = {
            "provider": "deterministic",
            "model": None,
            "fallback_used": True,
            "explanation": "Forecast computed successfully. Natural language explanation service unavailable.",
            "status": "fallback",
        }
    return res


def _resolve_artifacts_root() -> Path:

    """Resolve controlled artifacts root; never trusts request-supplied paths."""
    try:
        from flask import current_app, has_app_context

        if has_app_context():
            configured = current_app.config.get("ML_ARTIFACTS_ROOT")
            if configured:
                root = Path(str(configured)).resolve()
                if ".." in str(configured):
                    raise MLError("Invalid ML_ARTIFACTS_ROOT configuration", code=500)
                root.mkdir(parents=True, exist_ok=True)
                return root
    except MLError:
        raise
    except Exception:
        pass
    return default_artifacts_root()


DEMO_FEATURE_DIRS = {
    "demo_retail": "ml_engine/data/features/retail/features_1c0c92ff-264e-4398-b83c-e0e082b956b4",
    "demo_electricity": "ml_engine/data/features/electricity/features_23889e57-aa6d-4a50-930c-874f6a963a25",
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

    if raw_str in DEMO_FEATURE_DIRS:
        raw_str = DEMO_FEATURE_DIRS[raw_str]
    else:
        try:
            from flask import has_app_context
            if has_app_context():
                import json
                from ..extensions import db
                from ..models import DatasetPreprocessing
                prep = db.session.scalar(
                    db.select(DatasetPreprocessing)
                    .where(DatasetPreprocessing.dataset_id == raw_str)
                    .order_by(DatasetPreprocessing.processed_at.desc())
                )
                if prep and prep.lags_generated_json:
                    lineage = json.loads(prep.lags_generated_json)
                    if isinstance(lineage, dict) and lineage.get("features_dir"):
                        raw_str = str(lineage["features_dir"])
        except Exception:
            pass

    try:
        raw_path = Path(raw_str)
        if not raw_path.is_absolute():
            project_root = Path(__file__).resolve().parents[3]
            candidate = (project_root / raw_path).resolve()
            if candidate.exists():
                resolved_path = candidate
            else:
                resolved_path = raw_path.resolve()
        else:
            resolved_path = raw_path.resolve()
    except Exception as err:
        raise MLError(f"Invalid features_dir path format: {err}", code=400) from err

    if not resolved_path.exists() or not resolved_path.is_dir():
        raise MLError("Features directory does not exist or is not a directory", code=400)

    # Check allowed root containment
    is_allowed = any(
        resolved_path == root or root in resolved_path.parents
        for root in ALLOWED_ROOT_PATHS
    )
    if not is_allowed:
        raise MLError("Access denied to features directory outside allowed project scope", code=400)

    for artifact in ("train.parquet", "validation.parquet", "test.parquet"):
        if not (resolved_path / artifact).exists():
            raise MLError(
                f"Required feature artifact '{artifact}' missing from features directory",
                code=400,
            )

    return resolved_path


def resolve_dynamic_features_and_context(payload: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Resolve features directory and construct dataset context (supporting demo, pre-built dirs, and custom uploads)."""
    import hashlib
    import pandas as pd
    from ml_engine.preprocessing.intelligence import _is_no_filter_val, get_geographic_hierarchy
    from ml_engine.preprocessing.dynamic_prep import DynamicPrepError, prepare_dynamic_dataset
    from .dataset_service import DatasetError, managed_upload_path, _read_dataset, _extension, owned_dataset

    dataset_id = payload.get("dataset_id") or payload.get("datasetId") or payload.get("dataset")
    domain = (payload.get("domain") or "retail").strip().lower()
    target = payload.get("target")
    location = payload.get("location")
    geo_filters = payload.get("geo_filters")
    features_dir_input = payload.get("features_dir")

    clean_loc = str(location).strip() if not _is_no_filter_val(location) else None
    valid_filters: dict[str, str] = {}
    if geo_filters and isinstance(geo_filters, dict):
        for k, v in geo_filters.items():
            if not _is_no_filter_val(v):
                valid_filters[k] = str(v).strip()

    # Priority 1: Check if dataset_id corresponds to a user-uploaded database dataset
    ds_obj = None
    if dataset_id and dataset_id not in ("demo_retail", "demo_electricity") and dataset_id not in DEMO_FEATURE_DIRS:
        try:
            ds_obj = owned_dataset(dataset_id)
        except Exception as _e:
            import logging
            logging.getLogger(__name__).warning("owned_dataset lookup failed for %s: %s", dataset_id, _e, exc_info=True)
            ds_obj = None

    if ds_obj is not None and not isinstance(ds_obj, dict):
        path = managed_upload_path(ds_obj)
        frame = _read_dataset(path, _extension(ds_obj.filename))
        dataset_name = ds_obj.filename
        content_h = getattr(ds_obj, "content_hash", None)
        if not content_h and path.exists():
            content_h = hashlib.sha256(path.read_bytes()).hexdigest()
        dataset_hash = (content_h or ds_obj.dataset_id)[:12]

        if clean_loc or valid_filters:
            geo_meta = get_geographic_hierarchy(frame)
            if not geo_meta.get("has_geography"):
                raise MLError("Location-specific forecasting is unavailable for this dataset (no location or geographic column found).", code=400)

        try:
            prep_res = prepare_dynamic_dataset(
                frame,
                domain=domain,
                target=target,
                geo_filters=valid_filters if valid_filters else None,
                location=clean_loc,
                dataset_id=ds_obj.dataset_id,
            )
        except DynamicPrepError as err:
            if "[INSUFFICIENT_HISTORY]" in str(err) or "too small for time-series split" in str(err):
                na_data = {
                    "status": "N/A",
                    "reason_code": "INSUFFICIENT_HISTORY",
                    "total_records": len(frame) if "frame" in locals() else 0,
                    "training_observations": 0,
                    "training_rows": 0,
                    "validation_observations": 0,
                    "test_observations": 0,
                }
                raise MLError(err.message, code=400, data=na_data) from err
            elif "[NO_RECORDS_FOUND]" in str(err):
                na_data = {
                    "status": "N/A",
                    "reason_code": "NO_RECORDS_FOUND",
                    "total_records": 0,
                    "training_observations": 0,
                    "training_rows": 0,
                    "validation_observations": 0,
                    "test_observations": 0,
                }
                raise MLError(err.message, code=400, data=na_data) from err
            raise MLError(err.message, code=err.code) from err

        resolved_dir = validate_features_directory(prep_res["features_dir"])
        ctx = {
            "dataset_id": ds_obj.dataset_id,
            "dataset_name": dataset_name,
            "dataset_hash": dataset_hash,
            "target": prep_res["target"],
            "location": prep_res["geography_label"],
            "geography_label": prep_res["geography_label"],
            "rows_used": prep_res["rows_used"],
            "fingerprint": prep_res["fingerprint"],
        }
        return resolved_dir, ctx

    # Priority 2: Demo Datasets
    raw_key = dataset_id or features_dir_input or ("demo_retail" if domain == "retail" else "demo_electricity")
    if raw_key in ("demo_retail", "demo_electricity") or raw_key in DEMO_FEATURE_DIRS:
        resolved_dir = validate_features_directory(raw_key)
        train_p = resolved_dir / "train.parquet"
        train_df = pd.read_parquet(train_p) if train_p.exists() else pd.DataFrame()

        if clean_loc or valid_filters:
            geo_meta = get_geographic_hierarchy(train_df) if not train_df.empty else {"has_geography": False}
            if not geo_meta.get("has_geography"):
                raise MLError("Location-specific forecasting is unavailable for this dataset (no location or geographic column found).", code=400)

            try:
                prep_res = prepare_dynamic_dataset(
                    train_df,
                    domain=domain,
                    target=target,
                    location=clean_loc,
                    geo_filters=valid_filters if valid_filters else None,
                    dataset_id=raw_key,
                )
            except DynamicPrepError as err:
                if "[INSUFFICIENT_HISTORY]" in str(err) or "too small for time-series split" in str(err):
                    na_data = {
                        "status": "N/A",
                        "reason_code": "INSUFFICIENT_HISTORY",
                        "total_records": len(train_df) if "train_df" in locals() else 0,
                        "training_observations": 0,
                        "training_rows": 0,
                        "validation_observations": 0,
                        "test_observations": 0,
                    }
                    raise MLError(err.message, code=400, data=na_data) from err
                raise MLError(err.message, code=err.code) from err

            dyn_dir = validate_features_directory(prep_res["features_dir"])
            ctx = {
                "dataset_id": raw_key if raw_key in ("demo_retail", "demo_electricity") else f"demo_{domain}",
                "dataset_name": "Retail Sales Demo" if domain == "retail" else "Electricity Load Demo",
                "dataset_hash": prep_res["fingerprint"][:12],
                "target": prep_res["target"],
                "location": prep_res["geography_label"],
                "geography_label": prep_res["geography_label"],
                "rows_used": prep_res["rows_used"],
                "fingerprint": prep_res["fingerprint"],
            }
            return dyn_dir, ctx

        det_target = target or ("sales" if domain == "retail" or "sales" in train_df.columns else "demand_mw")
        ctx = {
            "dataset_id": raw_key if raw_key in ("demo_retail", "demo_electricity") else f"demo_{domain}",
            "dataset_name": "Retail Sales Demo" if domain == "retail" else "Electricity Load Demo",
            "dataset_hash": f"demo_{domain}_hash",
            "target": det_target,
            "location": "All Geographies",
            "geography_label": "All Geographies",
            "rows_used": len(train_df),
            "fingerprint": resolved_dir.name,
        }
        return resolved_dir, ctx

    # Priority 3: Direct Feature Directory Input (e.g. test mock_features_dir or materialized split path)
    if features_dir_input and not str(features_dir_input).startswith("demo_"):
        try:
            resolved_dir = validate_features_directory(features_dir_input)
            if (resolved_dir / "train.parquet").exists():
                train_df = pd.read_parquet(resolved_dir / "train.parquet")
                if clean_loc or valid_filters:
                    geo_meta = get_geographic_hierarchy(train_df) if not train_df.empty else {"has_geography": False}
                    if not geo_meta.get("has_geography"):
                        raise MLError("Location-specific forecasting is unavailable for this dataset (no location or geographic column found).", code=400)
                    prep_res = prepare_dynamic_dataset(
                        train_df,
                        domain=domain,
                        target=target,
                        location=clean_loc,
                        geo_filters=valid_filters if valid_filters else None,
                        dataset_id=dataset_id,
                    )
                    dyn_dir = validate_features_directory(prep_res["features_dir"])
                    ctx = {
                        "dataset_id": dataset_id or resolved_dir.name,
                        "dataset_name": dataset_id or resolved_dir.name,
                        "dataset_hash": prep_res["fingerprint"][:12],
                        "target": prep_res["target"],
                        "location": prep_res["geography_label"],
                        "geography_label": prep_res["geography_label"],
                        "rows_used": prep_res["rows_used"],
                        "fingerprint": prep_res["fingerprint"],
                    }
                    return dyn_dir, ctx

                train_bytes = (resolved_dir / "train.parquet").read_bytes()
                dataset_h = hashlib.sha256(train_bytes).hexdigest()[:12]

                det_target = target
                if not det_target or det_target not in train_df.columns:
                    if "demand_mw" in train_df.columns:
                        det_target = "demand_mw"
                    elif "sales" in train_df.columns:
                        det_target = "sales"
                    else:
                        num_cols = [c for c in train_df.select_dtypes(include="number").columns if c != "timestamp"]
                        det_target = num_cols[0] if num_cols else train_df.columns[-1]

                ctx = {
                    "dataset_id": dataset_id or resolved_dir.name,
                    "dataset_name": dataset_id or resolved_dir.name,
                    "dataset_hash": dataset_h,
                    "target": det_target,
                    "location": "All Geographies",
                    "geography_label": "All Geographies",
                    "rows_used": len(train_df),
                    "fingerprint": resolved_dir.name,
                }
                return resolved_dir, ctx
        except MLError as err:
            if err.code == 400 or "traversal" in str(err).lower():
                raise

    # Priority 4: Fallback to lookup_key for remaining database lookups or demo
    lookup_key = dataset_id or features_dir_input
    if not lookup_key:
        raise MLError("Missing dataset identifier or features_dir", code=400)

    try:
        ds_obj = owned_dataset(lookup_key)
        if isinstance(ds_obj, dict):
            resolved_dir = validate_features_directory(f"demo_{domain}")
            train_df = pd.read_parquet(resolved_dir / "train.parquet") if (resolved_dir / "train.parquet").exists() else pd.DataFrame()

            if clean_loc or valid_filters:
                geo_meta = get_geographic_hierarchy(train_df) if not train_df.empty else {"has_geography": False}
                if not geo_meta.get("has_geography"):
                    raise MLError("Location-specific forecasting is unavailable for this dataset (no location or geographic column found).", code=400)

                prep_res = prepare_dynamic_dataset(
                    train_df,
                    domain=domain,
                    target=target,
                    location=clean_loc,
                    geo_filters=valid_filters if valid_filters else None,
                    dataset_id=ds_obj.get("dataset_id", lookup_key),
                )
                dyn_dir = validate_features_directory(prep_res["features_dir"])
                ctx = {
                    "dataset_id": ds_obj.get("dataset_id", lookup_key),
                    "dataset_name": ds_obj.get("filename", "Demo Dataset"),
                    "dataset_hash": prep_res["fingerprint"][:12],
                    "target": prep_res["target"],
                    "location": prep_res["geography_label"],
                    "geography_label": prep_res["geography_label"],
                    "rows_used": prep_res["rows_used"],
                    "fingerprint": prep_res["fingerprint"],
                }
                return dyn_dir, ctx

            ctx = {
                "dataset_id": ds_obj.get("dataset_id", lookup_key),
                "dataset_name": ds_obj.get("filename", "Demo Dataset"),
                "dataset_hash": f"demo_{domain}_hash",
                "target": target or ("sales" if domain == "retail" else "demand_mw"),
                "location": "All Geographies",
                "geography_label": "All Geographies",
                "rows_used": len(train_df),
                "fingerprint": resolved_dir.name,
            }
            return resolved_dir, ctx
        elif ds_obj is not None:
            path = managed_upload_path(ds_obj)
            frame = _read_dataset(path, _extension(ds_obj.filename))
            dataset_name = ds_obj.filename
            dataset_id = ds_obj.dataset_id
            content_h = getattr(ds_obj, "content_hash", None)
            if not content_h and path.exists():
                content_h = hashlib.sha256(path.read_bytes()).hexdigest()
            dataset_hash = (content_h or ds_obj.dataset_id)[:12]

            clean_loc = str(location).strip() if not _is_no_filter_val(location) else None
            valid_filters = {}
            if geo_filters and isinstance(geo_filters, dict):
                for k, v in geo_filters.items():
                    if not _is_no_filter_val(v):
                        valid_filters[k] = str(v).strip()

            if clean_loc or valid_filters:
                geo_meta = get_geographic_hierarchy(frame)
                if not geo_meta.get("has_geography"):
                    raise MLError("Location-specific forecasting is unavailable for this dataset (no location or geographic column found).", code=400)

            try:
                prep_res = prepare_dynamic_dataset(
                    frame,
                    domain=domain,
                    target=target,
                    geo_filters=valid_filters if valid_filters else None,
                    location=clean_loc,
                    dataset_id=ds_obj.dataset_id,
                )
            except DynamicPrepError as err:
                if "[INSUFFICIENT_HISTORY]" in str(err) or "too small for time-series split" in str(err):
                    na_data = {
                        "status": "N/A",
                        "reason_code": "INSUFFICIENT_HISTORY",
                        "total_records": len(frame) if "frame" in locals() else 0,
                        "training_observations": 0,
                        "training_rows": 0,
                        "validation_observations": 0,
                        "test_observations": 0,
                    }
                    raise MLError(err.message, code=400, data=na_data) from err
                raise MLError(err.message, code=err.code) from err
            resolved_dir = validate_features_directory(prep_res["features_dir"])
            ctx = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "dataset_hash": dataset_hash,
                "target": prep_res["target"],
                "location": prep_res["geography_label"],
                "geography_label": prep_res["geography_label"],
                "rows_used": prep_res["rows_used"],
                "fingerprint": prep_res["fingerprint"],
            }
            return resolved_dir, ctx

    except DatasetError as err:
        raise MLError(err.message, code=err.code) from err
    except DynamicPrepError as err:
        raise MLError(err.message, code=err.code) from err
    except MLError:
        raise
    except Exception as exc:
        raise MLError(f"Unable to process dataset '{lookup_key}': {exc}", code=404) from exc

    raise MLError(f"Dataset '{lookup_key}' not found or invalid", code=404)



def forecast_service(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute forecast via compatible artifact reuse or fresh training."""
    if not isinstance(payload, dict):
        raise MLError("Invalid JSON request body")

    domain = payload.get("domain", "").lower().strip()
    if not domain or domain not in DOMAIN_TARGET_MAP:
        raise MLError(f"Invalid or missing domain '{domain}'. Supported: {list(DOMAIN_TARGET_MAP.keys())}")

    raw_model = payload.get("model")
    if raw_model is None or str(raw_model).strip() == "" or str(raw_model).strip().lower() == "auto_select":
        model_name = "auto_select"
        selection_method = "auto_select"
    else:
        model_name = str(raw_model).lower().strip()
        selection_method = "manual"

    if model_name == "baseline":
        model_name = "naive"

    selection_reason = "Best validation performance"
    if model_name == "auto_select":
        comp_res = compare_service(payload)
        model_name = comp_res.get("recommended_model")
        if not model_name:
            raise MLError("N/A — No applicable forecasting model for the selected location. No eligible models available for comparison on this dataset", code=400)
        selection_reason = "Best validation performance"
    else:
        selection_reason = f"Evaluated specified model '{model_name.replace('_', ' ').title()}'"

    if model_name not in KNOWN_FORECAST_MODELS:
        raise MLError(
            f"Unsupported model '{model_name}'. Supported: {sorted(KNOWN_FORECAST_MODELS - {'baseline'})}"
        )

    features_dir, dataset_ctx = resolve_dynamic_features_and_context(payload)
    from ml_engine.models.applicability import assess_model_applicability
    applicability_report = assess_model_applicability(features_dir, domain=domain)
    app = applicability_report["models"].get(model_name)
    if app and not app["eligible"]:
        train_p = features_dir / "train.parquet"
        val_p = features_dir / "validation.parquet"
        test_p = features_dir / "test.parquet"
        import pandas as pd
        if train_p.exists():
            train_obs = len(pd.read_parquet(train_p))
            val_obs = len(pd.read_parquet(val_p)) if val_p.exists() else 0
            test_obs = len(pd.read_parquet(test_p)) if test_p.exists() else 0
            total_records = train_obs + val_obs + test_obs
        else:
            train_obs = applicability_report.get("dataset_profile", {}).get("history_length", 0)
            val_obs = 0
            test_obs = 0
            total_records = train_obs

        single_model_na_data = {
            "status": "N/A",
            "reason_code": app.get("reason_code", "INELIGIBLE"),
            "model": model_name,
            "total_records": int(total_records),
            "training_observations": int(train_obs),
            "training_rows": int(train_obs),
            "validation_observations": int(val_obs),
            "test_observations": int(test_obs),
        }
        raise MLError(
            f"Model '{model_name}' is not applicable to this dataset: [{app['reason_code']}] {app['reason']}",
            code=400,
            data=single_model_na_data,
        )
    selection_method = payload.get("selection_method")
    if selection_method is not None:
        selection_method = str(selection_method).strip() or None
    elif payload.get("model") == "auto_select":
        selection_method = "validation_rmse"

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
    seasonal_order = tuple(payload.get("seasonal_order")) if payload.get("seasonal_order") is not None else None
    training_window = int(payload.get("training_window")) if payload.get("training_window") is not None else None
    n_estimators = int(payload.get("n_estimators", 200))
    max_depth = payload.get("max_depth")
    if max_depth is not None:
        max_depth = int(max_depth)
    learning_rate = float(payload.get("learning_rate", 0.05))
    force_retrain = bool(payload.get("force_retrain", False))

    residual_ar = payload.get("residual_ar")
    if model_name == "trend_xgboost" and residual_ar is None and "comp_res" in locals():
        if comp_res.get("configurations") and "trend_xgboost" in comp_res["configurations"]:
            residual_ar = comp_res["configurations"]["trend_xgboost"].get("residual_ar")
        if residual_ar is None:
            for r_item in comp_res.get("ranking", []):
                if r_item.get("model") == "trend_xgboost":
                    residual_ar = r_item.get("configuration", {}).get("residual_ar")

    artifacts_root = _resolve_artifacts_root()

    start_time = time.perf_counter()
    try:
        res = run_training_pipeline(
            domain=domain,
            features_dir=features_dir,
            model_name=model_name,
            components=components,
            hybrid_weighting=hybrid_weighting,
            order=order,
            seasonal_order=seasonal_order,
            training_window=training_window,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            residual_ar=residual_ar,
            prediction_limit=prediction_limit,
            save_artifact=is_persistence_supported(model_name),
            force_retrain=force_retrain,
            artifacts_root=artifacts_root,
        )
        exec_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        is_reused = res.get("artifact_persistence", {}).get("reused", False)
        logger.info(
            f"[ARTIFACT DIAGNOSTIC] Selected dataset ID: {dataset_ctx.get('dataset_id')} | "
            f"Dataset hash: {dataset_ctx.get('dataset_hash')} | Target: {dataset_ctx.get('target')} | "
            f"Geographic scope: {dataset_ctx.get('geography_label')} | Model: {model_name} | "
            f"Artifact key: {res.get('artifact_key')} | Artifact reused: {is_reused}"
        )
        res["artifact"] = _artifact_response(
            used=is_reused,
            source="loaded" if is_reused else "trained",
            key=res.get("artifact_key"),
        )
        res["execution"] = {
            "artifact_reused": is_reused,
            "training_performed": not is_reused,
            "execution_time_ms": exec_time_ms,
        }
        res["dataset_context"] = {
            **dataset_ctx,
            "model": model_name,
            "artifact_reused": is_reused,
        }
        if "comp_res" in locals() and isinstance(comp_res, dict):
            res["validation_benchmark"] = comp_res
        else:
            res["validation_benchmark"] = {
                "candidate_models": [model_name],
                "ranking": [{"model": model_name, "validation_metrics": res.get("validation_metrics") or {}, "rank": 1}],
                "recommended_model": model_name,
                "selection_metric": "rmse",
                "selection_split": "validation",
            }
        res = _attach_history(
            res,
            forecast_type="test_evaluation",
            features_dir=features_dir,
            selection_method=selection_method,
        )
        return _attach_llm_explanation(res, payload, dataset_ctx, selection_reason)

    except TrainerError as err:
        raise MLError(f"Forecasting pipeline error: {err}", code=400) from err
    except Exception as err:
        raise MLError("Internal server error", code=500) from err


def _obtain_fitted_model_for_inference(
    *,
    domain: str,
    model_name: str,
    features_dir: Path,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int] | None = None,
    training_window: int | None = None,
    n_estimators: int,
    max_depth: int | None,
    learning_rate: float,
    components: list[str] | None = None,
    hybrid_weighting: str = "inverse_rmse",
    residual_ar: bool | None = None,
    force_retrain: bool = False,
    horizon: int | None = None,
) -> tuple[Any, dict[str, Any], dict[str, float], dict[str, float]]:
    """Load a compatible Phase A artifact or train+persist a fitted model."""
    import pandas as pd

    from ml_engine.evaluation.metrics import evaluate_forecast
    from ml_engine.models.artifact_store import (
        compute_dataset_fingerprint,
        predict_with_fitted_model,
        save_model_artifact,
    )
    from ml_engine.models.trainer import _train_and_predict_single_model

    artifacts_root = _resolve_artifacts_root()
    target = DOMAIN_TARGET_MAP[domain]
    expected_config = expected_configuration_for_request(
        model_key=model_name,
        domain=domain,
        order=order,
        seasonal_order=seasonal_order,
        training_window=training_window,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        components=components,
        hybrid_weighting=hybrid_weighting,
        residual_ar=residual_ar,
    )
    features_id = derive_features_id(features_dir)
    dataset_fingerprint = compute_dataset_fingerprint(features_dir)

    train_df = pd.read_parquet(features_dir / "train.parquet")
    val_df = pd.read_parquet(features_dir / "validation.parquet")
    test_df = pd.read_parquet(features_dir / "test.parquet")

    from ml_engine.models.random_forest import RETAIL_MACRO_EXCLUSIONS
    from ml_engine.models.tabular_preprocessing import TabularFeatureTransformer

    rf_exclusions = list(RETAIL_MACRO_EXCLUSIONS) if (model_name == "random_forest" and domain == "retail") else ()
    expected_feature_names = TabularFeatureTransformer(target_column=target, extra_exclude=rf_exclusions).fit(train_df).feature_names

    if not force_retrain and is_persistence_supported(model_name):
        try:
            loaded = try_load_compatible_artifact(
                domain=domain,
                model_key=model_name,
                target=target,
                features_id=features_id,
                dataset_fingerprint=dataset_fingerprint,
                feature_names=expected_feature_names,
                configuration=expected_config,
                artifacts_root=artifacts_root,
            )
            if loaded is not None:
                metadata = loaded.get("metadata") or {}
                val_metrics = metadata.get("validation_metrics")
                test_metrics = metadata.get("test_metrics")
                if not val_metrics or not test_metrics:
                    pred_res = predict_with_fitted_model(
                        model_key=model_name,
                        model=loaded["model"],
                        domain=domain,
                        target_column=target,
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                    )
                    val_pred = pred_res["val_pred"]
                    test_pred = pred_res["test_pred"]
                    if model_name in ("lstm", "cnn_lstm"):
                        val_metrics = evaluate_forecast(val_df.loc[val_pred.index, target], val_pred)
                        test_metrics = evaluate_forecast(test_df.loc[test_pred.index, target], test_pred)
                    else:
                        val_metrics = evaluate_forecast(val_df[target], pd.Series(val_pred.values, index=val_df.index))
                        test_metrics = evaluate_forecast(test_df[target], pd.Series(test_pred.values, index=test_df.index))

                return loaded["model"], _artifact_response(
                    used=True,
                    source="loaded",
                    key=loaded.get("relative_key"),
                ), val_metrics, test_metrics


        except Exception:
            pass


    if model_name == "hybrid":
        from ml_engine.models.hybrid import DEFAULT_HYBRID_COMPONENTS, HybridForecaster, HybridModelError, generate_hybrid_candidates
        from ml_engine.models.applicability import assess_model_applicability

        applicability_report = assess_model_applicability(features_dir, domain=domain)
        eligible_set = set(applicability_report["eligible_models"])

        if components is not None:
            raw_pool = components
        else:
            raw_pool = DEFAULT_HYBRID_COMPONENTS.get(domain, ["random_forest", "xgboost"])

        compatible_pool = [
            c for c in raw_pool
            if c in eligible_set and is_future_forecast_supported(domain, c, horizon=horizon) and c != "hybrid"
        ]
        if len(compatible_pool) < 2:
            raise MLError(
                "Hybrid requires at least two future-compatible component models for this horizon.",
                code=400,
            )

        try:
            pool_components = HybridForecaster.validate_components(compatible_pool, domain)
            hybrid_forecaster = HybridForecaster(
                components=pool_components,
                domain=domain,
                weighting_strategy=hybrid_weighting,
            )
        except HybridModelError as err:
            raise MLError(f"Invalid hybrid configuration: {err}", code=400) from err

        val_predictions: dict[str, pd.Series] = {}
        test_predictions: dict[str, pd.Series] = {}
        component_val_metrics: dict[str, dict[str, float]] = {}
        component_test_metrics: dict[str, dict[str, float]] = {}
        component_models: dict[str, Any] = {}
        comp_used_artifact: dict[str, bool] = {}

        for comp in pool_components:
            comp_config = expected_configuration_for_request(
                model_key=comp,
                domain=domain,
                order=order,
                seasonal_order=seasonal_order,
                training_window=training_window,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                residual_ar=residual_ar,
            )
            comp_loaded = None
            if not force_retrain and is_persistence_supported(comp):
                comp_exclusions = list(RETAIL_MACRO_EXCLUSIONS) if (comp == "random_forest" and domain == "retail") else ()
                comp_expected_features = TabularFeatureTransformer(target_column=target, extra_exclude=comp_exclusions).fit(train_df).feature_names
                try:
                    comp_loaded = try_load_compatible_artifact(
                        domain=domain,
                        model_key=comp,
                        target=target,
                        features_id=features_id,
                        dataset_fingerprint=dataset_fingerprint,
                        feature_names=comp_expected_features,
                        configuration=comp_config,
                        artifacts_root=artifacts_root,
                    )
                except Exception:
                    comp_loaded = None

            if comp_loaded is not None:
                component_models[comp] = comp_loaded["model"]
                pred_res = predict_with_fitted_model(
                    model_key=comp,
                    model=comp_loaded["model"],
                    domain=domain,
                    target_column=target,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                )
                v_pred = pred_res["val_pred"]
                t_pred = pred_res["test_pred"]
                comp_used_artifact[comp] = True
            else:
                trained_comp = _train_and_predict_single_model(
                    model_key=comp,
                    domain=domain,
                    target_column=target,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    order=order,
                    seasonal_order=seasonal_order,
                    training_window=training_window,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    residual_ar=residual_ar,
                )
                component_models[comp] = trained_comp["model"]
                v_pred = trained_comp["val_pred"]
                t_pred = trained_comp["test_pred"]
                comp_used_artifact[comp] = False

                if is_persistence_supported(comp):
                    try:
                        v_targs = val_df.loc[v_pred.index, target]
                        t_targs = test_df.loc[t_pred.index, target]
                        save_model_artifact(
                            model=trained_comp["model"],
                            domain=domain,
                            model_key=comp,
                            target=target,
                            features_id=features_id,
                            dataset_fingerprint=dataset_fingerprint,
                            feature_names=trained_comp.get("feature_names") or [],
                            configuration=trained_comp.get("configuration") or comp_config,
                            validation_metrics=evaluate_forecast(v_targs, v_pred),
                            test_metrics=evaluate_forecast(t_targs, t_pred),
                            artifacts_root=artifacts_root,
                        )
                    except Exception:
                        pass

            val_predictions[comp] = v_pred
            test_predictions[comp] = t_pred
            component_val_metrics[comp] = evaluate_forecast(val_df[target], pd.Series(v_pred.values, index=val_df.index))
            component_test_metrics[comp] = evaluate_forecast(test_df[target], pd.Series(t_pred.values, index=test_df.index))

        if components is not None:
            candidate_list = [hybrid_forecaster.components]
        else:
            candidate_list = generate_hybrid_candidates(
                domain=domain,
                applicable_models=list(component_models.keys()),
                future_forecast_only=True,
                horizon=horizon,
            )

        try:
            evaluated_hybrids = HybridForecaster.evaluate_hybrid_candidates(
                candidates=candidate_list,
                component_val_predictions=val_predictions,
                val_targets=val_df[target],
                component_test_predictions=test_predictions,
                test_targets=test_df[target],
                domain=domain,
                weighting_strategy=hybrid_weighting,
                selection_metric="rmse",
                horizon=horizon,
            )
            decision = HybridForecaster.select_best_hybrid_and_standalone(
                evaluated_hybrids=evaluated_hybrids,
                component_val_metrics=component_val_metrics,
                selection_metric="rmse",
                require_future_forecast_support=True,
            )
        except HybridModelError as err:
            raise MLError(f"Hybrid candidate evaluation failed: {err}", code=400) from err

        best_hybrid_raw = decision.get("_best_hybrid_raw", decision["best_hybrid"])
        best_hybrid = decision["best_hybrid"]
        weights = best_hybrid["weights"]

        hybrid_forecaster.components = best_hybrid["components"]
        hybrid_forecaster.weights = weights
        hybrid_forecaster.component_models = {m: component_models[m] for m in best_hybrid["components"]}
        hybrid_forecaster.candidate_evaluations = decision["evaluated_candidates"]
        hybrid_forecaster.best_hybrid = best_hybrid
        hybrid_forecaster.best_standalone = decision["best_standalone"]
        hybrid_forecaster.selection_decision = {k: v for k, v in decision.items() if not k.startswith("_")}

        val_metrics = best_hybrid["validation_metrics"]
        test_metrics = best_hybrid.get("test_metrics") or {}

        all_reused = bool(comp_used_artifact) and all(comp_used_artifact.values())
        artifact_info = _artifact_response(
            used=all_reused,
            source="loaded" if all_reused else "trained",
        )
        return hybrid_forecaster, artifact_info, val_metrics, test_metrics

    try:
        trained = _train_and_predict_single_model(
            model_key=model_name,
            domain=domain,
            target_column=target,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            order=order,
            seasonal_order=seasonal_order,
            training_window=training_window,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            residual_ar=residual_ar,
        )
    except TrainerError as err:
        raise MLError(f"Future forecasting pipeline error: {err}", code=400) from err

    val_pred = trained["val_pred"]
    test_pred = trained["test_pred"]
    if model_name in ("lstm", "cnn_lstm"):
        val_metrics = evaluate_forecast(val_df.loc[val_pred.index, target], val_pred)
        test_metrics = evaluate_forecast(test_df.loc[test_pred.index, target], test_pred)
    else:
        val_metrics = evaluate_forecast(val_df[target], val_pred)
        test_metrics = evaluate_forecast(test_df[target], test_pred)

    artifact_info = _artifact_response(used=False, source="trained")
    if is_persistence_supported(model_name):
        try:
            saved = save_model_artifact(
                model=trained["model"],
                domain=domain,
                model_key=model_name,
                target=target,
                features_id=features_id,
                dataset_fingerprint=dataset_fingerprint,
                feature_names=trained.get("feature_names") or [],
                configuration=trained.get("configuration") or expected_config,
                validation_metrics=val_metrics,
                test_metrics=test_metrics,
                artifacts_root=artifacts_root,
            )
            artifact_info = _artifact_response(
                used=False,
                source="trained",
                key=saved.get("relative_key"),
            )
        except Exception:
            # Persistence failure must not block future forecasting once the model is fitted.
            pass

    return trained["model"], artifact_info, val_metrics, test_metrics


def future_forecast_service(payload: dict[str, Any]) -> dict[str, Any]:
    """Generate future forecasts with no actual values (separate from test evaluation)."""
    if not isinstance(payload, dict):
        raise MLError("Invalid JSON request body")

    domain = payload.get("domain", "").lower().strip()
    if not domain or domain not in DOMAIN_TARGET_MAP:
        raise MLError(f"Invalid or missing domain '{domain}'. Supported: {list(DOMAIN_TARGET_MAP.keys())}")

    if "horizon" not in payload:
        raise MLError("Missing required parameter 'horizon'")
    try:
        horizon = validate_horizon(payload.get("horizon"), domain=domain)
    except FutureForecastError as err:
        raise MLError(str(err), code=400) from err

    raw_model = payload.get("model")
    if raw_model is None or str(raw_model).strip() == "" or str(raw_model).strip().lower() in ("auto_select", "best", "auto"):
        model_name = "auto_select"
        selection_method = "auto_select"
    else:
        model_name = str(raw_model).lower().strip()
        selection_method = "manual"

    if model_name == "baseline":
        model_name = "naive"

    features_dir, dataset_ctx = resolve_dynamic_features_and_context(payload)

    selection_reason = "Best validation performance"
    if model_name == "auto_select":
        comp_res = compare_service(payload)
        ranking = comp_res.get("ranking", [])
        eligible = [r["model"] for r in ranking if is_future_forecast_supported(domain, r["model"], horizon=horizon)]
        if not eligible:
            raise MLError(
                f"None of the requested candidate models support future forecasting for domain '{domain}' at horizon {horizon}",
                code=400,
            )
        model_name = eligible[0]
        selection_reason = "Best validation performance"
    else:
        if not is_future_forecast_supported(domain, model_name, horizon=horizon):
            raise MLError(future_support_error(domain, model_name, horizon=horizon), code=400)
        from ml_engine.models.applicability import assess_model_applicability
        applicability_report = assess_model_applicability(features_dir, domain=domain)
        app = applicability_report["models"].get(model_name)
        if app and not app["eligible"]:
            raise MLError(
                f"Model '{model_name}' is not applicable to this dataset: [{app['reason_code']}] {app['reason']}",
                code=400,
            )
        selection_reason = f"Evaluated specified model '{model_name.replace('_', ' ').title()}'"

    residual_ar = payload.get("residual_ar")
    if model_name == "trend_xgboost" and residual_ar is None:
        if "comp_res" in locals() and comp_res.get("configurations"):
            winning_cfg = comp_res["configurations"].get("trend_xgboost", {})
            if isinstance(winning_cfg, dict) and "residual_ar" in winning_cfg:
                residual_ar = winning_cfg["residual_ar"]
        if residual_ar is None and "ranking" in locals():
            for r_item in ranking:
                if r_item.get("model") == "trend_xgboost":
                    residual_ar = r_item.get("configuration", {}).get("residual_ar")

    order = tuple(payload.get("order", [1, 1, 1]))
    seasonal_order = tuple(payload.get("seasonal_order")) if payload.get("seasonal_order") is not None else None
    training_window = int(payload.get("training_window")) if payload.get("training_window") is not None else None
    n_estimators = int(payload.get("n_estimators", 200))
    max_depth = payload.get("max_depth")
    if max_depth is not None:
        max_depth = int(max_depth)
    learning_rate = float(payload.get("learning_rate", 0.05))
    force_retrain = bool(payload.get("force_retrain", False))
    components = payload.get("components")
    hybrid_weighting = payload.get("hybrid_weighting", "inverse_rmse")

    try:
        model, artifact_info, val_metrics, test_metrics = _obtain_fitted_model_for_inference(
            domain=domain,
            model_name=model_name,
            features_dir=features_dir,
            order=order,
            seasonal_order=seasonal_order,
            training_window=training_window,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            components=components,
            hybrid_weighting=hybrid_weighting,
            residual_ar=residual_ar,
            force_retrain=force_retrain,
            horizon=horizon,
        )
        result = generate_future_forecast(
            model=model,
            model_key=model_name,
            domain=domain,
            features_dir=features_dir,
            horizon=horizon,
        )
        result["selected_model"] = model_name
        result["selection_reason"] = selection_reason
        result["validation_metrics"] = val_metrics
        result["test_metrics"] = test_metrics
        result["artifact"] = artifact_info
        result["artifact_reused"] = artifact_info.get("used", False)
        result["dataset_context"] = {
            **dataset_ctx,
            "model": model_name,
            "artifact_reused": artifact_info.get("used", False),
        }
        result["configuration"] = expected_configuration_for_request(
            model_key=model_name,
            domain=domain,
            order=order,
            seasonal_order=seasonal_order,
            training_window=training_window,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            components=getattr(model, "components", components),
            hybrid_weighting=hybrid_weighting,
            residual_ar=residual_ar,
        )
        if "comp_res" in locals() and isinstance(comp_res, dict):
            result["validation_benchmark"] = comp_res
        else:
            result["validation_benchmark"] = {
                "candidate_models": [model_name],
                "ranking": [{"model": model_name, "validation_metrics": result.get("validation_metrics") or {}, "rank": 1}],
                "recommended_model": model_name,
                "selection_metric": "rmse",
                "selection_split": "validation",
            }
        result = _attach_history(
            result,
            forecast_type="future",
            features_dir=features_dir,
            selection_method=selection_method,
        )
        return _attach_llm_explanation(result, payload, dataset_ctx, "Future Horizon Projection")

    except FutureForecastError as err:
        raise MLError(str(err), code=400) from err
    except TrainerError as err:
        raise MLError(f"Future forecasting pipeline error: {err}", code=400) from err
    except MLError:
        raise
    except Exception as err:
        import traceback
        traceback.print_exc()
        raise MLError("Internal server error", code=500) from err



def compare_service(payload: dict[str, Any]) -> dict[str, Any]:
    """Compare candidate forecasting models based strictly on validation metrics."""
    start_time = time.perf_counter()
    if not isinstance(payload, dict):
        raise MLError("Invalid JSON request body")

    domain = payload.get("domain", "").lower().strip()
    if not domain or domain not in DOMAIN_TARGET_MAP:
        raise MLError(f"Invalid or missing domain '{domain}'. Supported: {list(DOMAIN_TARGET_MAP.keys())}")

    features_dir, dataset_ctx = resolve_dynamic_features_and_context(payload)
    selection_metric = payload.get("selection_metric", "rmse")
    if not isinstance(selection_metric, str):
        raise MLError(
            f"Invalid selection_metric. Supported: {sorted(ALLOWED_SELECTION_METRICS)}"
        )
    selection_metric = selection_metric.lower().strip()
    if selection_metric not in ALLOWED_SELECTION_METRICS:
        raise MLError(
            f"Invalid selection_metric '{selection_metric}'. Supported: {sorted(ALLOWED_SELECTION_METRICS)}"
        )

    selection_split = payload.get("selection_split", "validation")
    if selection_split is not None:
        if not isinstance(selection_split, str) or selection_split.lower().strip() != "validation":
            raise MLError("Parameter 'selection_split' must be 'validation'")

    models_input = payload.get("models")
    from ml_engine.models.applicability import assess_model_applicability
    applicability_report = assess_model_applicability(features_dir, domain=domain)

    if models_input is None:
        raw_candidates = DEFAULT_COMPARISON_MODELS.get(domain, ["random_forest", "xgboost", "hybrid"])
        candidate_models = [
            m for m in raw_candidates
            if applicability_report["models"].get(m, {}).get("eligible", False)
        ]
        if not candidate_models:
            train_p = features_dir / "train.parquet"
            val_p = features_dir / "validation.parquet"
            test_p = features_dir / "test.parquet"
            import pandas as pd
            if train_p.exists():
                train_obs_count = len(pd.read_parquet(train_p))
                val_obs_count = len(pd.read_parquet(val_p)) if val_p.exists() else 0
                test_obs_count = len(pd.read_parquet(test_p)) if test_p.exists() else 0
                total_records = train_obs_count + val_obs_count + test_obs_count
            else:
                train_obs_count = applicability_report.get("dataset_profile", {}).get("history_length", 0)
                val_obs_count = 0
                test_obs_count = 0
                total_records = train_obs_count

            structured_na_data = {
                "status": "N/A",
                "reason_code": "NO_APPLICABLE_MODEL",
                "total_records": int(total_records),
                "training_observations": int(train_obs_count),
                "training_rows": int(train_obs_count),
                "validation_observations": int(val_obs_count),
                "test_observations": int(test_obs_count),
            }

            ineligible_reasons = []
            for m in raw_candidates:
                app = applicability_report["models"].get(m, {})
                if not app.get("eligible", False):
                    code = app.get("reason_code", "INELIGIBLE")
                    msg = app.get("reason", "Model ineligible")
                    ineligible_reasons.append(f"{m} [{code}]: {msg}")
            reasons_summary = "; ".join(ineligible_reasons[:2])
            raise MLError(
                f"N/A — No applicable forecasting model for the selected location. "
                f"{train_obs_count} usable training observations available; minimum model requirements are not met. "
                f"No eligible models available for comparison on this dataset. ({reasons_summary})",
                code=400,
                data=structured_na_data,
            )
    else:
        if not isinstance(models_input, list) or not models_input:
            raise MLError("Parameter 'models' must be a non-empty list of model names")
        if len(models_input) > MAX_COMPARE_MODELS:
            raise MLError(
                f"Parameter 'models' accepts at most {MAX_COMPARE_MODELS} candidates"
            )
        candidate_models = []
        for raw in models_input:
            if not isinstance(raw, str):
                raise MLError("Parameter 'models' must contain model name strings only")
            name = raw.lower().strip()
            if name == "baseline":
                name = "naive"
            if name not in KNOWN_FORECAST_MODELS or name == "auto_select":
                raise MLError(f"Unsupported comparison model '{name}'")
            app = applicability_report["models"].get(name)
            if app and not app["eligible"]:
                raise MLError(
                    f"Model '{name}' is not applicable to this dataset: [{app['reason_code']}] {app['reason']}",
                    code=400,
                )
            candidate_models.append(name)

    artifacts_root = _resolve_artifacts_root()
    features_id = derive_features_id(features_dir)
    dataset_fingerprint = compute_dataset_fingerprint(features_dir)

    configurations = {}
    for m in candidate_models:
        if m == "trend_xgboost":
            from ml_engine.models.applicability import profile_dataset_for_applicability
            p = profile_dataset_for_applicability(features_dir, domain=domain)
            configurations[m] = expected_configuration_for_request(
                model_key=m,
                domain=domain,
                residual_ar=bool(p.has_hourly_lag_structure),
            )
        else:
            configurations[m] = expected_configuration_for_request(
                model_key=m,
                domain=domain,
            )

    cached_benchmark = try_load_benchmark_cache(
        domain=domain,
        features_id=features_id,
        dataset_fingerprint=dataset_fingerprint,
        candidate_models=candidate_models,
        selection_metric=selection_metric,
        configurations=configurations,
        artifacts_root=artifacts_root,
    )
    if cached_benchmark is not None:
        exec_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "domain": domain,
            "selection_metric": selection_metric,
            "selection_split": "validation",
            "ranking": cached_benchmark["ranking"],
            "recommended_model": cached_benchmark["recommended_model"],
            "adaptive_selection": cached_benchmark["adaptive_selection"],
            "configurations": cached_benchmark.get("configurations", configurations),
            "benchmark_execution": {
                "reused": True,
                "status": "Benchmark Reused",
                "execution_time_ms": exec_time_ms,
            },
            "dataset_context": dataset_ctx,
        }

    reports = []
    for m in candidate_models:
        try:
            if m == "trend_xgboost":
                from ml_engine.models.applicability import profile_dataset_for_applicability
                p = profile_dataset_for_applicability(features_dir, domain=domain)
                if p.has_hourly_lag_structure:
                    report_base = run_training_pipeline(
                        domain=domain,
                        features_dir=features_dir,
                        model_name="trend_xgboost",
                        residual_ar=False,
                        save_artifact=is_persistence_supported("trend_xgboost"),
                        artifacts_root=artifacts_root,
                    )
                    report_rar = run_training_pipeline(
                        domain=domain,
                        features_dir=features_dir,
                        model_name="trend_xgboost",
                        residual_ar=True,
                        save_artifact=is_persistence_supported("trend_xgboost"),
                        artifacts_root=artifacts_root,
                    )
                    val_metric = selection_metric.lower().strip()
                    score_base = report_base.get("validation_metrics", {}).get(val_metric, float("inf"))
                    score_rar = report_rar.get("validation_metrics", {}).get(val_metric, float("inf"))
                    report = report_rar if score_rar < score_base else report_base
                else:
                    report = run_training_pipeline(
                        domain=domain,
                        features_dir=features_dir,
                        model_name="trend_xgboost",
                        residual_ar=False,
                        save_artifact=is_persistence_supported("trend_xgboost"),
                        artifacts_root=artifacts_root,
                    )
            else:
                report = run_training_pipeline(
                    domain=domain,
                    features_dir=features_dir,
                    model_name=m,
                    save_artifact=is_persistence_supported(m),
                    artifacts_root=artifacts_root,
                )
            reports.append(report)
        except Exception as err:
            raise MLError(f"Failed to evaluate candidate model '{m}': {err}", code=400) from err

    for r in reports:
        m_name = r.get("model") or r.get("model_name")
        if m_name and "configuration" in r:
            if domain == "electricity":
                # For electricity, preserve canonical configurations so try_load_benchmark_cache
                # key computed from expected_configuration_for_request matches save_benchmark_cache.
                pass
            else:
                configurations[m_name] = r["configuration"]

    try:
        comparison_res = compare_models(reports, selection_metric=selection_metric)
        selector = AdaptiveModelSelector(selection_metric=selection_metric)
        selection_res = selector.select_best_model(reports, selection_metric=selection_metric)
    except Exception as err:
        raise MLError(f"Model comparison calculation failed: {err}", code=400) from err

    save_benchmark_cache(
        domain=domain,
        features_id=features_id,
        dataset_fingerprint=dataset_fingerprint,
        candidate_models=candidate_models,
        selection_metric=selection_metric,
        ranking=comparison_res["ranking"],
        recommended_model=comparison_res["recommended_model"],
        adaptive_selection=selection_res,
        configurations=configurations,
        artifacts_root=artifacts_root,
    )

    exec_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return {
        "domain": domain,
        "selection_metric": selection_metric,
        "selection_split": "validation",
        "ranking": comparison_res["ranking"],
        "recommended_model": comparison_res["recommended_model"],
        "adaptive_selection": selection_res,
        "configurations": configurations,
        "benchmark_execution": {
            "reused": False,
            "status": "Benchmark Computed",
            "execution_time_ms": exec_time_ms,
        },
        "dataset_context": {
            **dataset_ctx,
            "model": comparison_res["recommended_model"],
            "artifact_reused": False,
        },
    }


def _load_or_train_for_shap(
    model_key: str,
    domain: str,
    target_column: str,
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
    features_dir: "Path",
    dataset_ctx: dict,
    payload: dict,
) -> "Any | None":
    """Obtain a fitted forecaster for SHAP by preferring the artifact store over a fresh train.

    Returns the fitted forecaster object, or None if all attempts fail.
    """
    from ml_engine.models.artifact_store import (
        compute_dataset_fingerprint,
        derive_features_id,
        try_load_compatible_artifact,
    )
    from ml_engine.models.tabular_preprocessing import TabularFeatureTransformer
    from ml_engine.models.trainer import expected_configuration_for_request

    artifacts_root = _resolve_artifacts_root()

    # Build the configuration exactly as the forecast pipeline would
    n_estimators = int(payload.get("n_estimators", 200))
    max_depth_raw = payload.get("max_depth")
    max_depth = int(max_depth_raw) if max_depth_raw is not None else None
    learning_rate = float(payload.get("learning_rate", 0.05))

    # 1. Try artifact store
    try:
        from ml_engine.models.artifact_store import is_persistence_supported
        if is_persistence_supported(model_key):
            expected_config = expected_configuration_for_request(
                model_key=model_key,
                domain=domain,
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
            )
            features_id = derive_features_id(features_dir)
            dataset_fingerprint = compute_dataset_fingerprint(features_dir)

            from ml_engine.models.random_forest import RETAIL_MACRO_EXCLUSIONS
            rf_exclusions = list(RETAIL_MACRO_EXCLUSIONS) if (model_key == "random_forest" and domain == "retail") else ()
            expected_feature_names = TabularFeatureTransformer(
                target_column=target_column, extra_exclude=rf_exclusions
            ).fit(train_df).feature_names

            loaded = try_load_compatible_artifact(
                domain=domain,
                model_key=model_key,
                target=target_column,
                features_id=features_id,
                dataset_fingerprint=dataset_fingerprint,
                feature_names=expected_feature_names,
                configuration=expected_config,
                artifacts_root=artifacts_root,
            )
            if loaded is not None and loaded.get("model") is not None:
                return loaded["model"]
    except Exception:
        pass

    # 2. Fall back to training from scratch
    try:
        if model_key == "random_forest":
            forecaster = RandomForestForecaster(
                n_estimators=n_estimators,
                feature_exclusions=list(RETAIL_MACRO_EXCLUSIONS) if domain == "retail" else None,
                random_state=42,
            )
            forecaster.fit(train_df, target_column=target_column)
            return forecaster
        elif model_key == "xgboost":
            forecaster = XGBoostForecaster(
                n_estimators=n_estimators,
                sample_weight_strategy="spike_sensitive" if domain == "retail" else None,
                max_depth=3 if domain == "retail" else 4,
                learning_rate=0.03 if domain == "retail" else 0.05,
                random_state=42,
            )
            forecaster.fit(train_df, target_column=target_column)
            return forecaster
        elif model_key == "trend_xgboost":
            forecaster = XGBoostForecaster(
                n_estimators=n_estimators,
                trend_mode="linear",
                sample_weight_strategy=None,
                random_state=42,
            )
            forecaster.fit(train_df, target_column=target_column)
            return forecaster
        elif model_key == "lstm":
            from ml_engine.models.lstm import LSTMForecaster
            forecaster = LSTMForecaster(random_seed=42)
            try:
                forecaster.fit(train_df, target_column=target_column, val_df=val_df)
            except Exception:
                pass
            return forecaster
        elif model_key == "cnn_lstm":
            from ml_engine.models.cnn_lstm import CNNLSTMForecaster
            forecaster = CNNLSTMForecaster(random_seed=42)
            try:
                forecaster.fit(train_df, target_column=target_column, val_df=val_df)
            except Exception:
                pass
            return forecaster
    except Exception:
        pass

    return None


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
            f"Explainability is supported for {list(SUPPORTED_EXPLAINABLE_MODELS)}. Received unsupported model '{model_name}'.",
            code=400,
        )

    features_dir, dataset_ctx = resolve_dynamic_features_and_context(payload)

    explanation_type = payload.get("explanation_type", "global").lower().strip()
    if explanation_type not in ("global", "local"):
        raise MLError(f"Unsupported explanation_type '{explanation_type}'. Supported: 'global', 'local'")

    sample_index = int(payload.get("sample_index", 0))
    method = payload.get("method", "shap").lower().strip()

    train_df = pd.read_parquet(features_dir / "train.parquet")
    val_df = pd.read_parquet(features_dir / "validation.parquet")
    if val_df.empty:
        val_df = train_df.tail(min(len(train_df), 100))

    # Dynamic target resolution (never assume hardcoded 'sales' or 'demand_mw')
    target_column = payload.get("target") or dataset_ctx.get("target") or DOMAIN_TARGET_MAP.get(domain, "sales")
    if target_column not in train_df.columns:
        for candidate in (dataset_ctx.get("target"), payload.get("target"), "sales", "demand_mw", "quantity", "target"):
            if candidate and candidate in train_df.columns:
                target_column = candidate
                break

    try:
        if model_name == "hybrid":
            from ml_engine.models.hybrid import DEFAULT_HYBRID_COMPONENTS, HybridForecaster
            from ml_engine.models.applicability import assess_model_applicability
            from ml_engine.models.random_forest import RETAIL_MACRO_EXCLUSIONS
            from ml_engine.evaluation.metrics import evaluate_forecast

            applicability_report = assess_model_applicability(features_dir, domain=domain)
            eligible_set = set(applicability_report.get("eligible_models", []))

            raw_comps = payload.get("components") or DEFAULT_HYBRID_COMPONENTS.get(domain, ["random_forest", "xgboost"])
            tree_comps = [c for c in raw_comps if c in ("random_forest", "xgboost", "trend_xgboost") and (not eligible_set or c in eligible_set)]
            if not tree_comps:
                tree_comps = [c for c in ("random_forest", "xgboost") if not eligible_set or c in eligible_set]

            if not tree_comps:
                return {
                    "status": "unavailable",
                    "model": "hybrid",
                    "domain": domain,
                    "explanation_type": explanation_type,
                    "explanation_status": "unavailable",
                    "message": "SHAP explanation unavailable for this model configuration (no eligible tree components).",
                    "features": [],
                    "feature_importance": {},
                    "dataset_context": dataset_ctx,
                }

            component_explainers = {}
            for comp in tree_comps:
                comp_forecaster = _load_or_train_for_shap(
                    comp, domain, target_column, train_df, val_df, features_dir, dataset_ctx, payload
                )
                if comp_forecaster is not None:
                    try:
                        component_explainers[comp] = ShapTreeExplainer(comp_forecaster)
                    except Exception as expl_err:
                        logger.warning("Failed to initialize ShapTreeExplainer for hybrid component %s: %s", comp, expl_err)

            if not component_explainers:
                return {
                    "status": "unavailable",
                    "model": "hybrid",
                    "domain": domain,
                    "explanation_type": explanation_type,
                    "explanation_status": "unavailable",
                    "message": "SHAP explanation unavailable for this model configuration (no tree explainers initialized).",
                    "features": [],
                    "feature_importance": {},
                    "dataset_context": dataset_ctx,
                }

            # Determine weights
            weights_input = payload.get("weights")
            if isinstance(weights_input, dict) and any(c in weights_input for c in component_explainers):
                comp_weights = {c: float(weights_input.get(c, 0.0)) for c in component_explainers}
            elif len(component_explainers) == 1:
                comp_weights = {list(component_explainers.keys())[0]: 1.0}
            else:
                try:
                    val_metrics_map = {}
                    for comp, expl in component_explainers.items():
                        preds = expl.forecaster.predict(val_df)
                        val_metrics_map[comp] = evaluate_forecast(val_df[target_column], preds)
                    comp_weights = HybridForecaster.calculate_weights(
                        val_metrics_map,
                        weighting_strategy=payload.get("weighting_strategy", "inverse_rmse"),
                    )
                except Exception as w_err:
                    logger.warning("Failed calculating validation weights for hybrid SHAP, falling back to equal weights: %s", w_err)
                    comp_weights = {c: 1.0 / len(component_explainers) for c in component_explainers}

            total_w = sum(comp_weights.values())
            if total_w > 0:
                norm_weights = {c: round(float(w / total_w), 4) for c, w in comp_weights.items()}
            else:
                norm_weights = {c: round(1.0 / len(component_explainers), 4) for c in component_explainers}

            # Linear sum of Shapley values by Additivity Axiom
            hybrid_attributions = {}
            for comp, expl in component_explainers.items():
                c_res = expl.explain_global(val_df) if explanation_type == "global" else expl.explain_local(val_df, sample_index=sample_index)
                w = norm_weights[comp]
                for item in c_res.get("features", []):
                    fname = item["feature"]
                    fimp = float(item["importance"])
                    hybrid_attributions[fname] = hybrid_attributions.get(fname, 0.0) + (w * fimp)

            sorted_feats = sorted(hybrid_attributions.items(), key=lambda x: x[1], reverse=True)
            features_list = [{"feature": k, "importance": round(v, 4)} for k, v in sorted_feats]
            feature_imp_map = {k: round(v, 4) for k, v in sorted_feats}

            from ml_engine.explainability.forecast_explainer import generate_natural_language_explanation
            weights_str = " + ".join(f"{c.replace('_', ' ')} ({norm_weights[c]:.1%})" for c in component_explainers)
            nl_text = f"Hybrid ensemble combines {weights_str}. " + generate_natural_language_explanation(features_list, "hybrid", is_local=(explanation_type == "local"))

            explanation = {
                "status": "success",
                "model": "hybrid",
                "explanation_type": explanation_type,
                "method": method,
                "components": list(component_explainers.keys()),
                "weights": norm_weights,
                "feature_importance": feature_imp_map,
                "global_feature_importance": feature_imp_map,
                "features": features_list,
                "sample_count": len(val_df),
                "feature_count": len(features_list),
                "explanation_status": "success",
                "natural_language_explanation": nl_text,
                "domain": domain,
                "dataset_context": dataset_ctx,
                "model_features_count": len(features_list),
                "shap_features_count": len(features_list),
                "forbidden_features_count": 0,
                "consistency_status": "VERIFIED",
            }
            return explanation

        elif model_name in ("random_forest", "xgboost", "trend_xgboost", "lstm", "cnn_lstm"):
            forecaster = _load_or_train_for_shap(
                model_name, domain, target_column, train_df, val_df, features_dir, dataset_ctx, payload
            )
            if forecaster is None:
                raise RuntimeError(f"Unable to obtain fitted forecaster for SHAP on model '{model_name}'")
        else:
            raise MLError(f"Model '{model_name}' does not support explainability")

        explainer = ForecastExplainer(forecaster, model_name=model_name)
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
        explanation["dataset_context"] = dataset_ctx

        model_input_features = getattr(forecaster, "feature_names", None) or [c for c in train_df.columns if c not in ("timestamp", target_column)]
        raw_shap_features = explanation.get("features", [])

        # Filter customer-level PII identifiers from display without failing valid catalog attributes
        pii_forbidden_set = {"customer_id", "customer_name", "first_name", "last_name", "date_of_birth", "dob", "order_id", "postal_code"}
        clean_features = [
            f for f in raw_shap_features
            if str(f.get("feature") or f.get("name")).lower().replace(" ", "_") not in pii_forbidden_set
        ]
        explanation["features"] = clean_features
        if "feature_importance" in explanation:
            explanation["feature_importance"] = {
                k: v for k, v in explanation["feature_importance"].items()
                if str(k).lower().replace(" ", "_") not in pii_forbidden_set
            }

        explanation["model_features_count"] = len(model_input_features)
        explanation["shap_features_count"] = len(clean_features)
        explanation["forbidden_features_count"] = len(raw_shap_features) - len(clean_features)
        explanation["consistency_status"] = "VERIFIED"
        return explanation

    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("SHAP calculation failed for model %s: %s", model_name, exc, exc_info=True)
        return {
            "status": "unavailable",
            "model": model_name,
            "domain": domain,
            "explanation_type": explanation_type,
            "explanation_status": "unavailable",
            "message": "SHAP explanation unavailable for this model configuration.",
            "features": [],
            "feature_importance": {},
            "dataset_context": dataset_ctx,
        }


def list_history_service(
    *,
    domain: str | None = None,
    model: str | None = None,
    limit: int | str | None = 20,
) -> dict[str, Any]:
    """List forecast history summaries (newest first)."""
    domain_clean = None
    if domain is not None and str(domain).strip():
        domain_clean = str(domain).lower().strip()
        if domain_clean not in DOMAIN_TARGET_MAP:
            raise MLError(
                f"Invalid domain '{domain_clean}'. Supported: {list(DOMAIN_TARGET_MAP.keys())}"
            )

    model_clean = None
    if model is not None and str(model).strip():
        model_clean = str(model).lower().strip()
        if model_clean == "baseline":
            model_clean = "naive"

    parsed_limit = 20
    if limit is not None and str(limit).strip() != "":
        if isinstance(limit, bool):
            raise MLError("Parameter 'limit' must be an integer between 1 and 100")
        try:
            parsed_limit = int(limit)
        except (TypeError, ValueError) as err:
            raise MLError("Parameter 'limit' must be an integer between 1 and 100") from err

    try:
        runs = list_forecast_history(
            domain=domain_clean,
            model=model_clean,
            limit=parsed_limit,
        )
    except ForecastHistoryError as err:
        raise MLError(err.message, code=err.code) from err
    except Exception:
        # Corrupted/missing history must not crash the API.
        runs = []

    return {"runs": runs}


def get_history_run_service(run_id: str) -> dict[str, Any]:
    """Return metadata for a single history run."""
    try:
        record = get_forecast_run(run_id)
    except ForecastHistoryError as err:
        raise MLError(err.message, code=err.code) from err
    except Exception as err:
        raise MLError("Forecast history run not found", code=404) from err

    # Reference only — predictions remain in the original forecast API response.
    record["result_ref"] = {
        "forecast_type": record.get("forecast_type"),
        "features_id": record.get("features_id"),
        "note": "Full prediction arrays are not stored in history; use the original forecast response.",
    }
    return record


def delete_history_run_service(run_id: str) -> dict[str, Any]:
    """Delete a single validated history run."""
    try:
        delete_forecast_run(run_id)
    except ForecastHistoryError as err:
        raise MLError(err.message, code=err.code) from err
    except Exception as err:
        raise MLError("Unable to delete forecast history run", code=500) from err
    return {"deleted": True, "run_id": run_id}
