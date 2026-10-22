"""File-based forecast history for SmartXSense (Phase D).

Storage layout (controlled project-local directory):

    ml_engine/artifacts/history/
        index.json
        runs/<run_id>.json

Index entries are lightweight summaries (no prediction arrays).
Run files store metadata only; huge prediction payloads are never persisted.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORY_SCHEMA_VERSION = 1
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
_LOCK = threading.Lock()

DEFAULT_HISTORY_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "history"


class ForecastHistoryError(ValueError):
    """Raised for invalid history operations."""

    def __init__(self, message: str, code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


def default_history_root() -> Path:
    return DEFAULT_HISTORY_ROOT.resolve()


def resolve_history_root(explicit: Path | str | None = None) -> Path:
    """Resolve history root from argument, Flask config, or project default."""
    if explicit is not None:
        return Path(explicit).resolve()
    try:
        from flask import current_app, has_app_context

        if has_app_context():
            configured = current_app.config.get("FORECAST_HISTORY_ROOT")
            if configured:
                return Path(str(configured)).resolve()
    except Exception:
        pass
    return default_history_root()


def _index_path(root: Path) -> Path:
    return root / "index.json"


def _runs_dir(root: Path) -> Path:
    return root / "runs"


def _run_path(root: Path, run_id: str) -> Path:
    if not _SAFE_RUN_ID.match(run_id):
        raise ForecastHistoryError("Invalid run_id", code=404)
    path = (_runs_dir(root) / f"{run_id}.json").resolve()
    if _runs_dir(root).resolve() not in path.parents:
        raise ForecastHistoryError("Invalid run_id", code=404)
    return path


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return default


def _load_index(root: Path) -> list[dict[str, Any]]:
    raw = _read_json(_index_path(root), default=[])
    if not isinstance(raw, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        run_id = item.get("run_id")
        if not isinstance(run_id, str) or not _SAFE_RUN_ID.match(run_id):
            continue
        # Drop any accidentally persisted prediction arrays from index entries.
        if "predictions" in item:
            item = {key: value for key, value in item.items() if key != "predictions"}
        if not item.get("domain") or not item.get("model") or not item.get("forecast_type"):
            continue
        cleaned.append(item)
    return cleaned


def _summary_from_record(record: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "run_id": record["run_id"],
        "timestamp": record["timestamp"],
        "domain": record["domain"],
        "model": record["model"],
        "target": record["target"],
        "forecast_type": record["forecast_type"],
        "prediction_count": record.get("prediction_count", 0),
        "features_id": record.get("features_id"),
        "artifact_source": record.get("artifact_source"),
    }
    if record.get("metrics") is not None:
        summary["metrics"] = record["metrics"]
    if record.get("horizon") is not None:
        summary["horizon"] = record["horizon"]
    if record.get("selection_method"):
        summary["selection_method"] = record["selection_method"]
    return summary


def build_history_record(
    *,
    domain: str,
    model: str,
    target: str,
    forecast_type: str,
    features_id: str | None,
    prediction_count: int,
    metrics: dict[str, float] | None = None,
    horizon: int | None = None,
    artifact_source: str | None = None,
    selection_method: str | None = None,
    summary_facts: dict[str, Any] | None = None,
    dataset_info: dict[str, Any] | None = None,
    model_info: dict[str, Any] | None = None,
    validation_benchmark: dict[str, Any] | None = None,
    test_evaluation: dict[str, Any] | None = None,
    forecast_data: dict[str, Any] | None = None,
    explainability: dict[str, Any] | None = None,
    dataset_intelligence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a comprehensive forecast history and report record (never stores raw 'predictions' key)."""
    if forecast_type not in {"test_evaluation", "future"}:
        raise ForecastHistoryError("Invalid forecast_type")
    if model.lower().strip() == "auto_select":
        raise ForecastHistoryError("History must record the actual selected model, not auto_select")

    record: dict[str, Any] = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "run_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "domain": domain.lower().strip(),
        "model": model.lower().strip(),
        "target": target,
        "forecast_type": forecast_type,
        "prediction_count": int(prediction_count),
        "features_id": features_id,
        "artifact_source": artifact_source,
    }
    if metrics:
        clean_metrics = {}
        for key in ("mae", "rmse", "mape", "r2", "wape", "smape"):
            if key in metrics and metrics[key] is not None:
                clean_metrics[key] = float(metrics[key])
        record["metrics"] = clean_metrics
    if horizon is not None:
        record["horizon"] = int(horizon)
    if selection_method:
        record["selection_method"] = str(selection_method)
    if summary_facts:
        record["summary_facts"] = {str(key): value for key, value in summary_facts.items() if isinstance(value, (str, int, float, bool)) or value is None}
    if dataset_info:
        record["dataset_info"] = dataset_info
    if model_info:
        record["model_info"] = model_info
    if validation_benchmark:
        record["validation_benchmark"] = validation_benchmark
    if test_evaluation:
        record["test_evaluation"] = test_evaluation
    if forecast_data:
        record["forecast_data"] = forecast_data
    if explainability:
        record["explainability"] = explainability
    if dataset_intelligence:
        record["dataset_intelligence"] = dataset_intelligence
    return record


def record_forecast_run(
    record: dict[str, Any],
    *,
    history_root: Path | str | None = None,
) -> dict[str, Any]:
    """Persist a history record. Never stores prediction arrays."""
    if "predictions" in record:
        raise ForecastHistoryError("History records must not include predictions")

    root = resolve_history_root(history_root)
    with _LOCK:
        root.mkdir(parents=True, exist_ok=True)
        _runs_dir(root).mkdir(parents=True, exist_ok=True)
        run_file = _run_path(root, record["run_id"])
        _atomic_write_json(run_file, record)

        index = _load_index(root)
        summary = _summary_from_record(record)
        index = [summary, *[item for item in index if item.get("run_id") != record["run_id"]]]
        _atomic_write_json(_index_path(root), index)
    return summary


def list_forecast_history(
    *,
    domain: str | None = None,
    model: str | None = None,
    limit: int = 20,
    history_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Return newest-first history summaries with optional filters."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ForecastHistoryError("Parameter 'limit' must be an integer between 1 and 100")
    if limit < 1 or limit > 100:
        raise ForecastHistoryError("Parameter 'limit' must be between 1 and 100")

    root = resolve_history_root(history_root)
    with _LOCK:
        index = _load_index(root)

    runs = index
    if domain:
        domain_clean = domain.lower().strip()
        runs = [item for item in runs if item.get("domain") == domain_clean]
    if model:
        model_clean = model.lower().strip()
        runs = [item for item in runs if item.get("model") == model_clean]

    # Newest first by timestamp (index is already prepended; re-sort for safety).
    runs = sorted(runs, key=lambda item: str(item.get("timestamp", "")), reverse=True)
    return runs[:limit]


def get_forecast_run(
    run_id: str,
    *,
    history_root: Path | str | None = None,
) -> dict[str, Any]:
    root = resolve_history_root(history_root)
    with _LOCK:
        path = _run_path(root, run_id)
        if not path.is_file():
            raise ForecastHistoryError("Forecast history run not found", code=404)
        record = _read_json(path, default=None)
    if not isinstance(record, dict) or record.get("run_id") != run_id:
        raise ForecastHistoryError("Forecast history run not found", code=404)
    if "predictions" in record:
        # Defensive: never return accidentally stored prediction arrays.
        record = {key: value for key, value in record.items() if key != "predictions"}
    return record


def delete_forecast_run(
    run_id: str,
    *,
    history_root: Path | str | None = None,
) -> None:
    root = resolve_history_root(history_root)
    with _LOCK:
        path = _run_path(root, run_id)
        if not path.is_file():
            raise ForecastHistoryError("Forecast history run not found", code=404)
        try:
            path.unlink()
        except OSError as err:
            raise ForecastHistoryError("Unable to delete forecast history run", code=500) from err
        index = [item for item in _load_index(root) if item.get("run_id") != run_id]
        _atomic_write_json(_index_path(root), index)


def safe_record_from_forecast_result(
    *,
    result: dict[str, Any],
    forecast_type: str,
    features_id: str | None,
    selection_method: str | None = None,
    history_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Best-effort history write that never raises into the forecasting API."""
    try:
        artifact = result.get("artifact") if isinstance(result.get("artifact"), dict) else {}
        metrics = None
        if forecast_type == "test_evaluation":
            metrics = result.get("test_metrics") or result.get("metrics")

        # 1. Dataset information
        ds_ctx = result.get("dataset_context") or {}
        dataset_info = {
            "dataset_id": ds_ctx.get("dataset_id"),
            "dataset_name": ds_ctx.get("dataset_name") or ds_ctx.get("name") or f"{str(result.get('domain', '')).capitalize()} Dataset",
            "dataset_hash": ds_ctx.get("dataset_hash"),
            "domain": result.get("domain"),
            "target": result.get("target"),
            "frequency": ds_ctx.get("frequency") or ("Hourly (h)" if result.get("domain") == "electricity" else "Daily (D)"),
            "geographic_scope": ds_ctx.get("geography_label") or ds_ctx.get("location") or "All Geographies",
            "observation_count": int(result.get("prediction_count") or len(result.get("predictions") or [])),
            "features_id": features_id,
        }

        # 2. Model information
        raw_m = str(result.get("model", "")).lower().strip()
        model_info = {
            "model": raw_m,
            "model_name": raw_m.replace("_", " ").title(),
            "selection_method": selection_method or ("auto_select" if result.get("validation_benchmark") else "manual"),
            "selection_criterion": "Lowest Validation RMSE" if (selection_method == "auto_select" or result.get("validation_benchmark")) else "User specified model",
            "artifact_source": artifact.get("source"),
            "artifact_key": result.get("artifact_key") or artifact.get("key"),
            "artifact_reused": bool(result.get("execution", {}).get("artifact_reused", artifact.get("used", False))),
            "configuration": result.get("configuration") or {},
        }

        # 3. Validation benchmark
        val_bench = result.get("validation_benchmark")
        if isinstance(val_bench, dict):
            validation_benchmark = {
                "candidate_models": val_bench.get("candidate_models") or [r.get("model") for r in val_bench.get("ranking", [])],
                "ranking": val_bench.get("ranking") or [],
                "recommended_model": val_bench.get("recommended_model") or raw_m,
                "selection_metric": val_bench.get("selection_metric", "rmse"),
                "selection_split": "validation",
                "description": "Candidate models evaluated on chronologically separated validation set (Lowest RMSE selected).",
            }
        else:
            validation_benchmark = None

        # 4. Test evaluation
        if forecast_type == "test_evaluation" and isinstance(metrics, dict):
            test_evaluation = {
                "split": "held_out_test",
                "label": "Held-out Test Evaluation",
                "description": "Performance metrics evaluated strictly on unseen chronologically held-out test data.",
                "metrics": {k: float(v) for k, v in metrics.items() if v is not None and isinstance(v, (int, float))},
            }
        else:
            test_evaluation = None

        # 5. Forecast data (stored as 'forecast_data', never 'predictions' key)
        raw_preds = result.get("predictions") or []
        points = []
        pred_values = []
        for p in raw_preds:
            if isinstance(p, dict):
                p_val = float(p["predicted"]) if p.get("predicted") is not None else None
                a_val = float(p["actual"]) if (forecast_type == "test_evaluation" and p.get("actual") is not None) else None
                r_val = round(a_val - p_val, 4) if (a_val is not None and p_val is not None) else None
                pt = {
                    "timestamp": str(p.get("timestamp")),
                    "predicted": p_val,
                }
                if p.get("lower_bound") is not None:
                    pt["lower_bound"] = float(p["lower_bound"])
                elif p.get("lower") is not None:
                    pt["lower_bound"] = float(p["lower"])
                if p.get("upper_bound") is not None:
                    pt["upper_bound"] = float(p["upper_bound"])
                elif p.get("upper") is not None:
                    pt["upper_bound"] = float(p["upper"])

                if forecast_type == "test_evaluation":
                    pt["actual"] = a_val
                    pt["residual"] = r_val
                points.append(pt)
                if p_val is not None:
                    pred_values.append(p_val)

        summary_stats = {
            "count": len(points),
            "forecast_type": forecast_type,
            "horizon": result.get("horizon") if forecast_type == "future" else len(points),
            "mean": round(float(sum(pred_values) / len(pred_values)), 2) if pred_values else None,
            "min": round(float(min(pred_values)), 2) if pred_values else None,
            "max": round(float(max(pred_values)), 2) if pred_values else None,
        }
        forecast_data = {
            "summary": summary_stats,
            "points": points,
        }

        # 6. Dataset Intelligence & Data Quality
        ds_intel = result.get("dataset_intelligence") or ds_ctx.get("dataset_intelligence")
        if not isinstance(ds_intel, dict):
            clean_sum = ds_ctx.get("cleaning_summary") or {}
            suit_data = ds_ctx.get("suitability") or {}
            orig_rows = ds_ctx.get("raw_rows") or clean_sum.get("raw_row_count") or ds_ctx.get("rows_used")
            cleaned_rows = ds_ctx.get("cleaned_rows") or clean_sum.get("final_cleaned_rows")
            tr_rows = ds_ctx.get("train_rows")
            val_rows = ds_ctx.get("validation_rows")
            te_rows = ds_ctx.get("test_rows")
            horizon_steps = result.get("horizon") if forecast_type == "future" else len(points)

            ds_intel = {
                "dataset_name": dataset_info.get("dataset_name"),
                "dataset_id": dataset_info.get("dataset_id"),
                "original_row_count": orig_rows,
                "original_column_count": ds_ctx.get("column_count"),
                "cleaned_row_count": cleaned_rows,
                "training_observations": tr_rows,
                "validation_observations": val_rows,
                "test_observations": te_rows,
                "forecast_horizon": horizon_steps,
                "detected_time_column": ds_ctx.get("time_column") or "timestamp",
                "detected_target_column": result.get("target"),
                "canonical_target": result.get("canonical_target") or result.get("target"),
                "unique_timestamps": ds_ctx.get("unique_timestamps") or suit_data.get("unique_timestamps"),
                "duplicate_rows_removed": clean_sum.get("exact_duplicates_removed") or ds_ctx.get("duplicate_rows_removed"),
                "duplicate_timestamps_handled": bool(clean_sum.get("duplicate_timestamps_count") or ds_ctx.get("duplicate_timestamps_count")),
                "duplicate_timestamps_count": clean_sum.get("duplicate_timestamps_count") or ds_ctx.get("duplicate_timestamps_count"),
                "missing_values_handled": clean_sum.get("missing_values_handled") or ds_ctx.get("missing_values_handled"),
                "excluded_features": clean_sum.get("excluded_features") or ds_ctx.get("excluded_features") or [],
                "excluded_reasons": clean_sum.get("excluded_reasons") or ds_ctx.get("excluded_reasons") or {},
                "excluded_features_count": clean_sum.get("excluded_features_count") or ds_ctx.get("excluded_features_count"),
                "final_forecasting_features_count": ds_ctx.get("final_forecasting_features_count"),
                "domain_fit_score": ds_ctx.get("confidence") or suit_data.get("confidence"),
                "domain_fit_label": ds_ctx.get("confidence_label") or suit_data.get("confidence_label"),
                "suitability_status": suit_data.get("classification") or ("SUITABLE" if suit_data.get("is_suitable") else None),
                "suitability_label": suit_data.get("status_label"),
                "suitability_warnings": suit_data.get("warnings") or [],
                "suitability_reasons": suit_data.get("reasons") or [],
                "explanation": suit_data.get("explanation") or result.get("dataset_explanation"),
            }

        # 7. Explainability
        feat_imp = result.get("feature_importance") or result.get("feature_importances")
        llm_exp = result.get("llm_explanation")
        shap_data = result.get("shap_explanation") or result.get("shap_features")
        lime_data = result.get("lime_explanation")

        val_rmse = None
        if isinstance(val_bench, dict) and val_bench.get("ranking"):
            for candidate in val_bench["ranking"]:
                if candidate.get("model") == raw_m or candidate.get("rank") == 1:
                    val_rmse = (candidate.get("validation_metrics") or candidate.get("metrics") or {}).get("rmse")
                    if val_rmse is not None:
                        break
        if val_rmse is None and isinstance(result.get("validation_metrics"), dict):
            val_rmse = result["validation_metrics"].get("rmse")

        explainability = {
            "selected_model": raw_m,
            "selection_reason": result.get("selection_reason") or model_info.get("selection_method"),
            "selection_criterion": model_info.get("selection_criterion", "Lowest Validation RMSE"),
            "validation_rmse": val_rmse,
            "feature_importance": feat_imp if isinstance(feat_imp, (dict, list)) else None,
            "shap_features": shap_data if isinstance(shap_data, (dict, list)) else None,
            "shap_explanation": result.get("shap_explanation"),
            "lime_explanation": lime_data if isinstance(lime_data, (dict, list)) else None,
            "ai_explanation": llm_exp if isinstance(llm_exp, (dict, str)) else None,
        }

        record = build_history_record(
            domain=str(result.get("domain", "")),
            model=raw_m,
            target=str(result.get("target", "")),
            forecast_type=forecast_type,
            features_id=features_id,
            prediction_count=int(result.get("prediction_count") or len(raw_preds)),
            metrics=metrics if isinstance(metrics, dict) else None,
            horizon=result.get("horizon") if forecast_type == "future" else None,
            artifact_source=artifact.get("source"),
            selection_method=selection_method,
            summary_facts=result.get("summary_facts") if isinstance(result.get("summary_facts"), dict) else None,
            dataset_info=dataset_info,
            model_info=model_info,
            validation_benchmark=validation_benchmark,
            test_evaluation=test_evaluation,
            forecast_data=forecast_data,
            explainability=explainability,
            dataset_intelligence=ds_intel,
        )
        summary = record_forecast_run(record, history_root=history_root)
        return summary
    except Exception:
        return None
