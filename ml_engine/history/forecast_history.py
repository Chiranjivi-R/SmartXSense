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
) -> dict[str, Any]:
    """Build a metadata-only history record (no prediction arrays)."""
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
        # Persist only standard scalar metrics; never prediction lists.
        clean_metrics = {}
        for key in ("mae", "rmse", "mape", "r2"):
            if key in metrics and metrics[key] is not None:
                clean_metrics[key] = float(metrics[key])
        record["metrics"] = clean_metrics
    if horizon is not None:
        record["horizon"] = int(horizon)
    if selection_method:
        record["selection_method"] = str(selection_method)
    if summary_facts:
        record["summary_facts"] = {str(key): value for key, value in summary_facts.items() if isinstance(value, (str, int, float, bool)) or value is None}
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
            metrics = result.get("test_metrics")
        record = build_history_record(
            domain=str(result.get("domain", "")),
            model=str(result.get("model", "")),
            target=str(result.get("target", "")),
            forecast_type=forecast_type,
            features_id=features_id,
            prediction_count=int(result.get("prediction_count") or len(result.get("predictions") or [])),
            metrics=metrics if isinstance(metrics, dict) else None,
            horizon=result.get("horizon") if forecast_type == "future" else None,
            artifact_source=artifact.get("source"),
            selection_method=selection_method,
            summary_facts=result.get("summary_facts") if isinstance(result.get("summary_facts"), dict) else None,
        )
        summary = record_forecast_run(record, history_root=history_root)
        return summary
    except Exception:
        return None
