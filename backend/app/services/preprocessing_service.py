"""Orchestration between dataset persistence and ML preprocessing utilities."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from flask import current_app
from sqlalchemy.exc import SQLAlchemyError

from ml_engine.preprocessing.cleaner import PreprocessingError
from ml_engine.preprocessing.domain import preprocess_frame

from ..extensions import db
from ..models import Dataset, DatasetPreprocessing
from .dataset_service import DatasetError, _extension, _read_dataset, managed_upload_path


ALLOWED_IMPUTATION = {"linear", "mean", "knn", "none"}
ALLOWED_SCALERS = {"standard", "minmax", "robust", "none"}
ALLOWED_OUTLIER_METHODS = {"iqr", "none"}


def _options(payload: object) -> dict:
    body = payload if isinstance(payload, dict) else {}
    imputation = body.get("imputation_strategy", "none")
    scaler = body.get("scaler", "none")
    outlier_method = body.get("outlier_method", "none")
    generate_lags = body.get("generate_lags", False)
    lag_intervals = body.get("lag_intervals", [])
    if imputation not in ALLOWED_IMPUTATION or scaler not in ALLOWED_SCALERS or outlier_method not in ALLOWED_OUTLIER_METHODS:
        raise DatasetError("Invalid preprocessing parameters", errors={"preprocessing": ["Unsupported imputation, scaler, or outlier method."]})
    if not isinstance(generate_lags, bool) or not isinstance(lag_intervals, list) or not all(isinstance(value, int) and value > 0 for value in lag_intervals):
        raise DatasetError("Invalid preprocessing parameters", errors={"generate_lags": ["Must be a boolean."], "lag_intervals": ["Must contain positive integers."]})
    if generate_lags:
        raise DatasetError("Invalid preprocessing parameters", errors={"generate_lags": ["Lag features are not available during preprocessing."]})
    return {"imputation": imputation, "scaler": scaler, "outlier_method": outlier_method}


def _processed_root() -> Path:
    return Path(current_app.config["PROCESSED_DATA_FOLDER"]).resolve()


def _safe_relative_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[3]
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        # Configuration may deliberately use external managed storage. Do not
        # expose it; the API still returns a stable managed-file identifier.
        return f"ml_engine/data/processed/{path.parent.name}/{path.name}"


def preprocess_dataset(dataset: Dataset, payload: object) -> tuple[DatasetPreprocessing, dict]:
    """Create a cleaned Parquet dataset and persist the supported lineage metadata."""
    options = _options(payload)
    raw_path = managed_upload_path(dataset)
    preprocessing_id = str(uuid.uuid4())
    output_dir = _processed_root() / dataset.domain
    output_path = output_dir / f"proc_{preprocessing_id}.parquet"

    try:
        frame = _read_dataset(raw_path, _extension(dataset.filename))
        processed, features, _removed_rows = preprocess_frame(frame, dataset.domain)
        output_dir.mkdir(parents=True, exist_ok=True)
        processed.to_parquet(output_path, index=False)
    except (DatasetError, PreprocessingError) as exc:
        output_path.unlink(missing_ok=True)
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError("Dataset preprocessing failed", 400, {"schema": [str(exc)]}) from exc
    except (OSError, ValueError, ImportError) as exc:
        output_path.unlink(missing_ok=True)
        raise DatasetError("Dataset preprocessing failed", 500) from exc

    try:
        record = DatasetPreprocessing(
            preprocessing_id=preprocessing_id,
            dataset_id=dataset.dataset_id,
            imputation_strategy=options["imputation"],
            scaler_type=options["scaler"],
            lags_generated_json=json.dumps([]),
            processed_file_path=str(output_path),
        )
        db.session.add(record)
        dataset.status = "processed"
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        output_path.unlink(missing_ok=True)
        raise DatasetError("Unable to save preprocessing results", 500) from exc

    return record, {
        "processed_dataset_id": record.preprocessing_id,
        "processed_rows": len(processed),
        "features_generated": features,
        "output_file_path": _safe_relative_path(output_path),
    }
