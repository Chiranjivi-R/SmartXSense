"""Orchestration between dataset persistence and ML preprocessing utilities."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from flask import current_app
from sqlalchemy.exc import SQLAlchemyError

from ml_engine.features.artifacts import materialize_forecasting_features
from ml_engine.features.common import FeatureEngineeringError
from ml_engine.preprocessing.cleaner import PreprocessingError
from ml_engine.preprocessing.calendar_features import SUPPORTED_REGIONS
from ml_engine.preprocessing.domain import preprocess_frame
from ml_engine.splitting.time_series import TimeSeriesSplitError

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
    # Default True: connect upload flow to existing forecasting feature pipeline.
    generate_forecast_features = body.get("generate_forecast_features", True)
    dataset_region = str(body.get("dataset_region", "other")).lower().strip()
    if imputation not in ALLOWED_IMPUTATION or scaler not in ALLOWED_SCALERS or outlier_method not in ALLOWED_OUTLIER_METHODS:
        raise DatasetError("Invalid preprocessing parameters", errors={"preprocessing": ["Unsupported imputation, scaler, or outlier method."]})
    if not isinstance(generate_lags, bool) or not isinstance(lag_intervals, list) or not all(isinstance(value, int) and value > 0 for value in lag_intervals):
        raise DatasetError("Invalid preprocessing parameters", errors={"generate_lags": ["Must be a boolean."], "lag_intervals": ["Must contain positive integers."]})
    if generate_lags:
        raise DatasetError("Invalid preprocessing parameters", errors={"generate_lags": ["Lag features are not available during preprocessing."]})
    if not isinstance(generate_forecast_features, bool):
        raise DatasetError(
            "Invalid preprocessing parameters",
            errors={"generate_forecast_features": ["Must be a boolean."]},
        )
    if dataset_region not in SUPPORTED_REGIONS:
        raise DatasetError("Invalid preprocessing parameters", errors={"dataset_region": ["Must be India, UK, or Other."]})
    return {
        "imputation": imputation,
        "scaler": scaler,
        "outlier_method": outlier_method,
        "generate_forecast_features": generate_forecast_features,
        "dataset_region": dataset_region,
    }


def _processed_root() -> Path:
    return Path(current_app.config["PROCESSED_DATA_FOLDER"]).resolve()


def _features_root() -> Path:
    configured = current_app.config.get("FEATURES_DATA_FOLDER")
    if configured:
        return Path(str(configured)).resolve()
    return (Path(__file__).resolve().parents[3] / "ml_engine" / "data" / "features").resolve()


def _safe_relative_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[3]
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        # Configuration may deliberately use external managed storage. Do not
        # expose it; the API still returns a stable managed-file identifier.
        return f"ml_engine/data/processed/{path.parent.name}/{path.name}"


def _is_managed_features(path: Path) -> bool:
    try:
        path.resolve().relative_to(_features_root())
        return True
    except ValueError:
        return False


def preprocess_dataset(dataset: Dataset, payload: object) -> tuple[DatasetPreprocessing, dict]:
    """Create cleaned Parquet data and optional forecasting feature splits.

    Feature splits reuse ``prepare_forecasting_data`` (same pipeline as demo
    datasets). Small datasets that cannot satisfy chronological split rules
    still succeed for cleaning, with ``features_ready=false``.
    """
    options = _options(payload)
    raw_path = managed_upload_path(dataset)
    preprocessing_id = str(uuid.uuid4())
    output_dir = _processed_root() / dataset.domain
    output_path = output_dir / f"proc_{preprocessing_id}.parquet"
    features_dir_path: Path | None = None

    try:
        frame = _read_dataset(raw_path, _extension(dataset.filename))
        processed, features, _removed_rows = preprocess_frame(frame, dataset.domain, region=options["dataset_region"])
        # Region travels with the existing feature artifact.  It is a declared
        # dataset attribute, not inferred from the source or silently assumed.
        processed["calendar_region"] = options["dataset_region"]
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

    features_payload: dict = {
        "features_ready": False,
        "feature_names": [],
        "features_dir": None,
        "train_rows": 0,
        "validation_rows": 0,
        "test_rows": 0,
    }
    lineage: dict = {"lags": [], "features_ready": False}

    if options["generate_forecast_features"]:
        try:
            # Reconstruct a typed frame for feature engineering (parquet round-trip).
            processed_frame = pd_read_parquet_safe(output_path)
            materialized = materialize_forecasting_features(
                processed_frame,
                dataset.domain,
                output_root=_features_root(),
                artifact_id=preprocessing_id,
            )
            features_dir_path = Path(materialized["features_dir"])
            relative_features = _safe_relative_path(features_dir_path)
            features_payload = {
                "features_ready": True,
                "feature_names": materialized["feature_names"],
                "features_dir": relative_features,
                "features_id": materialized["features_id"],
                "train_rows": materialized["train_rows"],
                "validation_rows": materialized["validation_rows"],
                "test_rows": materialized["test_rows"],
                "feature_report": materialized["report"],
            }
            lineage = {
                "lags": list(materialized["feature_names"]),
                "features_ready": True,
                "features_dir": relative_features,
                "features_id": materialized["features_id"],
            }
        except (FeatureEngineeringError, TimeSeriesSplitError, ValueError, OSError) as exc:
            if features_dir_path is not None and _is_managed_features(features_dir_path):
                shutil.rmtree(features_dir_path, ignore_errors=True)
            features_payload = {
                "features_ready": False,
                "feature_names": [],
                "features_dir": None,
                "train_rows": 0,
                "validation_rows": 0,
                "test_rows": 0,
                "features_error": (
                    "Forecasting feature splits could not be generated from this dataset. "
                    f"Reason: {exc}. Ensure the dataset has enough chronological observations "
                    "after cleaning (at least 3 unique timestamps; lag warm-up may remove early rows)."
                ),
            }
            lineage = {"lags": [], "features_ready": False, "features_error": str(exc)}

    try:
        record = DatasetPreprocessing(
            preprocessing_id=preprocessing_id,
            dataset_id=dataset.dataset_id,
            imputation_strategy=options["imputation"],
            scaler_type=options["scaler"],
            lags_generated_json=json.dumps(lineage),
            processed_file_path=str(output_path),
        )
        db.session.add(record)
        dataset.status = "processed"
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        output_path.unlink(missing_ok=True)
        if features_dir_path is not None and _is_managed_features(features_dir_path):
            shutil.rmtree(features_dir_path, ignore_errors=True)
        raise DatasetError("Unable to save preprocessing results", 500) from exc

    return record, {
        "processed_dataset_id": record.preprocessing_id,
        "processed_rows": len(processed),
        "features_generated": features,
        "output_file_path": _safe_relative_path(output_path),
        **features_payload,
    }


def pd_read_parquet_safe(path: Path):
    import pandas as pd

    return pd.read_parquet(path)


def cleanup_feature_artifacts_for_preprocessing(record: DatasetPreprocessing) -> None:
    """Remove managed feature-split directories referenced by a preprocessing record."""
    if not record.lags_generated_json:
        return
    try:
        payload = json.loads(record.lags_generated_json)
    except json.JSONDecodeError:
        return
    if not isinstance(payload, dict):
        return
    features_dir = payload.get("features_dir")
    if not features_dir:
        return
    path = Path(str(features_dir))
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[3]
        path = (project_root / path).resolve()
    else:
        path = path.resolve()
    if _is_managed_features(path) and path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
