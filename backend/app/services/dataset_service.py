"""Dataset upload, access, storage, and structural-validation business logic."""

from __future__ import annotations

import json
import logging
import os
import uuid
import zipfile
from pathlib import Path

import pandas as pd
from flask import current_app
from sqlalchemy import func, text
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from ml_engine.preprocessing.schema import DOMAIN_TARGETS, normalise_domain_schema

from ..extensions import db
from ..models import Dataset, DatasetPreprocessing, DatasetValidation
from .client_error_sanitize import sanitize_client_error_message

log = logging.getLogger(__name__)


ALLOWED_EXTENSIONS = {"csv", "xlsx", "parquet"}
ALLOWED_MIMETYPES = {
    "csv": {"text/csv", "application/csv", "text/plain"},
    "xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
    "parquet": {"application/x-parquet", "application/octet-stream", "application/vnd.apache.parquet"},
}
ALLOWED_DOMAINS = {"retail", "electricity"}


def _sanitize_error_map(errors: dict | None) -> dict | None:
    if not errors:
        return None
    cleaned: dict = {}
    for key, values in errors.items():
        if isinstance(values, list):
            cleaned[key] = [sanitize_client_error_message(str(v)) for v in values]
        else:
            cleaned[key] = sanitize_client_error_message(str(values))
    return cleaned


class DatasetError(Exception):
    """Expected API-facing dataset failure."""

    def __init__(self, message: str, code: int = 400, errors: dict | None = None):
        safe = sanitize_client_error_message(message)
        super().__init__(safe)
        self.message = safe
        self.code = code
        self.errors = _sanitize_error_map(errors)


def error_payload(error: DatasetError) -> dict:
    payload = {"status": "error", "code": error.code, "message": error.message}
    if error.errors:
        payload["errors"] = error.errors
    return payload


def _ensure_content_hash_column() -> None:
    """Safe one-shot migration: add content_hash to datasets if the column is missing.

    This is called at application startup so that an existing SQLite database that
    pre-dates the Phase 7B-6 content_hash feature is automatically brought up to
    date without requiring a manual ``flask db upgrade`` step.
    """
    try:
        engine = db.engine
        with engine.connect() as conn:
            # Only attempt the migration for SQLite databases.
            if not str(engine.url).startswith("sqlite"):
                return
            # PRAGMA table_info returns one row per column.
            rows = conn.execute(text("PRAGMA table_info(datasets)")).fetchall()
            existing_columns = {row[1] for row in rows}  # column name is index 1
            if rows and "content_hash" not in existing_columns:
                log.info(
                    "[dataset_service] datasets.content_hash column missing — "
                    "applying auto-migration (ALTER TABLE)"
                )
                conn.execute(
                    text(
                        "ALTER TABLE datasets "
                        "ADD COLUMN content_hash VARCHAR(64) DEFAULT NULL"
                    )
                )
                # SQLite supports index creation outside batch mode.
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_datasets_content_hash "
                        "ON datasets (content_hash)"
                    )
                )
                conn.commit()
                log.info(
                    "[dataset_service] Auto-migration complete: "
                    "datasets.content_hash column added."
                )
    except Exception as exc:  # pragma: no cover
        # Non-fatal: log and continue.  The column will still be missing and
        # uploads will fail with a clear 500, but the rest of the app works.
        log.error(
            "[dataset_service] Auto-migration of content_hash failed: %s", exc
        )


def _upload_root() -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"]).resolve()


def _processed_root() -> Path:
    return Path(current_app.config["PROCESSED_DATA_FOLDER"]).resolve()


def _is_managed_upload(path: Path) -> bool:
    try:
        path.resolve().relative_to(_upload_root())
        return True
    except ValueError:
        return False


def _is_managed_processed(path: Path) -> bool:
    try:
        path.resolve().relative_to(_processed_root())
        return True
    except ValueError:
        return False


def managed_upload_path(dataset: Dataset) -> Path:
    """Return a raw upload path only when it remains inside managed storage."""
    path = Path(dataset.file_path)
    if not _is_managed_upload(path) or not path.is_file():
        raise DatasetError("Dataset file is unavailable", 404)
    return path


def _extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def _read_dataset(path: Path, extension: str, *, preview_rows: int | None = None) -> pd.DataFrame:
    try:
        if extension == "csv":
            return pd.read_csv(path, nrows=preview_rows)
        elif extension == "parquet":
            df = pd.read_parquet(path)
            return df.head(preview_rows) if preview_rows else df
        return pd.read_excel(path, nrows=preview_rows)
    except (OSError, ValueError, UnicodeDecodeError, zipfile.BadZipFile, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DatasetError("Dataset file is empty or could not be read", 400) from exc


def _validate_upload(file: FileStorage | None, domain: str | None) -> tuple[str, str, str]:
    if file is None:
        raise DatasetError("A dataset file is required", errors={"file": ["This field is required."]})
    filename = secure_filename(file.filename or "")
    if not filename:
        raise DatasetError("A filename is required", errors={"file": ["Filename must not be empty."]})
    extension = _extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise DatasetError("Unsupported file format", 415, {"file": ["Only .csv, .xlsx, and .parquet files are supported."]})
    if file.mimetype and file.mimetype not in ALLOWED_MIMETYPES[extension]:
        raise DatasetError("Invalid file content type", 415)
    normalized_domain = (domain or "").strip().lower()
    if normalized_domain not in ALLOWED_DOMAINS:
        raise DatasetError("Invalid dataset domain", errors={"domain": ["Must be either retail or electricity."]})
    return filename, extension, normalized_domain


def _safe_table_records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.where(pd.notna(frame), None).to_json(orient="records", date_format="iso"))


def _format_timestamp(value: str) -> str:
    return value if "T" in value else f"{value.replace(' ', 'T')}Z"


def upload_dataset(file: FileStorage | None, *, domain: str | None, user_id: str) -> Dataset:
    """Save a sanitized upload and create its dataset metadata atomically where possible."""
    import hashlib
    filename, extension, normalized_domain = _validate_upload(file, domain)
    upload_root = _upload_root()
    stored_path = upload_root / f"{uuid.uuid4().hex}.{extension}"

    try:
        upload_root.mkdir(parents=True, exist_ok=True)
        file.save(stored_path)
        if not stored_path.is_file() or stored_path.stat().st_size == 0:
            raise DatasetError("Dataset file is empty", 400)

        file_bytes = stored_path.read_bytes()
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        # Deduplication check: return existing dataset record if exact content hash exists
        existing = db.session.scalar(
            db.select(Dataset).where(
                Dataset.user_id == user_id,
                Dataset.domain == normalized_domain,
                Dataset.content_hash == content_hash,
            )
        )
        if existing is not None:
            stored_path.unlink(missing_ok=True)
            return existing

        frame = _read_dataset(stored_path, extension)
        if len(frame.columns) == 0:
            raise DatasetError("Dataset file is empty or could not be read", 400)
        dataset = Dataset(
            dataset_id=str(uuid.uuid4()),
            user_id=user_id,
            filename=filename,
            domain=normalized_domain,
            file_path=str(stored_path),
            file_size_bytes=stored_path.stat().st_size,
            row_count=len(frame.index),
            column_count=len(frame.columns),
            content_hash=content_hash,
        )
        db.session.add(dataset)
        db.session.commit()
        return dataset
    except DatasetError:
        db.session.rollback()
        stored_path.unlink(missing_ok=True)
        raise
    except (OSError, SQLAlchemyError) as exc:
        db.session.rollback()
        stored_path.unlink(missing_ok=True)
        # Surface the underlying exception type and message to aid diagnosis
        # (sanitize_client_error_message strips sensitive paths/tokens).
        exc_type = type(exc).__name__
        exc_detail = sanitize_client_error_message(str(exc))
        log.error(
            "[dataset_service] upload_dataset failed — %s: %s",
            exc_type,
            exc_detail,
        )
        raise DatasetError(
            f"Unable to store dataset ({exc_type}: {exc_detail})", 500
        ) from exc


DEMO_DATASETS = [
    {
        "dataset_id": "demo_retail",
        "filename": "Retail Sales Demo",
        "domain": "retail",
        "row_count": 1000,
        "status": "processed",
        "created_at": "2026-01-01T00:00:00Z",
        "features_ready": True,
        "features_dir": "ml_engine/data/features/retail/features_1c0c92ff-264e-4398-b83c-e0e082b956b4",
        "is_demo": True,
    },
    {
        "dataset_id": "demo_electricity",
        "filename": "Electricity Load Demo",
        "domain": "electricity",
        "row_count": 2160,
        "status": "processed",
        "created_at": "2026-01-01T00:00:00Z",
        "features_ready": True,
        "features_dir": "ml_engine/data/features/electricity/features_23889e57-aa6d-4a50-930c-874f6a963a25",
        "is_demo": True,
    },
]


def list_datasets(user_id: str | None, *, domain: str | None, limit: int, offset: int) -> tuple[int, list[dict]]:
    if domain is not None and domain.lower() not in ALLOWED_DOMAINS:
        raise DatasetError("Invalid dataset domain", errors={"domain": ["Must be either retail or electricity."]})
    if limit < 1 or limit > 100 or offset < 0:
        raise DatasetError("Invalid pagination parameters", errors={"limit": ["Must be between 1 and 100."], "offset": ["Must be zero or greater."]})

    normalized_domain = domain.lower() if domain is not None else None
    items: list[dict] = []

    for demo in DEMO_DATASETS:
        if normalized_domain is None or demo["domain"] == normalized_domain:
            items.append(dict(demo))

    if user_id:
        try:
            statement = db.select(Dataset).where(Dataset.user_id == user_id)
            if normalized_domain is not None:
                statement = statement.where(Dataset.domain == normalized_domain)
            db_datasets = list(db.session.scalars(statement.order_by(Dataset.created_at.desc())))
            for dataset in db_datasets:
                items.append(list_item(dataset))
        except SQLAlchemyError as exc:
            raise DatasetError("Unable to retrieve datasets", 500) from exc

    total = len(items)
    paginated = items[offset:offset + limit]
    return total, paginated


def owned_dataset(dataset_id: str, user_id: str | None = None) -> Dataset | dict:
    if dataset_id in ("demo_retail", "demo_electricity"):
        for demo in DEMO_DATASETS:
            if demo["dataset_id"] == dataset_id:
                return dict(demo)
    try:
        stmt = db.select(Dataset).where(Dataset.dataset_id == dataset_id)
        if user_id and user_id != "demo_user":
            stmt = stmt.where(Dataset.user_id == user_id)
        dataset = db.session.scalar(stmt)
    except SQLAlchemyError as exc:
        raise DatasetError("Unable to retrieve dataset", 500) from exc
    if dataset is None:
        raise DatasetError(f"Dataset '{dataset_id}' not found", 404)
    return dataset


def dataset_detail(dataset: Dataset | dict, preview_rows: int) -> dict:
    if preview_rows < 1 or preview_rows > 100:
        raise DatasetError("Invalid preview_rows parameter", errors={"preview_rows": ["Must be between 1 and 100."]})
    if isinstance(dataset, dict):
        return {
            "dataset_id": dataset["dataset_id"],
            "filename": dataset["filename"],
            "domain": dataset["domain"],
            "file_size_bytes": 0,
            "row_count": dataset["row_count"],
            "column_count": 0,
            "status": dataset["status"],
            "features_ready": dataset.get("features_ready", True),
            "features_dir": dataset.get("features_dir"),
            "is_demo": True,
            "columns": [],
            "preview": [],
        }
    path = managed_upload_path(dataset)
    frame = _read_dataset(path, _extension(dataset.filename), preview_rows=preview_rows)
    detail = {
        "dataset_id": dataset.dataset_id,
        "filename": dataset.filename,
        "domain": dataset.domain,
        "file_size_bytes": dataset.file_size_bytes,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "status": dataset.status,
        "columns": [str(column) for column in frame.columns],
        "preview": _safe_table_records(frame),
    }
    features_dir = None
    features_ready = False
    if dataset.preprocessing_runs:
        latest = sorted(dataset.preprocessing_runs, key=lambda r: str(r.processed_at), reverse=True)[0]
        if latest.lags_generated_json:
            try:
                lineage = json.loads(latest.lags_generated_json)
                if isinstance(lineage, dict):
                    features_dir = lineage.get("features_dir")
                    features_ready = bool(lineage.get("features_ready", False))
            except json.JSONDecodeError:
                pass
    detail["features_dir"] = features_dir
    detail["features_ready"] = features_ready
    return detail


def delete_dataset(dataset: Dataset) -> None:
    """Remove a dataset's owned metadata and managed raw file without arbitrary deletion."""
    raw_path = Path(dataset.file_path)
    processed_paths: list[Path] = []
    feature_dirs: list[Path] = []
    try:
        # These dependent records have no ORM delete cascade in the existing schema.
        preprocessing = list(db.session.scalars(db.select(DatasetPreprocessing).where(DatasetPreprocessing.dataset_id == dataset.dataset_id)))
        for record in preprocessing:
            processed_paths.append(Path(record.processed_file_path))
            if record.lags_generated_json:
                try:
                    lineage = json.loads(record.lags_generated_json)
                except json.JSONDecodeError:
                    lineage = None
                if isinstance(lineage, dict) and lineage.get("features_dir"):
                    feature_dirs.append(Path(str(lineage["features_dir"])))
            db.session.delete(record)
        for validation in db.session.scalars(db.select(DatasetValidation).where(DatasetValidation.dataset_id == dataset.dataset_id)):
            db.session.delete(validation)
        db.session.delete(dataset)
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        raise DatasetError("Unable to delete dataset", 500) from exc

    if _is_managed_upload(raw_path):
        try:
            raw_path.unlink(missing_ok=True)
        except OSError as exc:
            # The record is already gone; surface a recoverable storage error.
            raise DatasetError("Dataset metadata was deleted but its file could not be removed", 500) from exc
    for processed_path in processed_paths:
        if _is_managed_processed(processed_path):
            try:
                processed_path.unlink(missing_ok=True)
            except OSError as exc:
                raise DatasetError("Dataset metadata was deleted but a processed file could not be removed", 500) from exc

    # Best-effort cleanup of managed forecasting feature splits (no migrations).
    import shutil

    project_root = Path(__file__).resolve().parents[3]
    features_root = Path(current_app.config.get(
        "FEATURES_DATA_FOLDER",
        str(project_root / "ml_engine" / "data" / "features"),
    )).resolve()
    for features_dir in feature_dirs:
        path = features_dir if features_dir.is_absolute() else (project_root / features_dir).resolve()
        try:
            path.relative_to(features_root)
        except ValueError:
            continue
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def _schema_issues(frame: pd.DataFrame, domain: str, strict_mode: bool) -> tuple[list[dict], dict[str, Any]]:
    from ml_engine.preprocessing.feature_discovery import discover_features_and_resolve_schema

    issues: list[dict] = []
    if frame.empty:
        if strict_mode:
            issues.append({"type": "empty_dataset", "message": "Dataset contains 0 rows."})
        return issues, {}

    schema_res = discover_features_and_resolve_schema(frame, domain=domain)

    missing = []
    if not schema_res.time_column:
        missing.append("timestamp / date column")
    if not schema_res.target_column and not getattr(schema_res, "is_multiseries", False):
        missing.append(f"{domain} target column")

    if missing:
        issues.append({"type": "missing_required_columns", "columns": missing})
    elif not schema_res.is_valid:
        for err in schema_res.validation_issues:
            issues.append({"type": "invalid_schema", "message": err})

    schema_metadata = {
        "target": schema_res.target_column,
        "time_column": schema_res.time_column,
        "target_column": schema_res.target_column,
        "canonical_target": schema_res.canonical_target,
        "target_candidates": [c["name"] for c in schema_res.target_candidates],
        "is_ambiguous_target": schema_res.is_ambiguous_target,
        "ambiguous_targets": schema_res.ambiguous_targets,
        "column_roles": schema_res.column_roles,
        "is_multiseries": getattr(schema_res, "is_multiseries", False),
        "dataset_format": getattr(schema_res, "dataset_format", "tabular_single"),
        "detected_geographic_series": getattr(schema_res, "detected_geographic_series", []),
        "geographic_level": getattr(schema_res, "geographic_level", None),
        "parent_level": getattr(schema_res, "parent_level", None),
        "aggregation_coverage": getattr(schema_res, "aggregation_coverage", {}),
    }

    return issues, schema_metadata


def validate_dataset(dataset: Dataset, payload: object) -> tuple[DatasetValidation, dict]:
    """Persist a data-quality audit; invalid schema audits are retained for troubleshooting."""
    body = payload if isinstance(payload, dict) else {}
    schema_type = body.get("schema_type", f"{dataset.domain}_v1")
    strict_mode = body.get("strict_mode", True)
    if schema_type != f"{dataset.domain}_v1" or not isinstance(strict_mode, bool):
        raise DatasetError("Invalid validation request", errors={"schema_type": [f"Must be {dataset.domain}_v1."], "strict_mode": ["Must be a boolean."]})

    path = Path(dataset.file_path)
    issues: list[dict]
    missing_summary: dict[str, int]
    duplicate_rows = 0
    schema_meta: dict[str, Any] = {}
    try:
        if not _is_managed_upload(path) or not path.is_file():
            issues = [{"type": "file_unavailable"}]
            missing_summary = {}
        else:
            frame = _read_dataset(path, _extension(dataset.filename))
            missing_summary = {str(column): int(count) for column, count in frame.isna().sum().items()}
            duplicate_rows = int(frame.duplicated().sum())
            issues, schema_meta = _schema_issues(frame, dataset.domain, strict_mode)
    except DatasetError as exc:
        issues = [{"type": "unreadable_file", "message": exc.message}]
        missing_summary = {}

    valid = not issues
    validation = DatasetValidation(
        validation_id=str(uuid.uuid4()),
        dataset_id=dataset.dataset_id,
        is_valid=valid,
        missing_row_count=sum(missing_summary.values()),
        duplicate_row_count=duplicate_rows,
        schema_issues_json=json.dumps(issues) if issues else None,
    )
    try:
        db.session.add(validation)
        if valid:
            dataset.status = "validated"
        db.session.commit()
    except SQLAlchemyError as exc:
        db.session.rollback()
        raise DatasetError("Unable to save validation results", 500) from exc

    logs = ["Schema check passed"] if valid else ["Schema validation failed"]
    if schema_meta.get("is_ambiguous_target"):
        logs.append(f"Multiple candidate targets detected: {schema_meta['ambiguous_targets']}. Confirm target selection if needed.")
    logs.extend(f"{count} missing values flagged in {column}" for column, count in missing_summary.items() if count)
    return validation, {
        "dataset_id": dataset.dataset_id,
        "is_valid": valid,
        "missing_value_summary": missing_summary,
        "duplicate_rows": duplicate_rows,
        "validation_logs": logs,
        **schema_meta,
    }


def upload_response(dataset: Dataset) -> dict:
    return {
        "dataset_id": dataset.dataset_id,
        "filename": dataset.filename,
        "domain": dataset.domain,
        "file_size_bytes": dataset.file_size_bytes,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "status": dataset.status,
        "content_hash": getattr(dataset, "content_hash", None),
        "upload_timestamp": _format_timestamp(dataset.created_at),
    }


def list_item(dataset: Dataset) -> dict:
    features_dir = None
    features_ready = False
    if dataset.preprocessing_runs:
        latest = sorted(dataset.preprocessing_runs, key=lambda r: str(r.processed_at), reverse=True)[0]
        if latest.lags_generated_json:
            try:
                lineage = json.loads(latest.lags_generated_json)
                if isinstance(lineage, dict):
                    features_dir = lineage.get("features_dir")
                    features_ready = bool(lineage.get("features_ready", False))
            except json.JSONDecodeError:
                pass
    return {
        "dataset_id": dataset.dataset_id,
        "filename": dataset.filename,
        "domain": dataset.domain,
        "row_count": dataset.row_count,
        "content_hash": getattr(dataset, "content_hash", None),
        "status": dataset.status,
        "created_at": _format_timestamp(dataset.created_at),
        "features_ready": features_ready,
        "features_dir": features_dir,
    }


def profile_dataset_service(dataset: Dataset | dict, domain: str | None = None, requested_target: str | None = None) -> dict:
    from ml_engine.preprocessing.intelligence import profile_dataset

    if isinstance(dataset, dict):
        demo_domain = domain or dataset.get("domain", "retail")
        features_dir_rel = dataset.get("features_dir")
        if not features_dir_rel:
            raise DatasetError("Dataset feature path is unavailable", 404)
        project_root = Path(__file__).resolve().parents[3]
        train_path = (project_root / features_dir_rel / "train.parquet").resolve()
        if not train_path.exists():
            raise DatasetError("Demo dataset artifact file is unavailable", 404)
        frame = pd.read_parquet(train_path)
        profile = profile_dataset(frame, domain=demo_domain, requested_target=requested_target)
        profile["dataset_id"] = dataset["dataset_id"]
        profile["filename"] = dataset["filename"]
        profile["is_demo"] = True
        return profile

    effective_domain = domain or dataset.domain
    path = managed_upload_path(dataset)
    frame = _read_dataset(path, _extension(dataset.filename))
    profile = profile_dataset(frame, domain=effective_domain, requested_target=requested_target)
    profile["dataset_id"] = dataset.dataset_id
    profile["filename"] = dataset.filename
    profile["is_demo"] = False
    return profile


def get_dataset_geography_service(dataset: Dataset | dict | str, geo_filters: dict[str, str] | None = None) -> dict:
    from ml_engine.preprocessing.intelligence import get_geographic_hierarchy

    if isinstance(dataset, str):
        dataset = owned_dataset(dataset)

    if isinstance(dataset, dict):
        features_dir_rel = dataset.get("features_dir")
        if not features_dir_rel:
            raise DatasetError("Dataset feature path is unavailable", 404)
        project_root = Path(__file__).resolve().parents[3]
        train_path = (project_root / features_dir_rel / "train.parquet").resolve()
        if not train_path.exists():
            raise DatasetError("Demo dataset artifact file is unavailable", 404)
        frame = pd.read_parquet(train_path)
        geo_meta = get_geographic_hierarchy(frame, geo_filters)
        geo_meta["dataset_id"] = dataset["dataset_id"]
        geo_meta["filename"] = dataset["filename"]
        return geo_meta

    path = managed_upload_path(dataset)
    frame = _read_dataset(path, _extension(dataset.filename))
    geo_meta = get_geographic_hierarchy(frame, geo_filters)
    geo_meta["dataset_id"] = dataset.dataset_id
    geo_meta["filename"] = dataset.filename
    return geo_meta
