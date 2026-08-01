"""Dataset upload, access, storage, and structural-validation business logic."""

from __future__ import annotations

import json
import os
import uuid
import zipfile
from pathlib import Path

import pandas as pd
from flask import current_app
from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from ..extensions import db
from ..models import Dataset, DatasetPreprocessing, DatasetValidation


ALLOWED_EXTENSIONS = {"csv", "xlsx"}
ALLOWED_MIMETYPES = {
    "csv": {"text/csv", "application/csv", "text/plain"},
    "xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
}
ALLOWED_DOMAINS = {"retail", "electricity"}
REQUIRED_COLUMNS = {
    "retail": {"timestamp", "store_id", "item_id", "sales"},
    "electricity": {"timestamp", "grid_load_mw"},
}


class DatasetError(Exception):
    """Expected API-facing dataset failure."""

    def __init__(self, message: str, code: int = 400, errors: dict | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.errors = errors


def error_payload(error: DatasetError) -> dict:
    payload = {"status": "error", "code": error.code, "message": error.message}
    if error.errors:
        payload["errors"] = error.errors
    return payload


def _upload_root() -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"]).resolve()


def _is_managed_upload(path: Path) -> bool:
    try:
        path.resolve().relative_to(_upload_root())
        return True
    except ValueError:
        return False


def _extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


def _read_dataset(path: Path, extension: str, *, preview_rows: int | None = None) -> pd.DataFrame:
    try:
        if extension == "csv":
            return pd.read_csv(path, nrows=preview_rows)
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
        raise DatasetError("Unsupported file format", 415, {"file": ["Only .csv and .xlsx files are supported."]})
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
    filename, extension, normalized_domain = _validate_upload(file, domain)
    upload_root = _upload_root()
    stored_path = upload_root / f"{uuid.uuid4().hex}.{extension}"

    try:
        upload_root.mkdir(parents=True, exist_ok=True)
        file.save(stored_path)
        if not stored_path.is_file() or stored_path.stat().st_size == 0:
            raise DatasetError("Dataset file is empty", 400)
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
        raise DatasetError("Unable to store dataset", 500) from exc


def list_datasets(user_id: str, *, domain: str | None, limit: int, offset: int) -> tuple[int, list[Dataset]]:
    if domain is not None and domain.lower() not in ALLOWED_DOMAINS:
        raise DatasetError("Invalid dataset domain", errors={"domain": ["Must be either retail or electricity."]})
    if limit < 1 or limit > 100 or offset < 0:
        raise DatasetError("Invalid pagination parameters", errors={"limit": ["Must be between 1 and 100."], "offset": ["Must be zero or greater."]})
    try:
        statement = db.select(Dataset).where(Dataset.user_id == user_id)
        if domain is not None:
            statement = statement.where(Dataset.domain == domain.lower())
        total = db.session.scalar(db.select(func.count()).select_from(statement.subquery())) or 0
        datasets = list(db.session.scalars(statement.order_by(Dataset.created_at.desc()).limit(limit).offset(offset)))
        return total, datasets
    except SQLAlchemyError as exc:
        raise DatasetError("Unable to retrieve datasets", 500) from exc


def owned_dataset(dataset_id: str, user_id: str) -> Dataset:
    try:
        dataset = db.session.scalar(db.select(Dataset).where(Dataset.dataset_id == dataset_id, Dataset.user_id == user_id))
    except SQLAlchemyError as exc:
        raise DatasetError("Unable to retrieve dataset", 500) from exc
    if dataset is None:
        # Do not disclose whether another user's dataset exists.
        raise DatasetError("Dataset not found", 404)
    return dataset


def dataset_detail(dataset: Dataset, preview_rows: int) -> dict:
    if preview_rows < 1 or preview_rows > 100:
        raise DatasetError("Invalid preview_rows parameter", errors={"preview_rows": ["Must be between 1 and 100."]})
    path = Path(dataset.file_path)
    if not _is_managed_upload(path) or not path.is_file():
        raise DatasetError("Dataset file is unavailable", 404)
    frame = _read_dataset(path, _extension(dataset.filename), preview_rows=preview_rows)
    return {
        "dataset_id": dataset.dataset_id,
        "filename": dataset.filename,
        "domain": dataset.domain,
        "columns": [str(column) for column in frame.columns],
        "preview": _safe_table_records(frame),
    }


def delete_dataset(dataset: Dataset) -> None:
    """Remove a dataset's owned metadata and managed raw file without arbitrary deletion."""
    raw_path = Path(dataset.file_path)
    try:
        # These dependent records have no ORM delete cascade in the existing schema.
        preprocessing = list(db.session.scalars(db.select(DatasetPreprocessing).where(DatasetPreprocessing.dataset_id == dataset.dataset_id)))
        for record in preprocessing:
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


def _schema_issues(frame: pd.DataFrame, domain: str, strict_mode: bool) -> list[dict]:
    columns = {str(column).strip().lower() for column in frame.columns}
    issues: list[dict] = []
    missing = sorted(REQUIRED_COLUMNS[domain] - columns)
    if missing:
        issues.append({"type": "missing_required_columns", "columns": missing})
    if "timestamp" in columns:
        source_name = next(str(column) for column in frame.columns if str(column).strip().lower() == "timestamp")
        if pd.to_datetime(frame[source_name], errors="coerce").isna().all():
            issues.append({"type": "invalid_timestamp_column", "column": source_name})
    target = "sales" if domain == "retail" else "grid_load_mw"
    if target in columns:
        source_name = next(str(column) for column in frame.columns if str(column).strip().lower() == target)
        if pd.to_numeric(frame[source_name], errors="coerce").isna().all():
            issues.append({"type": "invalid_numeric_column", "column": source_name})
    if strict_mode and frame.empty:
        issues.append({"type": "empty_dataset"})
    return issues


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
    try:
        if not _is_managed_upload(path) or not path.is_file():
            issues = [{"type": "file_unavailable"}]
            missing_summary = {}
        else:
            frame = _read_dataset(path, _extension(dataset.filename))
            missing_summary = {str(column): int(count) for column, count in frame.isna().sum().items()}
            duplicate_rows = int(frame.duplicated().sum())
            issues = _schema_issues(frame, dataset.domain, strict_mode)
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
    logs.extend(f"{count} missing values flagged in {column}" for column, count in missing_summary.items() if count)
    return validation, {
        "dataset_id": dataset.dataset_id,
        "is_valid": valid,
        "missing_value_summary": missing_summary,
        "duplicate_rows": duplicate_rows,
        "validation_logs": logs,
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
        "upload_timestamp": _format_timestamp(dataset.created_at),
    }


def list_item(dataset: Dataset) -> dict:
    return {
        "dataset_id": dataset.dataset_id,
        "filename": dataset.filename,
        "domain": dataset.domain,
        "row_count": dataset.row_count,
        "status": dataset.status,
        "created_at": _format_timestamp(dataset.created_at),
    }
