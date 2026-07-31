"""Dataset, validation, and preprocessing persistence models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..extensions import db

if TYPE_CHECKING:
    from .Forecast import TrainedModel
    from .User import User


class Dataset(db.Model):
    """Uploaded source dataset metadata."""

    __tablename__ = "datasets"
    __table_args__ = (Index("idx_datasets_user_domain", "user_id", "domain"),)

    dataset_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[str] = mapped_column(String(32), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    column_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'uploaded'"))
    created_at: Mapped[str] = mapped_column(String(30), nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    user: Mapped[User] = relationship(back_populates="datasets")
    validations: Mapped[list[DatasetValidation]] = relationship(back_populates="dataset")
    preprocessing_runs: Mapped[list[DatasetPreprocessing]] = relationship(back_populates="dataset")
    trained_models: Mapped[list[TrainedModel]] = relationship(back_populates="dataset")


class DatasetValidation(db.Model):
    """Dataset quality and schema-validation result."""

    __tablename__ = "dataset_validation"

    validation_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=False, index=True)
    is_valid: Mapped[bool] = mapped_column(nullable=False)
    missing_row_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    duplicate_row_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    schema_issues_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    validated_at: Mapped[str] = mapped_column(String(30), nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    dataset: Mapped[Dataset] = relationship(back_populates="validations")


class DatasetPreprocessing(db.Model):
    """Dataset transformation and feature-generation metadata."""

    __tablename__ = "dataset_preprocessing"

    preprocessing_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=False, index=True)
    imputation_strategy: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'linear'"))
    scaler_type: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'standard'"))
    lags_generated_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    processed_file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    processed_at: Mapped[str] = mapped_column(String(30), nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    dataset: Mapped[Dataset] = relationship(back_populates="preprocessing_runs")
