"""Model registry, evaluation, and forecast persistence models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Float, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..extensions import db

if TYPE_CHECKING:
    from .Dataset import Dataset
    from .Explainability import ExplainabilityReport
    from .User import User


class TrainedModel(db.Model):
    """Registered forecasting model artifact."""

    __tablename__ = "trained_models"
    __table_args__ = (Index("idx_trained_models_domain_active", "domain", "is_active"),)

    model_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    algorithm: Mapped[str] = mapped_column(String(64), nullable=False)
    domain: Mapped[str] = mapped_column(String(32), nullable=False)
    hyperparameters_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("1"))
    created_at: Mapped[str] = mapped_column(String(30), nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    dataset: Mapped[Dataset] = relationship(back_populates="trained_models")
    user: Mapped[User] = relationship(back_populates="trained_models")
    evaluations: Mapped[list[ModelEvaluation]] = relationship(back_populates="model")
    forecast_jobs: Mapped[list[ForecastJob]] = relationship(back_populates="model")
    explainability_reports: Mapped[list[ExplainabilityReport]] = relationship(back_populates="model")


class ModelEvaluation(db.Model):
    """Backtesting metrics for a trained model."""

    __tablename__ = "model_evaluation"

    eval_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("trained_models.model_id"), nullable=False, index=True)
    mae: Mapped[float] = mapped_column(Float, nullable=False)
    rmse: Mapped[float] = mapped_column(Float, nullable=False)
    mape: Mapped[float] = mapped_column(Float, nullable=False)
    wape: Mapped[float | None] = mapped_column(Float, nullable=True)
    r2_score: Mapped[float] = mapped_column(Float, nullable=False)
    evaluated_at: Mapped[str] = mapped_column(String(30), nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    model: Mapped[TrainedModel] = relationship(back_populates="evaluations")


class ForecastJob(db.Model):
    """Forecast execution job and its lifecycle state."""

    __tablename__ = "forecast_jobs"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("trained_models.model_id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'pending'"))
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[str | None] = mapped_column(String(30), nullable=True)
    completed_at: Mapped[str | None] = mapped_column(String(30), nullable=True)

    model: Mapped[TrainedModel] = relationship(back_populates="forecast_jobs")
    user: Mapped[User] = relationship(back_populates="forecast_jobs")
    results: Mapped[list[ForecastResult]] = relationship(back_populates="job")


class ForecastResult(db.Model):
    """A single time-series point produced by a forecast job."""

    __tablename__ = "forecast_results"
    __table_args__ = (Index("idx_forecast_results_job_date", "job_id", "timestamp"),)

    result_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("forecast_jobs.job_id"), nullable=False)
    timestamp: Mapped[str] = mapped_column(String(30), nullable=False)
    actual_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    predicted_value: Mapped[float] = mapped_column(Float, nullable=False)
    lower_ci: Mapped[float | None] = mapped_column(Float, nullable=True)
    upper_ci: Mapped[float | None] = mapped_column(Float, nullable=True)

    job: Mapped[ForecastJob] = relationship(back_populates="results")
