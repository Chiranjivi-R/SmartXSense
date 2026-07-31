"""Explainability report persistence model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..extensions import db

if TYPE_CHECKING:
    from .Forecast import TrainedModel


class ExplainabilityReport(db.Model):
    """SHAP or LIME attribution report generated for a trained model."""

    __tablename__ = "explainability_reports"
    __table_args__ = (Index("idx_explainability_model_method", "model_id", "xai_method"),)

    report_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("trained_models.model_id"), nullable=False)
    xai_method: Mapped[str] = mapped_column(String(16), nullable=False)
    base_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    attributions_json: Mapped[str] = mapped_column(Text, nullable=False)
    plot_artifact_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    generated_at: Mapped[str] = mapped_column(String(30), nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    model: Mapped[TrainedModel] = relationship(back_populates="explainability_reports")
