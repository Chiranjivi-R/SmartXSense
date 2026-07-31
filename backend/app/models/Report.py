"""Generated report persistence model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..extensions import db

if TYPE_CHECKING:
    from .User import User


class GeneratedReport(db.Model):
    """Metadata for a generated PDF or CSV analytics report."""

    __tablename__ = "generated_reports"

    report_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    format: Mapped[str] = mapped_column(String(16), nullable=False, server_default=text("'pdf'"))
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[str] = mapped_column(String(30), nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    user: Mapped[User] = relationship(back_populates="generated_reports")
