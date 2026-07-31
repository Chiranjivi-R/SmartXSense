"""Audit log persistence model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..extensions import db

if TYPE_CHECKING:
    from .User import User


class ActivityLog(db.Model):
    """Security and operational audit event."""

    __tablename__ = "activity_logs"
    __table_args__ = (Index("idx_activity_logs_user_created", "user_id", "created_at"),)

    log_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id"), nullable=True)
    action: Mapped[str] = mapped_column(String(128), nullable=False)
    details: Mapped[str | None] = mapped_column(Text, nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    created_at: Mapped[str] = mapped_column(String(30), nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    user: Mapped[User | None] = relationship(back_populates="activity_logs")
