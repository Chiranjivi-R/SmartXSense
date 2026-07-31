"""User and session persistence models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..extensions import db

if TYPE_CHECKING:
    from .AuditLog import ActivityLog
    from .Dataset import Dataset
    from .Forecast import ForecastJob, TrainedModel
    from .Report import GeneratedReport
    from .Settings import SystemSetting


class User(db.Model):
    """System user profile."""

    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, server_default=text("'analyst'"))
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("1"))
    created_at: Mapped[str] = mapped_column(String(30), nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[str] = mapped_column(
        String(30), nullable=False, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP")
    )

    sessions: Mapped[list[UserSession]] = relationship(back_populates="user")
    datasets: Mapped[list[Dataset]] = relationship(back_populates="user")
    trained_models: Mapped[list[TrainedModel]] = relationship(back_populates="user")
    forecast_jobs: Mapped[list[ForecastJob]] = relationship(back_populates="user")
    generated_reports: Mapped[list[GeneratedReport]] = relationship(back_populates="user")
    settings: Mapped[list[SystemSetting]] = relationship(back_populates="user")
    activity_logs: Mapped[list[ActivityLog]] = relationship(back_populates="user")


class UserSession(db.Model):
    """Refresh-token session metadata for a user."""

    __tablename__ = "user_sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    refresh_token: Mapped[str] = mapped_column(String(512), nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    is_revoked: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("0"))
    expires_at: Mapped[str] = mapped_column(String(30), nullable=False)
    created_at: Mapped[str] = mapped_column(String(30), nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    user: Mapped[User] = relationship(back_populates="sessions")
