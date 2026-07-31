"""System setting persistence model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..extensions import db

if TYPE_CHECKING:
    from .User import User


class SystemSetting(db.Model):
    """A user-scoped platform or dashboard setting."""

    __tablename__ = "system_settings"

    setting_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    setting_key: Mapped[str] = mapped_column(String(128), nullable=False)
    setting_value: Mapped[str] = mapped_column(String(1024), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(30), nullable=False, server_default=text("CURRENT_TIMESTAMP"))

    user: Mapped[User] = relationship(back_populates="settings")
