"""Application configuration loaded from environment variables."""

import os

from dotenv import load_dotenv


load_dotenv()


class Config:
    """Base configuration for the SmartXSense backend."""

    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-development-secret")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///smartxsense.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-this-jwt-development-secret")
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
