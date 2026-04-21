"""Configuration using Pydantic BaseSettings."""

import base64
import logging
from datetime import datetime, time
from typing import Optional
from zoneinfo import ZoneInfo
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Hosting settings
    DEBUG: bool = True
    DEPLOY_MODE: str = "development"

    # Logging settings
    LOG_LEVEL: str = "info"  # Global default log level
    FASTAPI_LOG_LEVEL: Optional[str] = (
        None  # FastAPI backend logging (falls back to LOG_LEVEL)
    )
    WORKER_LOG_LEVEL: Optional[str] = None  # Worker logging (falls back to LOG_LEVEL)
    ML_LOG_LEVEL: Optional[str] = None  # ML pipeline logging (falls back to LOG_LEVEL)

    # CORS settings
    FRONTEND_URLS: str = "http://localhost:3000"
    TEST_FRONTEND_URL: str = "http://localhost:5174"

    # Database settings
    DATABASE_URL: str

    # Redis settings
    REDIS_URL: str = "redis://localhost:6379/0"

    # Samsara API secret (base64 encoded)
    # for webhook signature verification
    SAMSARA_SECRET: Optional[str] = None

    # Samsara API key
    API_KEY: Optional[str] = None

    # Shubble settings
    CAMPUS_TZ: ZoneInfo = ZoneInfo("America/New_York")

    DAY_START: time = time(0, 0, 0)  # Default to midnight, can be overridden in .env

    @field_validator("DATABASE_URL")
    @classmethod
    def fix_database_url(cls, v: str) -> str:
        """Convert postgres:// to postgresql:// for SQLAlchemy compatibility."""
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql://", 1)
        return v

    @field_validator("DAY_START", mode="before")
    @classmethod
    def parse_day_start(cls, v: str | time) -> time:
        """Parse DAY_START from env into a datetime.time object."""
        if isinstance(v, time):
            return v
        try:
            return datetime.strptime(v, "%H:%M:%S").time()
        except ValueError as exc:
            logger.error(
                f"Invalid DAY_START format: {v}. Expected HH:MM:SS. Error: {exc}"
            )
            return time(0, 0, 0)  # Default to midnight if parsing fails

    @property
    def samsara_secret_decoded(self) -> Optional[bytes]:
        """Decode base64 Samsara secret."""
        if self.SAMSARA_SECRET:
            return base64.b64decode(self.SAMSARA_SECRET.encode("utf-8"))
        raise ValueError(
            "SAMSARA_SECRET is required for webhook signature verification."
        )

    def get_log_level(self, component: str = "default") -> str:
        """
        Get the effective log level for a specific component.

        Args:
            component: One of "fastapi", "worker", "ml", or "default"

        Returns:
            The log level string (e.g., "INFO", "DEBUG", "WARNING")
        """
        component_levels = {
            "fastapi": self.FASTAPI_LOG_LEVEL.lower()
            if self.FASTAPI_LOG_LEVEL
            else None,
            "worker": self.WORKER_LOG_LEVEL.lower() if self.WORKER_LOG_LEVEL else None,
            "ml": self.ML_LOG_LEVEL.lower() if self.ML_LOG_LEVEL else None,
        }

        # Return component-specific level if set, otherwise fall back to LOG_LEVEL
        component_level = component_levels.get(component)
        return component_level if component_level else self.LOG_LEVEL.lower()


# Global settings instance
settings = Settings()
