"""Configuration using Pydantic BaseSettings."""
import base64
from typing import Optional
from zoneinfo import ZoneInfo
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Hosting settings
    DEBUG: bool = True
    ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # CORS settings
    FRONTEND_URL: str = "http://localhost:3000"
    TEST_FRONTEND_URL: str = "http://localhost:5174"

    # Database settings
    DATABASE_URL: str

    # Redis settings
    REDIS_URL: str = "redis://localhost:6379/0"

    # Samsara API settings
    SAMSARA_SECRET_BASE64: Optional[str] = None

    # Shubble settings
    CAMPUS_TZ: ZoneInfo = ZoneInfo("America/New_York")

    @field_validator("DATABASE_URL")
    @classmethod
    def fix_database_url(cls, v: str) -> str:
        """Convert postgres:// to postgresql:// for SQLAlchemy compatibility."""
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql://", 1)
        return v

    @property
    def SAMSARA_SECRET(self) -> Optional[bytes]:
        """Decode base64 Samsara secret."""
        if self.SAMSARA_SECRET_BASE64:
            return base64.b64decode(self.SAMSARA_SECRET_BASE64.encode("utf-8"))
        return None


# Global settings instance
settings = Settings()
