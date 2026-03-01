"""Configuration for the Bubble AI agent."""
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BubbleSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    DATABASE_URL: str

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Google Gemini API key
    GEMINI_API_KEY: str

    # Comma-separated list of URLs to fetch as context for announcement generation
    BUBBLE_SOURCES: str = ""

    # Gemini model to use for generation
    BUBBLE_MODEL: str = "gemini-2.0-flash-lite"

    # How often to regenerate announcements (seconds); default 1 hour
    BUBBLE_INTERVAL: int = 3600

    # Logging
    LOG_LEVEL: str = "info"
    BUBBLE_LOG_LEVEL: Optional[str] = None

    @field_validator("DATABASE_URL")
    @classmethod
    def fix_database_url(cls, v: str) -> str:
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql://", 1)
        return v

    @property
    def sources_list(self) -> list[str]:
        return [s.strip() for s in self.BUBBLE_SOURCES.split(",") if s.strip()]

    def get_log_level(self) -> str:
        if self.BUBBLE_LOG_LEVEL:
            return self.BUBBLE_LOG_LEVEL.lower()
        return self.LOG_LEVEL.lower()


settings = BubbleSettings()
