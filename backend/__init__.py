"""Backend package - re-exports from backend.flask for backward compatibility."""
from backend.flask import app
from backend.config import settings

__all__ = ["app", "settings"]
