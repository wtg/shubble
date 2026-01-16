"""ML package for shuttle tracker prediction models."""
import logging
import sys

# Get ML-specific log level from settings
try:
    from backend.config import settings
    ml_log_level_str = settings.get_log_level("ml")
    ml_log_level = logging._nameToLevel.get(ml_log_level_str.upper(), logging.INFO)
except ImportError:
    # Fallback if settings not available (e.g., during testing)
    ml_log_level = logging.INFO
    ml_log_level_str = "INFO"

# Configure logging for the ML package
logger = logging.getLogger(__name__)

# Only configure if not already configured
if not logger.handlers:
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(ml_log_level)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)
    logger.setLevel(ml_log_level)
    logger.info(f"ML logging level: {ml_log_level_str}")

__all__ = ['logger']
