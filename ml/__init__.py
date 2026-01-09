"""ML package for shuttle tracker prediction models."""
import logging
import sys

# Configure logging for the ML package
logger = logging.getLogger(__name__)

# Only configure if not already configured
if not logger.handlers:
    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

__all__ = ['logger']
