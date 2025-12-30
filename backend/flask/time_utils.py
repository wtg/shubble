"""Time utility functions for timezone handling."""
from datetime import datetime, timezone

from .config import settings


def get_campus_start_of_day():
    """
    Get the start of the current day in campus timezone (America/New_York),
    converted to UTC.

    Returns:
        datetime: Midnight in campus timezone, converted to UTC
    """
    now = datetime.now(settings.CAMPUS_TZ)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    return midnight.astimezone(timezone.utc)
