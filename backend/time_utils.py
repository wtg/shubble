"""Time utility functions for timezone handling."""
from datetime import datetime, timezone

from backend.config import settings


def get_campus_start_of_day():
    """
    Get the start of the current day in campus timezone (America/New_York),
    converted to UTC.

    Returns:
        datetime: Start of the current day in campus timezone at the configured
            DAY_START time, converted to UTC.
    """
    now = datetime.now(settings.CAMPUS_TZ)
    
    # timestamp parse DAY_START
    day_start_time = datetime.strptime(settings.DAY_START, "%H:%M:%S").time()
    midnight = now.replace(hour=day_start_time.hour, minute=day_start_time.minute, second=day_start_time.second, microsecond=0)

    return midnight.astimezone(timezone.utc)
