"""Time utility functions for timezone handling."""
from datetime import datetime, timedelta, timezone

from backend.config import settings


def get_campus_start_of_day():
    """
    Get the current campus day-start timestamp in campus timezone,
    converted to UTC.

    Returns:
        datetime: The most recent DAY_START in campus timezone, converted to UTC.
    """
    now = datetime.now(settings.CAMPUS_TZ)
    
    # timestamp parse DAY_START
    day_start_time = datetime.strptime(settings.DAY_START, "%H:%M:%S").time()
    day_start_campus_tz = now.replace(
        hour=day_start_time.hour,
        minute=day_start_time.minute,
        second=day_start_time.second,
        microsecond=0,
    )

    # if day_start_campus_tz is in the future, subtract one day to get the most recent day start
    if now < day_start_campus_tz:
        day_start_campus_tz = day_start_campus_tz - timedelta(days=1)

    return day_start_campus_tz.astimezone(timezone.utc)
