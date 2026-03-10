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

    """
    This means that, if it is currently between 00:00 and settings.DAY_START in the campus timezone, the function will return the start of the current day in UTC.
    This is a problem, because this is used as a cutoff for when to stop showing shuttles, meaning that a shuttle that is still on campus at 12:30 am will not be shown, even though it should be.
    """

    # if day_start_campus_tz is in the future, subtract one day to get the most recent day start
    if now < day_start_campus_tz:
        day_start_campus_tz = day_start_campus_tz - timedelta(days=1)

    return day_start_campus_tz.astimezone(timezone.utc)
