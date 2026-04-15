"""Time utility functions for timezone handling."""
from datetime import datetime, timedelta, timezone

from backend.config import settings

# Dev time offset — mirrors frontend/src/utils/devTime.ts
# Shifts all backend time to simulate a target hour in development mode.
_DEV_OFFSET: timedelta = timedelta(0)
# Dev time disabled — uncomment to re-enable:
# if settings.DEPLOY_MODE == "development":
#     _target_hour = int(os.environ.get("DEV_TARGET_HOUR", "14"))
#     _target_minute = int(os.environ.get("DEV_TARGET_MINUTE", "0"))
#     _now = datetime.now(settings.CAMPUS_TZ)
#     _target = _now.replace(hour=_target_hour, minute=_target_minute, second=0, microsecond=0)
#     _DEV_OFFSET = _target - _now


def dev_now(tz=None) -> datetime:
    """Returns current time shifted by the dev offset (no-op in production)."""
    if tz:
        return datetime.now(tz) + _DEV_OFFSET
    return datetime.now() + _DEV_OFFSET


def get_campus_start_of_day():
    """
    Get the current campus day-start timestamp in campus timezone,
    converted to UTC.

    Returns:
        datetime: The most recent DAY_START in campus timezone, converted to UTC.
    """
    now = dev_now(settings.CAMPUS_TZ)                                                                                                   
                                     
    # Use configurable DAY_START (not midnight) from settings                                                                           
    day_start_time = settings.DAY_START                                                                                                 
    day_start_campus_tz = now.replace(                                                                                                  
      hour=day_start_time.hour,
      minute=day_start_time.minute,
      second=day_start_time.second,
      microsecond=0,
    )


    # if day_start_campus_tz is in the future, subtract one day to get the most recent day start
    if now < day_start_campus_tz:
        # This works based on local tests
        day_start_campus_tz = day_start_campus_tz - timedelta(days=1)

    return day_start_campus_tz.astimezone(timezone.utc)
