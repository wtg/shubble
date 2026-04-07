"""Time utility functions for timezone handling."""
import os
from datetime import datetime, timedelta, timezone

from backend.config import settings

# Dev time offset — mirrors frontend/src/utils/devTime.ts
# Shifts all backend time to simulate a target hour in development mode.
_DEV_OFFSET: timedelta = timedelta(0)
if settings.DEPLOY_MODE == "development":
    _target_hour = int(os.environ.get("DEV_TARGET_HOUR", "14"))
    _target_minute = int(os.environ.get("DEV_TARGET_MINUTE", "0"))
    _now = datetime.now(settings.CAMPUS_TZ)
    _target = _now.replace(hour=_target_hour, minute=_target_minute, second=0, microsecond=0)
    _DEV_OFFSET = _target - _now


def dev_now(tz=None) -> datetime:
    """Returns current time shifted by the dev offset (no-op in production)."""
    if tz:
        return datetime.now(tz) + _DEV_OFFSET
    return datetime.now() + _DEV_OFFSET


def get_campus_start_of_day():
    """
    Get the start of the current day in campus timezone (America/New_York),
    converted to UTC.

    Returns:
        datetime: Midnight in campus timezone, converted to UTC
    """
    now = dev_now(settings.CAMPUS_TZ)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    return midnight.astimezone(timezone.utc)
