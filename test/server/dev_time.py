"""
DEV ONLY: Centralized time override for the test server.
Mirrors frontend/src/utils/devTime.ts so that simulated shuttle
departures align with the frontend's shifted clock.

The offset is computed once at import and applied to all datetime.now() calls
in the test server via dev_now(). Real wall-clock sleeps are unaffected —
the simulation runs at real speed, just shifted in time.
"""

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

CAMPUS_TZ = ZoneInfo("America/New_York")

# Match frontend devTime.ts: TARGET_HOUR / TARGET_MINUTE
TARGET_HOUR = int(os.environ.get("DEV_TARGET_HOUR", "14"))
TARGET_MINUTE = int(os.environ.get("DEV_TARGET_MINUTE", "0"))

# Compute offset once at module load (same logic as devTime.ts)
_now = datetime.now(CAMPUS_TZ)
_target = _now.replace(hour=TARGET_HOUR, minute=TARGET_MINUTE, second=0, microsecond=0)
# Enable the offset only when DEV_TIME_SHIFT=1 is set; otherwise the
# test server uses wall-clock time. This keeps the default behavior
# predictable (shuttles follow real-time schedule) but lets us jump
# into a specific schedule window (e.g. lunch rotation) for testing.
OFFSET: timedelta = (
    _target - _now
    if os.environ.get("DEV_TIME_SHIFT") == "1"
    else timedelta(0)
)


def dev_now(tz: ZoneInfo | timezone = CAMPUS_TZ) -> datetime:
    """Returns current time shifted by the dev offset."""
    return datetime.now(tz) + OFFSET
