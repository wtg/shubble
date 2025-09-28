from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

CAMPUS_TZ = ZoneInfo("America/New_York")


def get_campus_start_of_day():
    now = datetime.now(CAMPUS_TZ)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    return midnight.astimezone(timezone.utc)
