from datetime import datetime, timezone

from flask import current_app


def get_campus_start_of_day():
    now = datetime.now(current_app.config['CAMPUS_TZ'])
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    return midnight.astimezone(timezone.utc)
