from datetime import datetime
from . import create_app, db
from .models import GeofenceEvent, VehicleLocation  # adjust import

def remove_events_before_today():
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    deleted_geofence = (
        db.session.query(GeofenceEvent)
        .filter(GeofenceEvent.event_time >= today_start)
        .delete(synchronize_session=False)
    )

    deleted_locations = (
        db.session.query(VehicleLocation)
        .filter(VehicleLocation.timestamp >= today_start)
        .delete(synchronize_session=False)
    )

    db.session.commit()

    print(f"Deleted {deleted_geofence} GeofenceEvents and {deleted_locations} VehicleLocations before {today_start}.")

if __name__ == "__main__":
    app = create_app()
    with app.app_context():  # <--- important
        remove_events_before_today()
