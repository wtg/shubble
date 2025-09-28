from flask import Flask, jsonify, request, send_from_directory
from threading import Thread, Lock
import time
import os
import logging
from .shuttle import Shuttle, ShuttleState
from datetime import datetime, date
from server.models import Vehicle, GeofenceEvent, VehicleLocation
from server.config import Config
from sqlalchemy import func, and_
from flask_sqlalchemy import SQLAlchemy
import numpy as np

shuttles = {}
shuttle_counter = 1
shuttle_lock = Lock()

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="../test-client/dist", static_url_path="")
app.config.from_object(Config)

db = SQLAlchemy()
db.init_app(app)

# setup function to populate the shuttles dict, same as in server/routes
def setup():
    # Start of today for filtering today's geofence events
    start_of_today = datetime.combine(date.today(), datetime.min.time())

    # Subquery: latest geofence event today per vehicle
    # Returns a query result of (vehicle_id, event_time)
    latest_geofence_events = db.session.query(
        GeofenceEvent.vehicle_id,
        func.max(GeofenceEvent.event_time).label('latest_time')
    ).filter(
        GeofenceEvent.event_time >= start_of_today
    ).group_by(GeofenceEvent.vehicle_id).subquery()

    # Join to get full geofence event rows where event is geofenceEntry
    # Returns a query result of (vehicle_id, event_time, ...geofence fields including event_type)
    geofence_entries = db.session.query(GeofenceEvent.vehicle_id).join(
        latest_geofence_events,
        and_(
            GeofenceEvent.vehicle_id == latest_geofence_events.c.vehicle_id,
            GeofenceEvent.event_time == latest_geofence_events.c.latest_time
        )
    ).filter(GeofenceEvent.event_type == 'geofenceEntry').subquery()

    # Subquery: latest vehicle location per vehicle
    # Returns a query result of (vehicle_id, location_time)
    latest_locations = db.session.query(
        VehicleLocation.vehicle_id,
        func.max(VehicleLocation.timestamp).label('latest_time')
    ).filter(
        VehicleLocation.vehicle_id.in_(db.session.query(geofence_entries.c.vehicle_id))
    ).group_by(VehicleLocation.vehicle_id).subquery()

    # Join to get full location and vehicle info for vehicles in geofence
    results = db.session.query(VehicleLocation, Vehicle).join(
        latest_locations,
        and_(
            VehicleLocation.vehicle_id == latest_locations.c.vehicle_id,
            VehicleLocation.timestamp == latest_locations.c.latest_time
        )
    ).join(
        Vehicle, VehicleLocation.vehicle_id == Vehicle.id
    ).all()

    # extract vehicle information
    for loc, vehicle in results:
        shuttles[vehicle.id] = Shuttle(vehicle.id, loc.latitude, loc.longitude)

with app.app_context():
    setup()

# --- Background Thread ---
def update_loop():
    while True:
        time.sleep(0.1)
        with shuttle_lock:
            for shuttle in shuttles.values():
                shuttle.update_state()

# Start background updater
t = Thread(target=update_loop, daemon=True)
t.start()

# --- API Routes ---
@app.route("/api/shuttles", methods=["GET"])
def list_shuttles():
    with shuttle_lock:
        return jsonify([s.to_dict() for s in shuttles.values()])

@app.route("/api/shuttles", methods=["POST"])
def create_shuttle():
    global shuttle_counter
    with shuttle_lock:
        shuttle_id = str(shuttle_counter).zfill(15)
        shuttle = Shuttle(shuttle_id)
        shuttles[shuttle_id] = shuttle
        logger.info(f"Created shuttle {shuttle_counter}")
        shuttle_counter += 1
        return jsonify(shuttle.to_dict()), 201

@app.route("/api/shuttles/<shuttle_id>/set-next-state", methods=["POST"])
def trigger_action(shuttle_id):
    next_state = request.json.get("state")
    with shuttle_lock:
        shuttle = shuttles.get(shuttle_id)
        if not shuttle:
            return {"error": "Shuttle not found"}, 404

        try:
            desired_state = ShuttleState(next_state)
        except ValueError:
            return {"error": "Invalid action"}, 400

        shuttle.set_next_state(desired_state)
        logger.info(f"Set shuttle {shuttle_id} next state to {next_state}")
        return jsonify(shuttle.to_dict())

# --- Frontend Serving ---
@app.route("/")
@app.route("/<path:path>")
def serve_frontend(path=""):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

@app.route('/fleet/vehicles/stats/feed')
def mock_feed():
    vehicle_ids = request.args.get('vehicleIds', '').split(',')
    after = request.args.get('after')

    logger.info(f'[MOCK API] Received stats request for vehicles {vehicle_ids} after={after}')

    # update timestamps
    with shuttle_lock:
        data = []
        for shuttle_id in vehicle_ids:
            if shuttle_id in shuttles:
                # add error to location
                lat, lon = shuttles[shuttle_id].location
                lat += np.random.normal(0, 0.00008)
                lon += np.random.normal(0, 0.00008)
                data.append({
                    'id': shuttle_id,
                    'name': shuttle_id[-3:],
                    'gps': [
                        {
                            'latitude': lat,
                            'longitude': lon,
                            'time': datetime.fromtimestamp(shuttles[shuttle_id].last_updated).isoformat(timespec='seconds').replace('+00:00', 'Z'),
                            'speedMilesPerHour': shuttles[shuttle_id].speed,
                            'headingDegrees': 90,
                            'reverseGeo': {'formattedLocation': 'Test Location'}
                        }
                    ]
                })

        return jsonify({
            'data': data,
            'pagination': {
                'hasNextPage': False,
                'endCursor': 'fake-token-next'
            }
        })

if __name__ == "__main__":
    app.run(debug=True, port=4000)
