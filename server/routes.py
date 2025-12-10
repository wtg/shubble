from flask import Blueprint, request, jsonify, send_from_directory, current_app
from . import db, cache
from .models import Vehicle, GeofenceEvent, VehicleLocation
from pathlib import Path
from sqlalchemy import func, and_
from sqlalchemy.dialects import postgresql
from datetime import datetime, date, timezone
from data.stops import Stops
from hashlib import sha256
import hmac
import logging
import json
import pandas as pd
from .services.eta_predictor import ETAPredictor
from .time_utils import get_campus_start_of_day
import numpy as np
logger = logging.getLogger(__name__)

bp = Blueprint('routes', __name__)

# --- LOAD STATIC DATA ---
# Load the inter-stop times (Historical Medians) once at startup
INTER_STOP_TIMES = {}
ROUTES_CONFIG = {}

data_dir = Path(__file__).parent.parent / '../data'
try:
    with open(data_dir / 'inter_stop_times.json') as f:
        INTER_STOP_TIMES = json.load(f)
except FileNotFoundError:
    logger.warning("inter_stop_times.json not found! Future ETAs will be inaccurate.")

try:
    with open(data_dir / 'routes.json') as f:
        ROUTES_CONFIG = json.load(f)
except FileNotFoundError:
    logger.error("routes.json not found! Cannot determine stop sequences.")

# --- HELPER FUNCTIONS ---


def _get_active_vehicles():
    """
    Reusable query logic to get vehicles currently inside the geofence.
    """
    start_of_today = get_campus_start_of_day()

    # 1. Find latest geofence event for each vehicle today
    latest_geofence_events = db.session.query(
        GeofenceEvent.vehicle_id,
        func.max(GeofenceEvent.event_time).label('latest_time')
    ).filter(
        GeofenceEvent.event_time >= start_of_today
    ).group_by(GeofenceEvent.vehicle_id).subquery()

    # 2. Filter for vehicles where the last event was an 'Entry'
    geofence_entries = db.session.query(GeofenceEvent.vehicle_id).join(
        latest_geofence_events,
        and_(
            GeofenceEvent.vehicle_id == latest_geofence_events.c.vehicle_id,
            GeofenceEvent.event_time == latest_geofence_events.c.latest_time
        )
    ).filter(GeofenceEvent.event_type == 'geofenceEntry').subquery()

    # 3. Get the most recent location ping for those vehicles
    latest_locations = db.session.query(
        VehicleLocation.vehicle_id,
        func.max(VehicleLocation.timestamp).label('latest_time')
    ).filter(
        VehicleLocation.vehicle_id.in_(db.session.query(geofence_entries.c.vehicle_id))
    ).group_by(VehicleLocation.vehicle_id).subquery()

    # 4. Return the full objects
    return db.session.query(VehicleLocation, Vehicle).join(
        latest_locations,
        and_(
            VehicleLocation.vehicle_id == latest_locations.c.vehicle_id,
            VehicleLocation.timestamp == latest_locations.c.latest_time
        )
    ).join(
        Vehicle, VehicleLocation.vehicle_id == Vehicle.id
    ).all()

def _get_ordered_stops_for_route(route_name):
    """
    Returns the list of stop IDs for a given route from routes.json.
    e.g. ['STUDENT_UNION', 'COLONIE', 'GEORGIAN', ...]
    """
    if route_name in ROUTES_CONFIG and 'STOPS' in ROUTES_CONFIG[route_name]:
        return ROUTES_CONFIG[route_name]['STOPS']
    return []

# --- ROUTES ---

@bp.route('/')
@bp.route('/schedule')
@bp.route('/about')
@bp.route('/data')
@bp.route('/map')
@bp.route('/generate-static-routes')
def serve_react():
    root_dir = Path(__file__).parent.parent / 'client' / 'dist'
    return send_from_directory(root_dir, 'index.html')

@bp.route('/api/locations', methods=['GET'])
@cache.cached(timeout=5, key_prefix="vehicle_locations")
def get_locations():
    """
    Returns the latest location for each active vehicle + ETA to the *immediate* next stop.
    """
    results = _get_active_vehicles()
    predictor = ETAPredictor() 
    response = {}
    
    for loc, vehicle in results:
        closest_distance, _, closest_route_name, polyline_index = Stops.get_closest_point(
            (loc.latitude, loc.longitude)
        )
        
        predicted_eta_seconds = None
        segment_id = None
        
        if closest_route_name: 
            segment_id = Stops.get_segment_id(closest_route_name, polyline_index)
            
            if segment_id:
                # Fetch recent history for this vehicle for the ML model
                history = VehicleLocation.query.filter_by(vehicle_id=loc.vehicle_id)\
                    .order_by(VehicleLocation.timestamp.desc())\
                    .limit(5)\
                    .all()
                
                if len(history) == 5:
                    history_df = pd.DataFrame([{
                        'latitude': h.latitude,
                        'longitude': h.longitude,
                        'heading_degrees': h.heading_degrees, 
                        'timestamp': h.timestamp,
                        # Pass speed if your model uses it
                        'speed_mph': h.speed_mph 
                    } for h in history]).sort_values('timestamp')

                    predicted_eta_seconds = predictor.predict(history_df, segment_id)

        response[loc.vehicle_id] = {
            'name': loc.name,
            'latitude': loc.latitude,
            'longitude': loc.longitude,
            'timestamp': loc.timestamp.isoformat(),
            'heading_degrees': loc.heading_degrees,
            'speed_mph': loc.speed_mph,
            'route_name': closest_route_name,
            'eta_seconds': predicted_eta_seconds,
            'segment_id': segment_id,
            'polyline_index': polyline_index,
            'formatted_location': loc.formatted_location,
            'license_plate': vehicle.license_plate,
            'vin': vehicle.vin,
        }

    return jsonify(response)

@bp.route('/api/stops/etas', methods=['GET'])
@cache.cached(timeout=10, key_prefix="all_stop_etas")
def get_all_stop_etas():
    results = _get_active_vehicles()
    best_etas = {} 
    predictor = ETAPredictor()

    for loc, vehicle in results:
        # 1. Identify where the bus is
        closest_dist, _, route_name, polyline_index = Stops.get_closest_point(
            (loc.latitude, loc.longitude)
        )
        
        if not route_name: continue

        # 2. Determine the Next Stop Target
        segment_id = Stops.get_segment_id(route_name, polyline_index)
        next_stop_id = None  # Initialize variable

        if segment_id:
            if "_To_" in segment_id:
                # "From_UNION_To_COLONIE" -> "COLONIE"
                next_stop_id = segment_id.split("_To_")[1]
            elif segment_id.startswith("AT_"):
                # "AT_UNION" -> Look up what comes after UNION
                current_at = segment_id.replace("AT_", "")
                _, public_stops = Stops.get_route_sequence(route_name)
                if current_at in public_stops:
                    idx = public_stops.index(current_at)
                    next_stop_id = public_stops[(idx + 1) % len(public_stops)]

        if not next_stop_id: continue

        # 3. Calculate Prediction (with "Arrival Clamp")
        predicted_seconds = None
        
        # Get coordinates to check if we are basically already there
        stop_lat, stop_lon = Stops.get_stop_coords(route_name, next_stop_id)
        
        if stop_lat:
            # Calculate distance in meters (approx)
            # 1 deg lat = 111,139 meters
            dist_sq = (loc.latitude - stop_lat)**2 + (loc.longitude - stop_lon)**2
            dist_meters = np.sqrt(dist_sq) * 111139
            
            # CLAMP: If within 40m, force 0 seconds
            if dist_meters < 40:
                predicted_seconds = 0
        
        # If not clamped (or coords missing), use ML model
        if predicted_seconds is None:
            history = VehicleLocation.query.filter_by(vehicle_id=loc.vehicle_id)\
                .order_by(VehicleLocation.timestamp.desc())\
                .limit(5).all()
            
            if len(history) == 5:
                history_df = pd.DataFrame([{
                    'latitude': h.latitude,
                    'longitude': h.longitude,
                    'heading_degrees': h.heading_degrees, 
                    'timestamp': h.timestamp
                } for h in history]).sort_values('timestamp')
                
                raw_pred = predictor.predict(history_df, segment_id)
                
                # SANITY CHECK: If model says > 20 mins (1200s), ignore it
                if raw_pred and (raw_pred < 0 or raw_pred > 1200):
                    predicted_seconds = 60
                else:
                    predicted_seconds = raw_pred

        if predicted_seconds is None: continue

        # 4. Propagate this ETA to all subsequent stops
        
        # Get full route info (including ghosts for calculation)
        all_stops, public_stops = Stops.get_route_sequence(route_name)
        
        if not all_stops: continue

        # Initialize accumulator
        current_eta = predicted_seconds

        # Find where we are in the FULL chain
        try:
            start_index = all_stops.index(next_stop_id)
        except ValueError:
            continue

        # Loop through the route (wrapping around)
        num_stops = len(all_stops)
        for k in range(num_stops):
            i = (start_index + k) % num_stops
            curr_stop = all_stops[i]
            
            # A. SAVE: If this is a real public stop, record the ETA
            if curr_stop in public_stops:
                # Use int() to keep JSON clean
                if curr_stop not in best_etas or current_eta < best_etas[curr_stop]:
                    best_etas[curr_stop] = int(current_eta)

            # B. ACCUMULATE: Add time to get to the next stop
            next_i = (i + 1) % num_stops
            next_stop = all_stops[next_i]
            
            travel_time = 30 # Default for unknown/ghost links
            if curr_stop in INTER_STOP_TIMES and next_stop in INTER_STOP_TIMES[curr_stop]:
                travel_time = INTER_STOP_TIMES[curr_stop][next_stop]
            
            # Add dwell time only for real stops
            dwell = 15 if curr_stop in public_stops else 0
            
            current_eta += (travel_time + dwell)

    return jsonify(best_etas)

@bp.route('/api/webhook', methods=['POST'])
def webhook():
    if secret := current_app.config['SAMSARA_SECRET']:
        # See https://developers.samsara.com/docs/webhooks#webhook-signatures
        try:
            timestamp = request.headers['X-Samsara-Timestamp']
            signature = request.headers['X-Samsara-Signature']

            prefix = 'v1:{0}:'.format(timestamp)
            message = bytes(prefix, 'utf-8') + request.data
            h = hmac.new(secret, message, sha256)
            expected_signature = 'v1=' + h.hexdigest()

            if expected_signature != signature:
                return jsonify({'status': 'error', 'message': 'Failed to authenticate request.'}), 401
        except KeyError as e:
            return jsonify({'status': 'error', 'message': str(e)}), 400

    """
    Handles incoming webhook events for geofence entries/exits.
    Expects JSON payload with event details.
    """
    data = request.get_json(force=True)

    if not data:
        logger.error(f'Invalid JSON received: {request.data}')
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400

    try:
        # parse top-level event details
        event_id = data.get('eventId')
        event_time = datetime.fromisoformat(data.get('eventTime').replace("Z", "+00:00"))
        event_data = data.get('data', {})

        # parse condition details
        conditions = event_data.get('conditions', [])
        if not conditions:
            logger.error(f'No conditions found in webhook data: {data}')
            return jsonify({'status': 'error', 'message': 'Missing conditions'}), 400

        for condition in conditions:
            details = condition.get('details', {})
            # determine if entry or exit
            if 'geofenceEntry' in details:
                geofence_event = details.get('geofenceEntry', {})
            else:
                geofence_event = details.get('geofenceExit', {})

            vehicle_data = geofence_event.get('vehicle')
            if not vehicle_data:
                continue  # skip conditions with no vehicle

            address = geofence_event.get('address', {})
            geofence = address.get('geofence', {})
            polygon = geofence.get('polygon', {})
            vertices = polygon.get('vertices', [])
            latitude = vertices[0].get('latitude') if vertices else None
            longitude = vertices[0].get('longitude') if vertices else None

            # extract vehicle info
            vehicle_id = vehicle_data.get('id')
            vehicle_name = vehicle_data.get('name')

            if not (vehicle_id and vehicle_name):
                continue  # skip invalid entries

            # find or create vehicle
            vehicle = Vehicle.query.get(vehicle_id)
            if not vehicle:
                vehicle = Vehicle(
                    id=vehicle_id,
                    name=vehicle_name,
                    asset_type=vehicle_data.get('assetType', 'vehicle'),
                    license_plate=vehicle_data.get('licensePlate'),
                    vin=vehicle_data.get('vin'),
                    maintenance_id=vehicle_data.get('externalIds', {}).get('maintenanceId'),
                    gateway_model=vehicle_data.get('gateway', {}).get('model'),
                    gateway_serial=vehicle_data.get('gateway', {}).get('serial'),
                )
                db.session.add(vehicle)

            db.session.execute(
                postgresql.insert(GeofenceEvent).on_conflict_do_nothing().values(
                    id=event_id,
                    vehicle_id=vehicle_id,
                    event_type='geofenceEntry' if 'geofenceEntry' in details else 'geofenceExit',
                    event_time=event_time,
                    address_name=address.get("name"),
                    address_formatted=address.get("formattedAddress"),
                    latitude=latitude,
                    longitude=longitude,
                )
            )

        db.session.commit()
        
        # Invalidate Cache
        cache.delete('vehicles_in_geofence') 
        
        return jsonify({'status': 'success'}), 200

    except Exception as e:
        db.session.rollback()

        logger.exception(f'Error processing webhook data: {e}')
        return jsonify({'status': 'error', 'message': str(e)}), 500

@bp.route('/api/today', methods=['GET'])
def data_today():
    now = datetime.now(timezone.utc)
    start_of_day = get_campus_start_of_day()
    locations_today = VehicleLocation.query.filter(
        and_(
            VehicleLocation.timestamp >= start_of_day,
            VehicleLocation.timestamp <= now
        )
    ).order_by(VehicleLocation.timestamp.asc()).all()

    events_today = db.session.query(GeofenceEvent).filter(
        and_(
            GeofenceEvent.event_time >= start_of_day,
            GeofenceEvent.event_time <= now
        )
    ).order_by(GeofenceEvent.event_time.asc()).all()

    locations_today_dict = {}
    for location in locations_today:
        vehicle_location = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "timestamp": location.timestamp,
            "speed_mph": location.speed_mph,
            "heading_degrees": location.heading_degrees,
            "address_id": location.address_id
        }
        if location.vehicle_id in locations_today_dict:
            locations_today_dict[location.vehicle_id]["data"].append(vehicle_location)
        else:
            locations_today_dict[location.vehicle_id] = {
                "entry": None,
                "exit": None,
                "data": [vehicle_location]
            }
    for e, geofence_event in enumerate(events_today):
        if geofence_event.event_type == "geofenceEntry":
            if "entry" not in locations_today_dict[geofence_event.vehicle_id]: # first entry
                locations_today_dict[geofence_event.vehicle_id]["entry"] = geofence_event.event_time
        elif geofence_event.event_type == "geofenceExit":
            if "entry" in locations_today_dict[geofence_event.vehicle_id]: # makes sure that the vehicle already entered
                locations_today_dict[geofence_event.vehicle_id]["exit"] = geofence_event.event_time

    return jsonify(locations_today_dict)

@bp.route('/api/routes', methods=['GET'])
def get_shuttle_routes():
    root_dir = Path(__file__).parent.parent
    return send_from_directory(root_dir / 'data', 'routes.json')

@bp.route('/api/schedule', methods=['GET'])
def get_shuttle_schedule():
    root_dir = Path(__file__).parent.parent
    return send_from_directory(root_dir / 'data', 'schedule.json')

@bp.route('/api/aggregated-schedule', methods=['GET'])
def get_aggregated_shuttle_schedule():
    root_dir = Path(__file__).parent.parent
    return send_from_directory(root_dir / 'data', 'aggregated_schedule.json')
