from flask import Blueprint, request, jsonify, send_from_directory, current_app
from . import db
from .models import Vehicle, GeofenceEvent, VehicleLocation
from pathlib import Path
from sqlalchemy import func, and_
from sqlalchemy.dialects import postgresql
from datetime import datetime, date, timezone
from data.stops import Stops
from hashlib import sha256
import hmac
import logging
from .services.eta_predictor import ETAPredictor
from .time_utils import get_campus_start_of_day
import pandas as pd

logger = logging.getLogger(__name__)

bp = Blueprint('routes', __name__)

@bp.route('/')
@bp.route('/schedule')
@bp.route('/about')
@bp.route('/data')
@bp.route('/generate-static-routes')
def serve_react():
    # serve the React app's index.html for all main routes
    root_dir = Path(__file__).parent.parent / 'client' / 'dist'
    return send_from_directory(root_dir, 'index.html')

@bp.route('/api/locations', methods=['GET'])
def get_locations():
    """
    Returns the latest location for each vehicle currently inside the geofence.
    The vehicle is considered inside the geofence if its latest geofence event
    today is a 'geofenceEntry'.
    """
    # Start of today for filtering today's geofence events
    start_of_today = get_campus_start_of_day()

    # Subquery: latest geofence event today per vehicle
    latest_geofence_events = db.session.query(
        GeofenceEvent.vehicle_id,
        func.max(GeofenceEvent.event_time).label('latest_time')
    ).filter(
        GeofenceEvent.event_time >= start_of_today
    ).group_by(GeofenceEvent.vehicle_id).subquery()

    # Join to get full geofence event rows where event is geofenceEntry
    geofence_entries = db.session.query(GeofenceEvent.vehicle_id).join(
        latest_geofence_events,
        and_(
            GeofenceEvent.vehicle_id == latest_geofence_events.c.vehicle_id,
            GeofenceEvent.event_time == latest_geofence_events.c.latest_time
        )
    ).filter(GeofenceEvent.event_type == 'geofenceEntry').subquery()

    # Subquery: latest vehicle location per vehicle
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

    predictor = ETAPredictor() # Get the singleton instance
    response = {}
    
    for loc, vehicle in results:
        # ... (Existing closest loop logic) ...
        closest_distance, _, closest_route_name, polyline_index = Stops.get_closest_point(
            (loc.latitude, loc.longitude)
        )
        
        # --- NEW: Calculate ETA ---
        predicted_eta_seconds = None
        
        # Only predict if we are on a known route
        if closest_route_name: 
            # Fetch last 5 locations for this vehicle for LSTM context
            history = VehicleLocation.query.filter_by(vehicle_id=loc.vehicle_id)\
                .order_by(VehicleLocation.timestamp.desc())\
                .limit(5)\
                .all()
            
            # We need 5 points to make a prediction
            if len(history) == 5:
                # Convert to DataFrame, sort ascending by time
                history_df = pd.DataFrame([{
                    'latitude': h.latitude,
                    'longitude': h.longitude,
                    'speed_mph': h.speed_mph if h.speed_mph else 0, # Handle None speeds
                    'heading_degrees': h.heading_degrees,
                    'timestamp': h.timestamp
                } for h in history]).sort_values('timestamp')

                # Run Inference
                # IMPORTANT: Pass the segment_id or route_name exactly as trained
                # You might need logic here to format "From_StopA_To_StopB" based on polyline_index
                predicted_eta_seconds = predictor.predict(history_df, closest_route_name)

        response[loc.vehicle_id] = {
            'name': loc.name,
            'latitude': loc.latitude,
            'longitude': loc.longitude,
            'timestamp': loc.timestamp.isoformat(),
            'heading_degrees': loc.heading_degrees,
            'speed_mph': loc.speed_mph,
            'route_name': closest_route_name,
            'eta_seconds': predicted_eta_seconds,
            'polyline_index': polyline_index,
            'is_ecu_speed': loc.is_ecu_speed,
            'formatted_location': loc.formatted_location,
            'address_id': loc.address_id,
            'address_name': loc.address_name,
            'license_plate': vehicle.license_plate,
            'vin': vehicle.vin,
            'asset_type': vehicle.asset_type,
            'gateway_model': vehicle.gateway_model,
            'gateway_serial': vehicle.gateway_serial,
        }

    return jsonify(response)
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
