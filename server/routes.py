from flask import Blueprint, request, jsonify, send_from_directory, current_app
from numpy import datetime_as_string
from . import db
from .models import Vehicle, GeofenceEvent, VehicleLocation
from pathlib import Path
from sqlalchemy import func, and_
from sqlalchemy.dialects import postgresql
from datetime import datetime, date, timezone
from data.stops import Stops
from data.schedules import Schedule
from hashlib import sha256
import hmac
import logging

from .time_utils import get_campus_start_of_day

logger = logging.getLogger(__name__)

bp = Blueprint('routes', __name__)

@bp.route('/')
@bp.route('/schedule')
@bp.route('/about')
@bp.route('/data')
@bp.route('/map')
@bp.route('/generate-static-routes')
def serve_react():
    # serve the React app's index.html for all main routes
    root_dir = Path(__file__).parent.parent / 'client' / 'dist'
    return send_from_directory(root_dir, 'index.html')

@bp.route('/api/locations', methods=['GET'])
@cache.cached(timeout=300, key_prefix="vehicle_locations")
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

    # Format response
    response = {}
    for loc, vehicle in results:
        # Get closest loop
        closest_distance, _, closest_route_name, polyline_index = Stops.get_closest_point(
            (loc.latitude, loc.longitude)
        )
        if closest_distance is None:
            route_name = "UNCLEAR"
        else:
            route_name = closest_route_name if closest_distance < 0.020 else None
        response[loc.vehicle_id] = {
            'name': loc.name,
            'latitude': loc.latitude,
            'longitude': loc.longitude,
            'timestamp': loc.timestamp.isoformat(),
            'heading_degrees': loc.heading_degrees,
            'speed_mph': loc.speed_mph,
            'route_name': route_name,
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

# see DATA_TODAY.md for more information
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

    locations_today_dict = {}  # returned by the function in JSON format
    threshold_noise = 0.004    # locational noise, threshold of 0.00008 determined from \shubble\test-server\server.py: mock_feed()
    threshold_atStop = 0.05    # at a stop, originally 0.02
    shuttle_state = {}         # helper, {"vehicle_id" -> "state"}
    shuttle_prev = {}          # helper, {"vehicle_id" -> [datetime.time_of_day, float.prev_latitude, float.prev_longitude]}
    for location in locations_today:
        # RELATED DATA:  
        # tuple with the distance to closest point, closest point (latitude, longitude), route name, and polyline index
        _closest_point = Stops.get_closest_point((location.latitude, location.longitude)) 
        # tuple with (route name, stop name) if close enough, else None.
        _at_stop = Stops.is_at_stop((location.latitude, location.longitude), threshold_atStop)[1]
        if _at_stop is None:
            _at_stop = "NONE"
        # datetime to string
        _timestamp = location.timestamp.strftime("%H:%M:%S") 

        # LOCATIONS:
        # setup dict nesting
        vehicle_location = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "closest_route_location": _closest_point[1],
            "distance": _closest_point[0],
            "closest_route": _closest_point[2],
            "closest_polyline": _closest_point[3],
            "at_stop": ""
        }
        # initialization: adding a new vehicle to the dict
        if location.vehicle_id not in locations_today_dict:
            locations_today_dict[location.vehicle_id] = {
                "locations": {_timestamp: vehicle_location},
                "loops": [],
                "breaks": [{"locations": [_timestamp]}]
            }
            # update helpers, start the first break as "entry"
            shuttle_state[location.vehicle_id] = "entry"
            shuttle_prev[location.vehicle_id] = [location.timestamp, location.latitude, location.longitude]
            continue
        else:
            locations_today_dict[location.vehicle_id]["locations"][_timestamp] = vehicle_location 

        # LOOPS/BREAKS:
        is_at_union = _at_stop[1] == "STUDENT_UNION" and (_closest_point[2] != "WEST" and _closest_point[2] != "NORTH")
        is_off_route = _closest_point[0] is not None and _closest_point[0] > 0.2
        is_stopped = False
        # determining bool.is_stopped 
        if location.vehicle_id not in shuttle_prev:
            shuttle_prev[location.vehicle_id] = [location.timestamp, location.latitude, location.longitude]
        else:
            # to update: technically shouldn't subtract lat and lon
            if abs(location.latitude - shuttle_prev[location.vehicle_id][1]) < threshold_noise and abs(location.longitude - shuttle_prev[location.vehicle_id][2]) < threshold_noise:
                if (location.timestamp - shuttle_prev[location.vehicle_id][0]).total_seconds() > 300:
                    is_stopped = True
            else:
                shuttle_prev[location.vehicle_id] = [location.timestamp, location.latitude, location.longitude]

        # initialization: "entry" state
        if shuttle_state[location.vehicle_id] == "entry" and is_at_union:
            shuttle_state[location.vehicle_id] = "break"
        # check: end break & start loop
        elif shuttle_state[location.vehicle_id] == "break" and not (is_at_union or is_off_route or is_stopped):
            # end break
            shuttle_state[location.vehicle_id] = "loop"
            # start new loop
            locations_today_dict[location.vehicle_id]["loops"].append({
                "locations": []
            })
        # check: end loop & start break
        elif shuttle_state[location.vehicle_id] == "loop" and is_at_union:
            # end loop
            shuttle_state[location.vehicle_id] = "break"
            # start new break
            locations_today_dict[location.vehicle_id]["breaks"].append({
                "locations": []
            })
        
        # update break/loop locations
        if shuttle_state[location.vehicle_id] == "break" or shuttle_state[location.vehicle_id] == "entry":
            locations_today_dict[location.vehicle_id]["breaks"][-1]["locations"].append(_timestamp)
        elif shuttle_state[location.vehicle_id] == "loop":
            locations_today_dict[location.vehicle_id]["loops"][-1]["locations"].append(_timestamp)

            
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

@bp.route('/api/matched-schedules', methods=['GET'])
def get_matched_shuttle_schedules():
    """
    Return cached matched schedules unless force_recompute=true,
    in which case recompute and update the cache.
    """

    try:
        # Parse URL argument, default False
        force_recompute = request.args.get("force_recompute", "false").lower() == "true"

        # If not forcing recompute, try cache first
        if not force_recompute:
            cached = cache.get("schedule_entries")
            if cached is not None:
                return jsonify({
                    "status": "success",
                    "matchedSchedules": cached,
                    "source": "cache"
                }), 200

        # Otherwise compute fresh and overwrite cache
        matched = Schedule.match_shuttles_to_schedules()
        cache.set("schedule_entries", matched, timeout=3600)

        return jsonify({
            "status": "success",
            "matchedSchedules": matched,
            "source": "recomputed" if force_recompute else "fallback_computed"
        }), 200

    except Exception as e:
        logger.exception(f"Error in matched schedule endpoint: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

