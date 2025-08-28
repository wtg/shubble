from flask import Blueprint, request, jsonify, send_from_directory
from . import db
from .models import Vehicle, GeofenceEvent, VehicleLocation
from pathlib import Path
from sqlalchemy import func, and_
from datetime import datetime, date
from data.stops import Stops
import logging

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
    start_of_today = datetime.combine(date.today(), datetime.min.time())

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
        closest_distance, _, closest_route_name, _ = Stops.get_closest_point(
            (loc.latitude, loc.longitude)
        )
        logger.info(f"Vehicle {loc.vehicle_id} closest route {closest_route_name} at distance {closest_distance}")
        if closest_distance is None:
            route_name = "UNCLEAR"
        else:
            route_name = closest_route_name if closest_distance < 0.0002 else None
        response[loc.vehicle_id] = {
            'name': loc.name,
            'latitude': loc.latitude,
            'longitude': loc.longitude,
            'timestamp': loc.timestamp.isoformat(),
            'heading_degrees': loc.heading_degrees,
            'speed_mph': loc.speed_mph,
            'route_name': route_name,
            'is_ecu_speed': loc.is_ecu_speed,
            'formatted_location': loc.formatted_location,
            'address_id': loc.address_id,
            'address_name': loc.address_name,
            'vehicle_name': vehicle.name,
            'license_plate': vehicle.license_plate,
            'vin': vehicle.vin,
            'asset_type': vehicle.asset_type,
            'gateway_model': vehicle.gateway_model,
            'gateway_serial': vehicle.gateway_serial,
        }

    return jsonify(response)

@bp.route('/api/webhook', methods=['POST'])
def webhook():
    """
    Handles incoming webhook events for geofence entries/exits.
    Expects JSON payload with event details.
    """
    data = request.get_json(force=True)

    if not data:
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400

    try:
        # parse top-level event details
        event_id = data.get('eventId')
        event_time = datetime.fromisoformat(data.get('eventTime').replace("Z", "+00:00"))
        event_type = data.get('eventType')
        event_data = data.get('data', {})

        # parse condition details
        conditions = event_data.get('conditions', [])
        if not conditions:
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

            # Create GeofenceEvent
            event = GeofenceEvent(
                id=event_id,
                vehicle_id=vehicle_id,
                event_type='geofenceEntry' if 'geofenceEntry' in details else 'geofenceExit',
                event_time=event_time,
                address_name=address.get("name"),
                address_formatted=address.get("formattedAddress"),
                latitude=latitude,
                longitude=longitude,
            )
            db.session.add(event)

        db.session.commit()
        return jsonify({'status': 'success'}), 200

    except Exception as e:
        db.session.rollback()
        logger.exception("Webhook processing failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500
