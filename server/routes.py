from flask import Blueprint, request, jsonify, send_from_directory
from . import db
from .models import Vehicle, GeofenceEvent, VehicleLocation
from pathlib import Path
from sqlalchemy import func
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

bp = Blueprint('routes', __name__)

@bp.route('/')
@bp.route('/schedule')
@bp.route('/about')
@bp.route('/data')
def serve_react():
    root_dir = Path(__file__).parent.parent / 'client' / 'dist'
    return send_from_directory(root_dir, 'index.html')

@bp.route('/api/locations', methods=['GET'])
def get_locations():
    logger.error('log working?')
    # Subquery: get the max timestamp per vehicle_id
    subquery = db.session.query(
        VehicleLocation.vehicle_id,
        func.max(VehicleLocation.timestamp).label('latest_time')
    ).group_by(VehicleLocation.vehicle_id).subquery()

    # Join VehicleLocation on vehicle_id and timestamp to get latest rows
    latest_locations = db.session.query(VehicleLocation).join(
        subquery,
        (VehicleLocation.vehicle_id == subquery.c.vehicle_id) &
        (VehicleLocation.timestamp == subquery.c.latest_time)
    ).all()

    # Prepare JSON response
    result = {}
    for loc in latest_locations:
        result[loc.vehicle_id] = {
            'name': loc.name,
            'latitude': loc.latitude,
            'longitude': loc.longitude,
            'timestamp': loc.timestamp.isoformat(),
            'heading_degrees': loc.heading_degrees,
            'speed_mph': loc.speed_mph,
            'is_ecu_speed': loc.is_ecu_speed,
            'formatted_location': loc.formatted_location,
            'address_id': loc.address_id,
            'address_name': loc.address_name,
        }

    return jsonify(result)

@bp.route('/api/webhook', methods=['POST'])
def webhook():
    data = request.get_json(force=True)
    logger.error(data)
    if not data:
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400

    try:
        # parse event details
        event_id = data.get('eventId')
        event_time = datetime.fromisoformat(data.get('eventTime').replace("Z", "+00:00"))
        event_type = data.get('eventType')
        event_data = data.get('data', {})

        # parse vehicle details
        vehicle_data = event_data.get('vehicle', {})
        vehicle_id = vehicle_data.get('id')
        vehicle_name = vehicle_data.get('name')

        # parse event location details
        address = event_data.get("address", {})
        geofence = address.get("geofence", {})
        circle = geofence.get("circle", {})
        latitude = circle.get("latitude")
        longitude = circle.get("longitude")

        if not (vehicle_id and vehicle_name and event_type):
            return jsonify({'status': 'error', 'message': 'Missing fields'}), 400

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
            db.session.commit()

        # Create GeofenceEvent
        event = GeofenceEvent(
            id=event_id,
            vehicle_id=vehicle_id,
            event_type=event_type,
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
        return jsonify({'status': 'error', 'message': 'Internal error'}), 500
