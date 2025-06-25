from flask import Blueprint, request, jsonify, send_from_directory
from . import db
from .models import Vehicle, GeofenceEvent
from pathlib import Path
import os
import threading

bp = Blueprint('routes', __name__)

vehicles_in_geofence = set()
vehicles_in_geofence_lock = threading.Lock()
latest_locations = {}

@bp.route('/')
@bp.route('/schedule')
@bp.route('/about')
def serve_react():
    root_dir = Path(__file__).parent.parent / 'client' / 'dist'
    return send_from_directory(root_dir, 'index.html')

@bp.route('/api/locations', methods=['GET'])
def get_locations():
    global latest_locations
    return jsonify(latest_locations)

@bp.route('/api/webhook', methods=['POST'])
def webhook():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400

    try:
        event_data = data.get('data', {})
        vehicle_data = event_data.get('vehicle', {})
        vehicle_id = vehicle_data.get('id')
        vehicle_name = vehicle_data.get('name')
        event_type = data.get('eventType')
        event_time = data.get('eventTime')

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
            vehicle_id=vehicle_id,
            event_type=event_type,
            event_time=event_time
        )
        db.session.add(event)
        db.session.commit()

        with vehicles_in_geofence_lock:
            if event_type == 'GeofenceEntry':
                vehicles_in_geofence.add(vehicle_id)
            elif event_type == 'GeofenceExit':
                vehicles_in_geofence.discard(vehicle_id)

        return jsonify({'status': 'success'}), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': 'Internal error'}), 500

@bp.route('/api/mapkit', methods=['GET'])
def get_mapkit():
    api_key = os.environ.get('MAPKIT_API_KEY')
    if not api_key:
        return {'status': 'error', 'message': 'MAPKIT_API_KEY not set'}, 400
    return jsonify(api_key)
