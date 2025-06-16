from flask import Flask, request, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import SQLAlchemyError
from flask_migrate import Migrate
from pathlib import Path
import os
from apscheduler.schedulers.background import BackgroundScheduler
import requests
import logging
import threading
from datetime import datetime

from models import Vehicle, GeofenceEvent

from config import Config

# Configure logging to only show errors and above
logging.basicConfig(level=logging.ERROR)

app = Flask(
    __name__,
    static_folder='../client/dist/',
    static_url_path='/'
)

# Load config settings
app.config.from_object(Config)

# Setup db and migrations
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Global variables to track vehicles inside geofence, latest GPS locations, and pagination token
vehicles_in_geofence = []
latest_locations = {}
after_token = None

# Lock to safely access shared vehicle/geofence data across threads
vehicles_in_geofence_lock = threading.Lock()

def update_locations():
    """
    Periodic function to fetch the latest GPS locations for vehicles
    currently inside geofence, using the Samsara API.
    """
    global latest_locations, vehicles_in_geofence, after_token

    with vehicles_in_geofence_lock:
        # If no vehicles to update, return early
        if not vehicles_in_geofence:
            app.logger.info('No vehicles_in_geofence to update')
            return

        # Get API key from environment variable
        api_key = os.environ.get('API_KEY')
        if not api_key:
            app.logger.error('API_KEY not set')
            return

        # Prepare headers and URL parameters for API call
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json',
        }
        url_params = {
            'vehicleIds': ','.join(vehicles_in_geofence),
            'types': 'gps',
        }
        if after_token:
            url_params['after'] = after_token

        url = 'https://api.samsara.com/fleet/vehicles_in_geofence/stats/feed'

        try:
            has_next_page = True
            while has_next_page:
                has_next_page = False
                response = requests.get(url, headers=headers, params=url_params)
                if response.status_code == 200:
                    data = response.json()

                    # Handle pagination info
                    pagination = data.get('pagination', None)
                    if not pagination:
                        app.logger.error('Invalid pagination')
                        return
                    if pagination.get('hasNextPage', False):
                        has_next_page = True

                    new_after_token = pagination.get('endCursor', None)
                    if not new_after_token:
                        app.logger.error('Invalid after token')
                        app.logger.error(data)
                        return
                    after_token = new_after_token

                    # Extract vehicle GPS data
                    api_data = data.get('data', None)
                    if api_data is None:
                        app.logger.error('Invalid data')
                        app.logger.error(data)
                        return

                    # Update latest_locations dict for each vehicle
                    for vehicle in api_data:
                        vehicle_id = vehicle.get('id', None)
                        if not vehicle_id:
                            app.logger.error('Invalid vehicle ID')
                            app.logger.error(vehicle)
                            continue

                        vehicle_name = vehicle.get('name', None)
                        if not vehicle_name:
                            app.logger.error('Invalid vehicle name')
                            app.logger.error(vehicle)
                            continue

                        gps_data_list = vehicle.get('gps', None)
                        if gps_data_list == []:
                            # No new GPS data since last update
                            continue
                        if gps_data_list is None:
                            app.logger.error('Invalid GPS data list')
                            app.logger.error(vehicle)
                            continue

                        gps_data = gps_data_list[0]
                        if not gps_data:
                            app.logger.error('Invalid GPS data')
                            app.logger.error(vehicle)
                            continue

                        # Sanity check that returned vehicle is in geofence list
                        if vehicle_id not in vehicles_in_geofence:
                            app.logger.warning(f'Vehicle {vehicle_id} not in geofence list')
                            continue

                        # Update the latest location info for this vehicle by name
                        latest_locations[vehicle_name] = {
                            'lat': gps_data.get('latitude', None),
                            'lng': gps_data.get('longitude', None),
                            'timestamp': gps_data.get('time', None),
                            'speed': gps_data.get('speedMilesPerHour', None),
                            'heading': gps_data.get('headingDegrees', None),
                            'address': gps_data.get('reverseGeo', {}).get('formattedLocation', ''),
                        }

                    app.logger.info(f'Updated locations: {latest_locations}')
                else:
                    # Log errors from API response
                    app.logger.error(f'Error fetching locations: {response.status_code} {response.text}')
                    if 'message' in response.data and response.data['message'] == 'Parameters differ from previous paginated request.':
                        after_token = None
        except requests.RequestException as e:
            app.logger.error(f'Error fetching locations: {e}')

# Create background scheduler to periodically update locations
scheduler = BackgroundScheduler()

# Routes serving the React frontend's index.html for different paths
@app.route('/')
@app.route('/schedule')
@app.route('/about')
def serve_react():
    root_dir = Path(app.static_folder)
    return send_from_directory(root_dir, 'index.html')

# API endpoint to get the latest vehicle locations as JSON
@app.route('/api/locations', methods=['GET'])
def get_locations():
    global latest_locations
    return jsonify(latest_locations)

@app.route('/api/webhook', methods=['POST'])
def webhook():
    # Attempt to parse the incoming JSON payload
    data = request.get_json()
    if not data:
        app.logger.error('Invalid JSON')
        return jsonify({'status': 'error', 'message': 'Invalid JSON'}), 400

    try:
        # Extract the nested "data" field from the request
        event_data = data.get('data', {})

        # Extract the "vehicle" field from the event data
        vehicle_data = event_data.get('vehicle', {})
        vehicle_id = vehicle_data.get('id')
        vehicle_name = vehicle_data.get('name')

        # Extract high-level event metadata
        event_type = data.get('eventType')
        event_time = data.get('eventTime')

        # Validate required fields
        if not (vehicle_id and vehicle_name and event_type):
            app.logger.error('Missing required fields in webhook')
            return jsonify({'status': 'error', 'message': 'Missing vehicle or event fields'}), 400

        app.logger.debug(f"Event {event_type} for vehicle {vehicle_id} at {event_time}")

        # Check if the vehicle exists in the database
        vehicle = Vehicle.query.get(vehicle_id)
        if not vehicle:
            # Create a new vehicle record if not found
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

        # Record the geofence event in the database
        geofence_event = GeofenceEvent(
            vehicle_id=vehicle_id,
            event_type=event_type,
            event_time=event_time,
        )
        db.session.add(geofence_event)

        # Update the in-memory geofence list with thread safety
        with vehicles_in_geofence_lock:
            if event_type == 'GeofenceEntry':
                vehicles_in_geofence.add(vehicle_id)
                app.logger.info(f'Geofence entry: {vehicle_id}')
            elif event_type == 'GeofenceExit':
                if vehicle_id in vehicles_in_geofence:
                    vehicles_in_geofence.remove(vehicle_id)
                    app.logger.info(f'Geofence exit: {vehicle_id}')
                else:
                    app.logger.warning(f'Geofence exit received for vehicle not in list: {vehicle_id}')
            else:
                app.logger.warning(f'Unknown event type: {event_type}')
                return jsonify({'status': 'error', 'message': 'Unknown event type'}), 400

        # Commit both vehicle and geofence event changes
        db.session.commit()
        return jsonify({'status': 'success'}), 200

    except SQLAlchemyError as db_err:
        # Roll back the database transaction on error
        db.session.rollback()
        app.logger.error(f'Database error: {db_err}')
        return jsonify({'status': 'error', 'message': 'Database error'}), 500

    except Exception as e:
        # Catch-all for unexpected issues
        app.logger.exception('Unexpected error in webhook handler')
        return jsonify({'status': 'error', 'message': 'Internal error'}), 500

# API endpoint to get MapKit API key from environment (for frontend)
@app.route('/api/mapkit', methods=['GET'])
def get_mapkit():
    api_key = os.environ.get('MAPKIT_API_KEY')
    if not api_key:
        app.logger.error('MAPKIT_API_KEY not set')
        return {'status': 'error', 'message': 'MAPKIT_API_KEY not set'}, 400
    return jsonify(api_key)


if __name__ == '__main__':
    # Start background scheduler for periodic location updates every 5 seconds
    scheduler.start()
    scheduler.add_job(update_locations, trigger="interval", seconds=5)

    # Run Flask app
    app.run()
