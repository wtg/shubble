import os
import logging
import requests
import threading
from time import sleep
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import Config

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app (required for SQLAlchemy context)
app = Flask(__name__)
app.config.from_object(Config)

db = SQLAlchemy(app)

# Shared data
vehicles_in_geofence = set()
latest_locations = {}
after_token = None
vehicles_lock = threading.Lock()


def update_locations():
    global after_token
    global latest_locations

    with vehicles_lock:
        if not vehicles_in_geofence:
            logger.info('No vehicles_in_geofence to update')
            return

        api_key = os.environ.get('API_KEY')
        if not api_key:
            logger.error('API_KEY not set')
            return

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

        url = 'https://api.samsara.com/fleet/vehicles/stats/feed'

        try:
            has_next_page = True
            while has_next_page:
                has_next_page = False
                response = requests.get(url, headers=headers, params=url_params)
                if response.status_code == 200:
                    data = response.json()

                    pagination = data.get('pagination', {})
                    if pagination.get('hasNextPage', False):
                        has_next_page = True
                    after_token = pagination.get('endCursor', after_token)

                    api_data = data.get('data', [])
                    for vehicle in api_data:
                        vehicle_id = vehicle.get('id')
                        vehicle_name = vehicle.get('name')
                        gps_data_list = vehicle.get('gps', [])

                        if not vehicle_id or not vehicle_name or not gps_data_list:
                            continue

                        gps_data = gps_data_list[0]
                        latest_locations[vehicle_name] = {
                            'lat': gps_data.get('latitude'),
                            'lng': gps_data.get('longitude'),
                            'timestamp': gps_data.get('time'),
                            'speed': gps_data.get('speedMilesPerHour'),
                            'heading': gps_data.get('headingDegrees'),
                            'address': gps_data.get('reverseGeo', {}).get('formattedLocation', ''),
                        }

                    logger.info(f'Updated locations: {latest_locations}')
                else:
                    logger.error(f'API error: {response.status_code} {response.text}')
                    if 'message' in response.text and 'Parameters differ' in response.text:
                        after_token = None

        except requests.RequestException as e:
            logger.error(f'Failed to fetch locations: {e}')


def run_worker():
    logger.info('Worker started...')
    while True:
        try:
            update_locations()
        except Exception as e:
            logger.exception(f'Error in worker loop: {e}')
        sleep(5)  # Polling interval


if __name__ == '__main__':
    with app.app_context():
        run_worker()
