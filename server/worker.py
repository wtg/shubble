from . import create_app, db, cache, Config
from .models import VehicleLocation, GeofenceEvent
from sqlalchemy import func, and_
import time
import requests
import os
import logging
from datetime import datetime, date, timedelta
import math

# Logging config
numeric_level = logging._nameToLevel.get(Config.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=numeric_level
)
logger = logging.getLogger(__name__)

app = create_app()

@cache.cached(timeout=300, key_prefix="vehicles_in_geofence")
def get_vehicles_in_geofence():
    """
    Returns a set of vehicle_ids where the latest geofence event from today
    is a geofenceEntry.
    """
    start_of_today = datetime.combine(date.today(), datetime.min.time())

    # Filter to today's events first
    today_events = db.session.query(GeofenceEvent).filter(
        GeofenceEvent.event_time >= start_of_today
    ).subquery()

    # Subquery to get latest event per vehicle from today's events
    subquery = db.session.query(
        today_events.c.vehicle_id,
        func.max(today_events.c.event_time).label('latest_time')
    ).group_by(today_events.c.vehicle_id).subquery()

    # Join back to get the latest event row
    latest_entries = db.session.query(today_events.c.vehicle_id).join(
        subquery,
        and_(
            today_events.c.vehicle_id == subquery.c.vehicle_id,
            today_events.c.event_time == subquery.c.latest_time
        )
    ).filter(today_events.c.event_type == 'geofenceEntry').all()

    return {row.vehicle_id for row in latest_entries}

def update_locations(after_token, previous_vehicle_ids, app):
    """
    Fetches and updates vehicle locations for vehicles currently in the geofence.
    Uses pagination token to fetch subsequent pages.
    Returns updated after_token and current vehicle IDs.
    """
    # Get the current list of vehicles in the geofence
    current_vehicle_ids = get_vehicles_in_geofence()

    # If vehicle list changed, reset pagination token
    if current_vehicle_ids != previous_vehicle_ids:
        logger.info('Vehicle list changed; resetting after_token')
        after_token = None

    # No vehicles to update
    if not current_vehicle_ids:
        logger.info('No vehicles in geofence to update')
        return after_token, current_vehicle_ids

    headers = {'Accept': 'application/json'}
    # Determine API URL based on environment
    if app.config['ENV'] == 'development':
        url = 'http://localhost:4000/fleet/vehicles/stats/feed'
    else:
        api_key = os.environ.get('API_KEY')
        if not api_key:
            logger.error('API_KEY not set')
            return after_token, current_vehicle_ids
        headers['Authorization'] = f'Bearer {api_key}'
        url = 'https://api.samsara.com/fleet/vehicles/stats/feed'

    url_params = {
        'vehicleIds': ','.join(current_vehicle_ids),
        'types': 'gps',
    }

    try:
        has_next_page = True
        new_records_added = 0
        while has_next_page:
            # Add pagination token if present
            if after_token:
                url_params['after'] = after_token
            has_next_page = False
            # Make the API request
            response = requests.get(url, headers=headers, params=url_params)
            # Handle non-200 responses
            if response.status_code != 200:
                logger.error(f'API error: {response.status_code} {response.text}')
                if 'message' in response.text and 'Parameters differ' in response.text:
                    return None, current_vehicle_ids
                return after_token, current_vehicle_ids

            data = response.json()
            # Handle pagination
            pagination = data.get('pagination', {})
            if pagination.get('hasNextPage'):
                has_next_page = True
            after_token = pagination.get('endCursor', after_token)

            for vehicle in data.get('data', []):
                # Process each vehicle's GPS data
                vehicle_id = vehicle.get('id')
                vehicle_name = vehicle.get('name')
                gps_data_list = vehicle.get('gps', [])

                if not vehicle_id or not gps_data_list:
                    continue

                # Use the first GPS entry (most recent)
                gps = gps_data_list[0]
                timestamp_str = gps.get('time')
                if not timestamp_str:
                    continue
                # Convert ISO 8601 string to datetime
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                
                # Check if we have a very recent location for this vehicle (within last 2 minutes)
                recent_location = VehicleLocation.query.filter(
                    VehicleLocation.vehicle_id == vehicle_id,
                    VehicleLocation.timestamp >= timestamp - timedelta(seconds=10)
                ).order_by(VehicleLocation.timestamp.desc()).first()
                
                print(gps_data_list)
                print(recent_location.latitude if recent_location else "None")
                if recent_location and haversine((gps['latitude'], gps['longitude']), (recent_location.latitude, recent_location.longitude)) < 0.01:
                    print("Location is the same")
                    continue
                
                
                # Create and add new VehicleLocation
                loc = VehicleLocation(
                    vehicle_id=vehicle_id,
                    timestamp=timestamp,
                    name=vehicle_name,
                    latitude=gps.get('latitude'),
                    longitude=gps.get('longitude'),
                    heading_degrees=gps.get('headingDegrees'),
                    speed_mph=gps.get('speedMilesPerHour'),
                    is_ecu_speed=gps.get('isEcuSpeed', False),
                    formatted_location=gps.get('reverseGeo', {}).get('formattedLocation'),
                    address_id=gps.get('address', {}).get('id'),
                    address_name=gps.get('address', {}).get('name'),
                )
                db.session.add(loc)
                new_records_added += 1

            # Only commit and invalidate cache if we actually added new records
            if new_records_added > 0:
                db.session.commit()
                cache.delete('vehicle_locations')
                logger.info(f'Updated locations for {len(current_vehicle_ids)} vehicles - {new_records_added} new records')
            else:
                logger.info(f'No new location data for {len(current_vehicle_ids)} vehicles')

    except requests.RequestException as e:
        logger.error(f'Failed to fetch locations: {e}')

    return after_token, current_vehicle_ids

def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points on the Earth
    using the Haversine formula.

    Parameters:
        coord1: (lat1, lon1) in decimal degrees
        coord2: (lat2, lon2) in decimal degrees

    Returns:
        Distance in kilometers.
    """
    # Earth radius in kilometers
    R = 6371.0

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def run_worker():
    logger.info('Worker started...')
    after_token = None
    previous_vehicle_ids = set()

    while True:
        try:
            with app.app_context():
                after_token, previous_vehicle_ids = update_locations(after_token, previous_vehicle_ids, app)
        except Exception as e:
            logger.exception(f'Error in worker loop: {e}')
        time.sleep(5)

if __name__ == '__main__':
    run_worker()
