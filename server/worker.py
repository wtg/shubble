from . import create_app, db, cache, Config
from .time_utils import get_campus_start_of_day
from .models import VehicleLocation, GeofenceEvent, Driver, DriverVehicleAssignment
from sqlalchemy import func, and_
import time
import requests
import os
import logging
from datetime import datetime, date, timedelta
from data.schedules import Schedule
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
    start_of_today = get_campus_start_of_day()

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

def update_locations(app):
    """
    Fetches and updates vehicle locations for vehicles currently in the geofence.
    Uses pagination token to fetch subsequent pages.
    """
    # Get the current list of vehicles in the geofence
    current_vehicle_ids = get_vehicles_in_geofence()

    # No vehicles to update
    if not current_vehicle_ids:
        logger.info('No vehicles in geofence to update')
        return

    headers = {'Accept': 'application/json'}
    # Determine API URL based on environment
    if app.config['ENV'] == 'development':
        url = 'http://localhost:4000/fleet/vehicles/stats'
    else:
        api_key = os.environ.get('API_KEY')
        if not api_key:
            logger.error('API_KEY not set')
            return
        headers['Authorization'] = f'Bearer {api_key}'
        url = 'https://api.samsara.com/fleet/vehicles/stats'

    url_params = {
        'vehicleIds': ','.join(current_vehicle_ids),
        'types': 'gps',
    }

    try:
        has_next_page = True
        after_token = None
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
                return

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
                gps = vehicle.get('gps')

                if not vehicle_id or not gps:
                    continue

                timestamp_str = gps.get('time')
                if not timestamp_str:
                    continue
                # Convert ISO 8601 string to datetime
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

                exists = VehicleLocation.query.filter_by(
                    vehicle_id=vehicle_id,
                    timestamp=timestamp
                ).first()
                if exists:
                    continue  # Skip if record already exists
                
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
                cache.delete("schedule_entries")
                logger.info(f'Updated locations for {len(current_vehicle_ids)} vehicles - {new_records_added} new records')
            else:
                logger.info(f'No new location data for {len(current_vehicle_ids)} vehicles')

    except requests.RequestException as e:
        logger.error(f'Failed to fetch locations: {e}')

    return


def update_driver_assignments(app, vehicle_ids):
    """
    Fetches and updates driver-vehicle assignments for vehicles currently in the geofence.
    Creates/updates driver records and tracks assignment changes.
    """
    if not vehicle_ids:
        logger.info('No vehicles to fetch driver assignments for')
        return

    headers = {'Accept': 'application/json'}
    # Determine API URL based on environment
    if app.config['ENV'] == 'development':
        url = 'http://localhost:4000/fleet/driver-vehicle-assignments'
    else:
        api_key = os.environ.get('API_KEY')
        if not api_key:
            logger.error('API_KEY not set for driver assignments')
            return
        headers['Authorization'] = f'Bearer {api_key}'
        url = 'https://api.samsara.com/fleet/driver-vehicle-assignments'

    url_params = {
        'vehicleIds': ','.join(vehicle_ids),
    }

    try:
        has_next_page = True
        after_token = None
        assignments_updated = 0

        while has_next_page:
            if after_token:
                url_params['after'] = after_token
            has_next_page = False

            response = requests.get(url, headers=headers, params=url_params)
            if response.status_code != 200:
                logger.error(f'Driver assignments API error: {response.status_code} {response.text}')
                return

            data = response.json()
            pagination = data.get('pagination', {})
            if pagination.get('hasNextPage'):
                has_next_page = True
            after_token = pagination.get('endCursor', after_token)

            now = datetime.utcnow()

            for assignment in data.get('data', []):
                driver_data = assignment.get('driver')
                vehicle_data = assignment.get('vehicle')

                if not driver_data or not vehicle_data:
                    continue

                driver_id = driver_data.get('id')
                driver_name = driver_data.get('name')
                vehicle_id = vehicle_data.get('id')
                assigned_at_str = assignment.get('assignedAtTime')

                if not driver_id or not vehicle_id:
                    continue

                # Parse assignment time
                if assigned_at_str:
                    assigned_at = datetime.fromisoformat(assigned_at_str.replace("Z", "+00:00"))
                else:
                    assigned_at = now

                # Create or update driver
                driver = Driver.query.get(driver_id)
                if not driver:
                    driver = Driver(id=driver_id, name=driver_name)
                    db.session.add(driver)
                    logger.info(f'Created new driver: {driver_name} ({driver_id})')
                elif driver.name != driver_name:
                    driver.name = driver_name

                # Check if there's an existing open assignment for this vehicle
                existing = DriverVehicleAssignment.query.filter_by(
                    vehicle_id=vehicle_id,
                    assignment_end=None
                ).first()

                if existing:
                    # If same driver, no change needed
                    if existing.driver_id == driver_id:
                        continue
                    # Different driver - close the old assignment
                    existing.assignment_end = now
                    logger.info(f'Closed assignment for driver {existing.driver_id} on vehicle {vehicle_id}')

                # Create new assignment
                new_assignment = DriverVehicleAssignment(
                    driver_id=driver_id,
                    vehicle_id=vehicle_id,
                    assignment_start=assigned_at,
                )
                db.session.add(new_assignment)
                assignments_updated += 1

            if assignments_updated > 0:
                db.session.commit()
                logger.info(f'Updated {assignments_updated} driver assignments')
            else:
                logger.info('No driver assignment changes detected')

    except requests.RequestException as e:
        logger.error(f'Failed to fetch driver assignments: {e}')


def run_worker():
    logger.info('Worker started...')

    while True:
        try:
            with app.app_context():
                # Get current vehicles in geofence before updating
                current_vehicle_ids = get_vehicles_in_geofence()

                update_locations(app)

                # Update driver assignments for vehicles in geofence
                if current_vehicle_ids:
                    update_driver_assignments(app, current_vehicle_ids)

                # Recompute matched schedules if data has changed
                if cache.get("schedule_entries") is None:
                    matched = Schedule.match_shuttles_to_schedules()
                    cache.set("schedule_entries", matched, timeout=3600)
                   
        except Exception as e:
            logger.exception(f'Error in worker loop: {e}')

        time.sleep(5)

if __name__ == '__main__':
    run_worker()
