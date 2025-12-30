"""Async background worker for fetching vehicle data from Samsara API."""
import asyncio
import logging
import os
from datetime import datetime, timezone

import httpx
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis

from backend.flask.config import settings
from backend.flask.database import create_async_db_engine, create_session_factory
from backend.flask.models import VehicleLocation, Driver, DriverVehicleAssignment
from backend.flask.utils import get_vehicles_in_geofence

# Logging config
numeric_level = logging._nameToLevel.get(settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def update_locations(session_factory):
    """
    Fetches and updates vehicle locations for vehicles currently in the geofence.
    Uses pagination token to fetch subsequent pages.
    """
    # Get the current list of vehicles in the geofence (cached)
    current_vehicle_ids = await get_vehicles_in_geofence(session_factory)

    # No vehicles to update
    if not current_vehicle_ids:
        logger.info("No vehicles in geofence to update")
        return

    headers = {"Accept": "application/json"}
    # Determine API URL based on environment
    if settings.ENV == "development":
        url = "http://localhost:4000/fleet/vehicles/stats"
    else:
        api_key = settings.API_KEY
        if not api_key:
            logger.error("API_KEY not set")
            return
        headers["Authorization"] = f"Bearer {api_key}"
        url = "https://api.samsara.com/fleet/vehicles/stats"

    url_params = {
        "vehicleIds": ",".join(current_vehicle_ids),
        "types": "gps",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            has_next_page = True
            after_token = None
            new_records_added = 0

            while has_next_page:
                # Add pagination token if present
                if after_token:
                    url_params["after"] = after_token
                has_next_page = False

                # Make the API request
                response = await client.get(url, headers=headers, params=url_params)

                # Handle non-200 responses
                if response.status_code != 200:
                    logger.error(f"API error: {response.status_code} {response.text}")
                    return

                data = response.json()

                # Handle pagination
                pagination = data.get("pagination", {})
                if pagination.get("hasNextPage"):
                    has_next_page = True
                after_token = pagination.get("endCursor", after_token)

                async with session_factory() as session:
                    for vehicle in data.get("data", []):
                        # Process each vehicle's GPS data
                        vehicle_id = vehicle.get("id")
                        vehicle_name = vehicle.get("name")
                        gps = vehicle.get("gps")

                        if not vehicle_id or not gps:
                            continue

                        timestamp_str = gps.get("time")
                        if not timestamp_str:
                            continue

                        # Convert ISO 8601 string to datetime
                        timestamp = datetime.fromisoformat(
                            timestamp_str.replace("Z", "+00:00")
                        )

                        # Use PostgreSQL upsert with ON CONFLICT DO NOTHING and RETURNING
                        insert_stmt = postgresql.insert(VehicleLocation).values(
                            vehicle_id=vehicle_id,
                            timestamp=timestamp,
                            name=vehicle_name,
                            latitude=gps.get("latitude"),
                            longitude=gps.get("longitude"),
                            heading_degrees=gps.get("headingDegrees"),
                            speed_mph=gps.get("speedMilesPerHour"),
                            is_ecu_speed=gps.get("isEcuSpeed", False),
                            formatted_location=gps.get("reverseGeo", {}).get(
                                "formattedLocation"
                            ),
                            address_id=gps.get("address", {}).get("id"),
                            address_name=gps.get("address", {}).get("name"),
                        )
                        # ON CONFLICT on the composite index (vehicle_id, timestamp) DO NOTHING
                        insert_stmt = insert_stmt.on_conflict_do_nothing(
                            index_elements=["vehicle_id", "timestamp"]
                        )
                        # RETURNING id to check if insert occurred
                        insert_stmt = insert_stmt.returning(VehicleLocation.id)

                        result = await session.execute(insert_stmt)
                        inserted_id = result.scalar_one_or_none()

                        # If a row was returned, an insert occurred
                        if inserted_id:
                            new_records_added += 1

                    # Only commit if we actually added new records
                    if new_records_added > 0:
                        await session.commit()
                        logger.info(
                            f"Updated locations for {len(current_vehicle_ids)} vehicles - {new_records_added} new records"
                        )
                        # Invalidate cache for locations
                        await FastAPICache.clear(namespace="vehicles_in_geofence")
                    else:
                        logger.info(
                            f"No new location data for {len(current_vehicle_ids)} vehicles"
                        )

    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch locations: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error in update_locations: {e}")


async def update_driver_assignments(session_factory, vehicle_ids):
    """
    Fetches and updates driver-vehicle assignments for vehicles currently in the geofence.
    Creates/updates driver records and tracks assignment changes.
    """
    if not vehicle_ids:
        logger.info("No vehicles to fetch driver assignments for")
        return

    headers = {"Accept": "application/json"}
    # Determine API URL based on environment
    if settings.ENV == "development":
        url = "http://localhost:4000/fleet/driver-vehicle-assignments"
    else:
        api_key = settings.API_KEY
        if not api_key:
            logger.error("API_KEY not set for driver assignments")
            return
        headers["Authorization"] = f"Bearer {api_key}"
        url = "https://api.samsara.com/fleet/driver-vehicle-assignments"

    url_params = {
        "filterBy": "vehicles",
        "vehicleIds": ",".join(vehicle_ids),
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            has_next_page = True
            after_token = None
            assignments_updated = 0

            while has_next_page:
                if after_token:
                    url_params["after"] = after_token
                has_next_page = False

                response = await client.get(url, headers=headers, params=url_params)
                if response.status_code != 200:
                    logger.error(
                        f"Driver assignments API error: {response.status_code} {response.text}"
                    )
                    return

                data = response.json()
                logger.info(
                    f'Driver assignments API response: {len(data.get("data", []))} assignments returned'
                )

                pagination = data.get("pagination", {})
                if pagination.get("hasNextPage"):
                    has_next_page = True
                after_token = pagination.get("endCursor", after_token)

                now = datetime.now(timezone.utc)

                async with session_factory() as session:
                    for assignment in data.get("data", []):
                        driver_data = assignment.get("driver")
                        vehicle_data = assignment.get("vehicle")

                        if not driver_data or not vehicle_data:
                            continue

                        driver_id = driver_data.get("id")
                        driver_name = driver_data.get("name")
                        vehicle_id = vehicle_data.get("id")
                        assigned_at_str = assignment.get("assignedAtTime")

                        if not driver_id or not vehicle_id:
                            continue

                        # Parse assignment time
                        if assigned_at_str:
                            assigned_at = datetime.fromisoformat(
                                assigned_at_str.replace("Z", "+00:00")
                            )
                        else:
                            assigned_at = now

                        # Create or update driver
                        driver_query = select(Driver).where(Driver.id == driver_id)
                        result = await session.execute(driver_query)
                        driver = result.scalar_one_or_none()

                        if not driver:
                            driver = Driver(id=driver_id, name=driver_name)
                            session.add(driver)
                            logger.info(f"Created new driver: {driver_name} ({driver_id})")
                        elif driver.name != driver_name:
                            driver.name = driver_name

                        # Check if there's an existing open assignment for this vehicle
                        existing_query = select(DriverVehicleAssignment).where(
                            DriverVehicleAssignment.vehicle_id == vehicle_id,
                            DriverVehicleAssignment.assignment_end.is_(None),
                        )
                        result = await session.execute(existing_query)
                        existing = result.scalar_one_or_none()

                        if existing:
                            # If same driver, no change needed
                            if existing.driver_id == driver_id:
                                continue
                            # Different driver - close the old assignment
                            existing.assignment_end = now
                            logger.info(
                                f"Closed assignment for driver {existing.driver_id} on vehicle {vehicle_id}"
                            )

                        # Create new assignment
                        new_assignment = DriverVehicleAssignment(
                            driver_id=driver_id,
                            vehicle_id=vehicle_id,
                            assignment_start=assigned_at,
                        )
                        session.add(new_assignment)
                        assignments_updated += 1

                    if assignments_updated > 0:
                        await session.commit()
                        logger.info(f"Updated {assignments_updated} driver assignments")
                    else:
                        logger.info("No driver assignment changes detected")

    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch driver assignments: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error in update_driver_assignments: {e}")


async def run_worker():
    """Main worker loop that runs continuously."""
    logger.info("Async worker started...")

    # Initialize database engine and session factory
    db_engine = create_async_db_engine(settings.DATABASE_URL, echo=settings.DEBUG)
    session_factory = create_session_factory(db_engine)
    logger.info("Database engine and session factory initialized")

    # Initialize Redis cache for FastAPI cache
    try:
        redis = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=False,
        )
        FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
        logger.info("Redis cache initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis cache: {e}")
        # Continue without cache

    try:
        while True:
            try:
                # Get current vehicles in geofence before updating (cached)
                current_vehicle_ids = await get_vehicles_in_geofence(session_factory)

                # Update locations and driver assignments in parallel
                await asyncio.gather(
                    update_locations(session_factory),
                    update_driver_assignments(session_factory, current_vehicle_ids),
                )

            except Exception as e:
                logger.exception(f"Error in worker loop: {e}")

            await asyncio.sleep(5)
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down worker...")
        await db_engine.dispose()
        logger.info("Database connections closed")


if __name__ == "__main__":
    asyncio.run(run_worker())
