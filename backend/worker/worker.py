"""Async background worker for fetching vehicle data from Samsara API."""
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import delete, select
from sqlalchemy.dialects import postgresql

from backend.config import settings
from backend.cache import init_cache, close_cache, soft_clear_namespace, get_redis
from backend.database import create_async_db_engine, create_session_factory
from backend.models import VehicleLocation, Driver, DriverVehicleAssignment, PredictedLocation
from backend.utils import get_vehicles_in_geofence
from backend.function_timer import timed
from backend.time_utils import dev_now
from backend.worker.data import generate_and_save_predictions, preload_lstm_models
from backend.cache_dataframe import update_today_dataframe

# Logging config for Worker
worker_log_level = settings.get_log_level("worker")
numeric_level = logging._nameToLevel.get(worker_log_level.upper(), logging.INFO)
logging.basicConfig(
    level=numeric_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.info(f"Worker logging level: {worker_log_level}")


# PERF: single shared httpx client so consecutive worker cycles reuse the TCP
# + TLS connection to Samsara (or the dev mock server) instead of paying the
# handshake cost every 5 seconds. Lazily created on first use and closed on
# worker shutdown.
_HTTP_CLIENT: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8),
        )
    return _HTTP_CLIENT


async def _close_http_client() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None and not _HTTP_CLIENT.is_closed:
        await _HTTP_CLIENT.aclose()
    _HTTP_CLIENT = None


@timed
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
    if settings.DEPLOY_MODE == "development":
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
        client = _get_http_client()
        has_next_page = True
        after_token = None
        new_records_added = 0
        inserted_vehicle_ids = []

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
                return []

            data = response.json()

            # Handle pagination
            pagination = data.get("pagination", {})
            if pagination.get("hasNextPage"):
                has_next_page = True
            after_token = pagination.get("endCursor", after_token)

            async with session_factory() as session:
                values_to_insert = []

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

                    values_to_insert.append({
                        "vehicle_id": vehicle_id,
                        "timestamp": timestamp,
                        "name": vehicle_name,
                        "latitude": gps.get("latitude"),
                        "longitude": gps.get("longitude"),
                        "heading_degrees": gps.get("headingDegrees"),
                        "speed_mph": gps.get("speedMilesPerHour"),
                        "is_ecu_speed": gps.get("isEcuSpeed", False),
                        "formatted_location": gps.get("reverseGeo", {}).get(
                            "formattedLocation"
                        ),
                        "address_id": gps.get("address", {}).get("id"),
                        "address_name": gps.get("address", {}).get("name"),
                    })

                if values_to_insert:
                    # Use PostgreSQL bulk upsert with ON CONFLICT DO NOTHING and RETURNING
                    insert_stmt = postgresql.insert(VehicleLocation).values(values_to_insert)
                    # ON CONFLICT on the composite index (vehicle_id, timestamp) DO NOTHING
                    insert_stmt = insert_stmt.on_conflict_do_nothing(
                        index_elements=["vehicle_id", "timestamp"]
                    )
                    # RETURNING vehicle_id to track which vehicles were actually updated
                    insert_stmt = insert_stmt.returning(VehicleLocation.vehicle_id)

                    result = await session.execute(insert_stmt)
                    batch_inserted_ids = result.scalars().all()

                    new_records_added += len(batch_inserted_ids)
                    inserted_vehicle_ids.extend(batch_inserted_ids)

                # Only commit if we actually added new records
                if new_records_added > 0:
                    await session.commit()
                    logger.info(
                        f"Updated locations for {len(current_vehicle_ids)} vehicles - {new_records_added} new records"
                    )
                    # Invalidate cache for locations
                    await soft_clear_namespace("locations")
                    # PUSH: notify /api/locations/stream SSE subscribers. The
                    # cache has been cleared, so next read pulls fresh data.
                    # Fire-and-forget; publish is a ~0ms no-op with no listeners.
                    redis = get_redis()
                    if redis is not None:
                        try:
                            await redis.publish("shubble:locations_updated", b"1")
                        except Exception as e:
                            logger.warning(f"Failed to publish shubble:locations_updated: {e}")
                else:
                    logger.info(
                        f"No new location data for {len(current_vehicle_ids)} vehicles"
                    )

        return inserted_vehicle_ids

    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch locations: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error in update_locations: {e}")
        return []


@timed
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
    if settings.DEPLOY_MODE == "development":
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
        client = _get_http_client()
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

            now = dev_now(timezone.utc)

            async with session_factory() as session:
                # First pass: collect driver_ids and vehicle_ids from this page
                # of the API response so we can bulk-fetch the existing rows in
                # two queries instead of 2×N (see plan improvement #5).
                page_driver_ids: set = set()
                page_vehicle_ids: set = set()
                for assignment in data.get("data", []):
                    driver_data = assignment.get("driver") or {}
                    vehicle_data = assignment.get("vehicle") or {}
                    d_id = driver_data.get("id")
                    v_id = vehicle_data.get("id")
                    if d_id and v_id:
                        page_driver_ids.add(d_id)
                        page_vehicle_ids.add(v_id)

                drivers_by_id: dict = {}
                if page_driver_ids:
                    driver_rows = await session.execute(
                        select(Driver).where(Driver.id.in_(page_driver_ids))
                    )
                    drivers_by_id = {d.id: d for d in driver_rows.scalars()}

                open_assignments_by_vehicle: dict = {}
                if page_vehicle_ids:
                    open_rows = await session.execute(
                        select(DriverVehicleAssignment).where(
                            DriverVehicleAssignment.vehicle_id.in_(page_vehicle_ids),
                            DriverVehicleAssignment.assignment_end.is_(None),
                        )
                    )
                    open_assignments_by_vehicle = {
                        a.vehicle_id: a for a in open_rows.scalars()
                    }

                # Second pass: iterate assignments using the pre-loaded dicts
                # — no queries inside the loop. Semantics identical to the
                # per-assignment version: same logging strings, same
                # create-driver / close-old-assignment / new-assignment flow.
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

                    # Create or update driver (from pre-loaded dict)
                    driver = drivers_by_id.get(driver_id)
                    if not driver:
                        driver = Driver(id=driver_id, name=driver_name)
                        session.add(driver)
                        drivers_by_id[driver_id] = driver
                        logger.info(f"Created new driver: {driver_name} ({driver_id})")
                    elif driver.name != driver_name:
                        driver.name = driver_name

                    # Check for existing open assignment (from pre-loaded dict)
                    existing = open_assignments_by_vehicle.get(vehicle_id)

                    if existing:
                        # If same driver, no change needed
                        if existing.driver_id == driver_id:
                            continue
                        # Different driver - close the old assignment
                        existing.assignment_end = now
                        logger.info(
                            f"Closed assignment for driver {existing.driver_id} on vehicle {vehicle_id}"
                        )

                    # Create new assignment. Update the local dict so later
                    # iterations that reference this vehicle in the SAME page
                    # see the new open assignment and behave identically to
                    # the original per-assignment code.
                    new_assignment = DriverVehicleAssignment(
                        driver_id=driver_id,
                        vehicle_id=vehicle_id,
                        assignment_start=assigned_at,
                    )
                    session.add(new_assignment)
                    open_assignments_by_vehicle[vehicle_id] = new_assignment
                    assignments_updated += 1

                if assignments_updated > 0:
                    await session.commit()
                    await soft_clear_namespace("driver_assignments")
                    logger.info(f"Updated {assignments_updated} driver assignments")
                else:
                    logger.info("No driver assignment changes detected")

    except httpx.HTTPError as e:
        logger.error(f"Failed to fetch driver assignments: {e}")
    except Exception as e:
        logger.exception(f"Unexpected error in update_driver_assignments: {e}")


async def _cleanup_old_predicted_locations(session_factory, days: int = 2) -> int:
    """Delete PredictedLocation rows older than `days` days.

    Called once at worker startup. Keeps the table bounded — the model
    writes ~50K rows/day (~18M/yr), and 2 days is enough history for any
    debugging/replay work. Production analytics use separate aggregation.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    try:
        async with session_factory() as session:
            result = await session.execute(
                delete(PredictedLocation).where(PredictedLocation.timestamp < cutoff)
            )
            await session.commit()
            return result.rowcount or 0
    except Exception as e:
        logger.error(f"PredictedLocation cleanup failed: {e}")
        return 0


async def run_worker():
    """Main worker loop that runs continuously."""
    logger.info("Async worker started...")

    # Initialize database engine and session factory
    db_engine = create_async_db_engine(settings.DATABASE_URL, echo=settings.DEBUG)
    session_factory = create_session_factory(db_engine)
    logger.info("Database engine and session factory initialized")

    # Initialize Redis cache
    try:
        await init_cache(settings.REDIS_URL)
        logger.info("Redis cache initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Redis cache: {e}")
        # Continue without cache

    # One-time cleanup: drop PredictedLocation rows older than 2 days. Runs
    # once per worker startup — the table otherwise grows unbounded.
    try:
        deleted = await _cleanup_old_predicted_locations(session_factory, days=2)
        if deleted:
            logger.info(f"Deleted {deleted} old PredictedLocation rows (> 2 days)")
    except Exception as e:
        logger.error(f"PredictedLocation cleanup errored on startup: {e}")

    # P4: preload LSTM models so the first prediction cycle doesn't eat a
    # cold-load latency spike per (route, polyline_idx). Blocking is fine —
    # this runs once at startup before the ticker.
    try:
        await asyncio.to_thread(preload_lstm_models)
    except Exception as e:
        logger.error(f"LSTM preload failed (will lazy-load on demand): {e}")

    # Worker interval in seconds
    interval = 5

    async def ticker(interval_seconds):
        """Async generator that yields at fixed intervals."""
        next_tick = asyncio.get_event_loop().time()
        while True:
            next_tick += interval_seconds
            yield
            now = asyncio.get_event_loop().time()
            sleep_time = next_tick - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # Tick took longer than interval - skip ahead
                logger.warning(f"Worker cycle exceeded {interval_seconds}s interval by {-sleep_time:.2f}s")
                next_tick = now

    # Per-cycle hard timeout so a stuck HTTP request, DB query, or
    # compute can't silently wedge the worker indefinitely. Sized
    # generously — a healthy cycle takes 5-10s, this kicks in only
    # on pathological hangs.
    CYCLE_TIMEOUT_SEC = 90

    async def _do_cycle():
        current_vehicle_ids = await get_vehicles_in_geofence(session_factory)
        results = await asyncio.gather(
            update_locations(session_factory),
            update_driver_assignments(session_factory, current_vehicle_ids),
        )
        updated_vehicles = results[0]
        if updated_vehicles:
            logger.info(f"Triggering ML cache update for {len(updated_vehicles)} vehicles")
            await update_today_dataframe()
            await generate_and_save_predictions(updated_vehicles)

    try:
        async for _ in ticker(interval):
            try:
                await asyncio.wait_for(_do_cycle(), timeout=CYCLE_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                logger.error(
                    f"Worker cycle exceeded hard timeout of {CYCLE_TIMEOUT_SEC}s — "
                    f"aborting and moving on. If this recurs check HTTP client "
                    f"pool health or DB responsiveness."
                )
            except Exception as e:
                logger.exception(f"Error in worker loop: {e}")

    finally:
        # Cleanup on shutdown
        logger.info("Shutting down worker...")
        await _close_http_client()
        await close_cache()
        await db_engine.dispose()
        logger.info("Database connections closed")


if __name__ == "__main__":
    asyncio.run(run_worker())
