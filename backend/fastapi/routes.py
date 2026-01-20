"""FastAPI routes for the Shubble API."""
import logging
import hmac
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request, Depends, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
from sqlalchemy import func, and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import selectinload

from backend.database import get_db
from backend.models import Vehicle, GeofenceEvent, VehicleLocation, DriverVehicleAssignment, ETA, PredictedLocation
from backend.config import settings
from backend.time_utils import get_campus_start_of_day
from backend.utils import get_vehicles_in_geofence_query, smart_closest_point
from shared.stops import Stops
# from shared.schedules import Schedule

logger = logging.getLogger(__name__)

router = APIRouter()


@cache(expire=60, namespace="predictions")
async def get_latest_etas_and_predicted_locations(vehicle_ids: list[str], db: AsyncSession):
    """
    Get the latest ETA and predicted location for each vehicle.

    Args:
        vehicle_ids: List of vehicle IDs to get predictions for
        db: Async database session

    Returns:
        Tuple of (etas_dict, predicted_locations_dict) where:
        - etas_dict maps vehicle_id to latest ETA data
        - predicted_locations_dict maps vehicle_id to latest predicted location data
    """
    if not vehicle_ids:
        return {}, {}

    # Subquery: latest ETA per vehicle
    latest_etas = (
        select(
            ETA.vehicle_id,
            func.max(ETA.timestamp).label("latest_time"),
        )
        .where(ETA.vehicle_id.in_(vehicle_ids))
        .group_by(ETA.vehicle_id)
        .subquery()
    )

    # Join to get full ETA data
    etas_query = (
        select(ETA)
        .join(
            latest_etas,
            and_(
                ETA.vehicle_id == latest_etas.c.vehicle_id,
                ETA.timestamp == latest_etas.c.latest_time,
            ),
        )
    )

    etas_result = await db.execute(etas_query)
    etas = etas_result.scalars().all()

    # Subquery: latest predicted location per vehicle
    latest_predicted = (
        select(
            PredictedLocation.vehicle_id,
            func.max(PredictedLocation.timestamp).label("latest_time"),
        )
        .where(PredictedLocation.vehicle_id.in_(vehicle_ids))
        .group_by(PredictedLocation.vehicle_id)
        .subquery()
    )

    # Join to get full predicted location data
    predicted_query = (
        select(PredictedLocation)
        .join(
            latest_predicted,
            and_(
                PredictedLocation.vehicle_id == latest_predicted.c.vehicle_id,
                PredictedLocation.timestamp == latest_predicted.c.latest_time,
            ),
        )
    )

    predicted_result = await db.execute(predicted_query)
    predicted_locations = predicted_result.scalars().all()

    # Build dictionaries
    etas_dict = {}
    for eta in etas:
        etas_dict[eta.vehicle_id] = {
            "stop_times": eta.etas,
            "timestamp": eta.timestamp.isoformat(),
        }

    predicted_dict = {}
    for pred in predicted_locations:
        predicted_dict[pred.vehicle_id] = {
            "speed_kmh": pred.speed_kmh,
            "timestamp": pred.timestamp.isoformat(),
        }

    return etas_dict, predicted_dict


@router.get("/api/locations")
@cache(expire=60, namespace="locations")
async def get_locations(response: Response, db: AsyncSession = Depends(get_db)):
    """
    Returns the latest location for each vehicle currently inside the geofence.
    The vehicle is considered inside the geofence if its latest geofence event
    today is a 'geofenceEntry'.
    """
    # Get query for vehicles in geofence and convert to subquery
    geofence_entries = get_vehicles_in_geofence_query().subquery()

    # Subquery: latest vehicle location per vehicle
    latest_locations = (
        select(
            VehicleLocation.vehicle_id,
            func.max(VehicleLocation.timestamp).label("latest_time"),
        )
        .where(VehicleLocation.vehicle_id.in_(select(geofence_entries.c.vehicle_id)))
        .group_by(VehicleLocation.vehicle_id)
        .subquery()
    )

    # Join to get full location and vehicle info for vehicles in geofence
    query = (
        select(VehicleLocation)
        .join(
            latest_locations,
            and_(
                VehicleLocation.vehicle_id == latest_locations.c.vehicle_id,
                VehicleLocation.timestamp == latest_locations.c.latest_time,
            ),
        )
        .options(selectinload(VehicleLocation.vehicle))
    )

    result = await db.execute(query)
    results = result.scalars().all()

    # Get current driver assignments for all vehicles in results
    vehicle_ids = [loc.vehicle_id for loc in results]
    current_assignments = {}
    if vehicle_ids:
        assignments_query = (
            select(DriverVehicleAssignment)
            .where(
                DriverVehicleAssignment.vehicle_id.in_(vehicle_ids),
                DriverVehicleAssignment.assignment_end.is_(None),
            )
        ).options(selectinload(DriverVehicleAssignment.driver))
        assignments_result = await db.execute(assignments_query)
        assignments = assignments_result.scalars().all()
        for assignment in assignments:
            current_assignments[assignment.vehicle_id] = assignment

    # Get latest ETAs and predicted locations
    etas_dict, predicted_dict = await get_latest_etas_and_predicted_locations(vehicle_ids, db)

    # Get route matching data from cached dataframe
    closest_points = await smart_closest_point(vehicle_ids)

    # Format response
    response_data = {}
    oldest_timestamp = None
    for loc in results:
        vehicle = loc.vehicle
        # Track oldest data point for latency calculation
        if oldest_timestamp is None or loc.timestamp < oldest_timestamp:
            oldest_timestamp = loc.timestamp

        # Get closest point result from smart_closest_point
        closest_distance, _, closest_route_name, polyline_index, _, stop_name = closest_points.get(
            loc.vehicle_id,
            (None, None, None, None, None, None)
        )

        route_name = closest_route_name if closest_distance is not None and closest_distance < 0.050 else None

        # Determine if vehicle is at a stop
        is_at_stop = stop_name is not None
        current_stop = stop_name if is_at_stop else None

        # Get current driver info
        driver_info = None
        assignment = current_assignments.get(loc.vehicle_id)
        if assignment and assignment.driver:
            driver_info = {
                "id": assignment.driver.id,
                "name": assignment.driver.name,
            }
        else:
            driver_info = None

        # Get stop times and predicted location for this vehicle
        # Withhold stop times if vehicle is at Student Union
        routes = Stops.routes_data
        if route_name:
            if stop_name == routes[route_name]['STOPS'][0]:
                stop_times_data = None
            else:
                stop_times_data = etas_dict.get(loc.vehicle_id)
        else:
            stop_times_data = None
        predicted_data = predicted_dict.get(loc.vehicle_id)

        response_data[loc.vehicle_id] = {
            "name": loc.name,
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "timestamp": loc.timestamp.isoformat(),
            "heading_degrees": loc.heading_degrees,
            "speed_mph": loc.speed_mph,
            "route_name": route_name,
            "polyline_index": polyline_index,
            "is_ecu_speed": loc.is_ecu_speed,
            "formatted_location": loc.formatted_location,
            "address_id": loc.address_id,
            "address_name": loc.address_name,
            "license_plate": vehicle.license_plate,
            "vin": vehicle.vin,
            "asset_type": vehicle.asset_type,
            "gateway_model": vehicle.gateway_model,
            "gateway_serial": vehicle.gateway_serial,
            "driver": driver_info,
            "stop_times": stop_times_data,
            "predicted_location": predicted_data,
            "is_at_stop": is_at_stop,
            "current_stop": current_stop,
        }

    # Add timing metadata as HTTP headers to help frontend synchronize with Samsara API
    now = datetime.now(timezone.utc)
    data_age = (now - oldest_timestamp).total_seconds() if oldest_timestamp else None

    response.headers['X-Server-Time'] = now.isoformat()
    response.headers['X-Oldest-Data-Time'] = oldest_timestamp.isoformat() if oldest_timestamp else ''
    response.headers['X-Data-Age-Seconds'] = str(data_age) if data_age is not None else ''

    return response_data


@router.post("/api/webhook")
async def webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Handles incoming webhook events for geofence entries/exits.
    Expects JSON payload with event details.
    """
    # Verify webhook signature if secret is configured
    if secret := settings.samsara_secret_decoded:
        try:
            timestamp = request.headers["X-Samsara-Timestamp"]
            signature = request.headers["X-Samsara-Signature"]

            # Read request body
            body = await request.body()

            prefix = f"v1:{timestamp}:"
            message = bytes(prefix, "utf-8") + body
            h = hmac.new(secret, message, sha256)
            expected_signature = "v1=" + h.hexdigest()

            if expected_signature != signature:
                return JSONResponse(
                    {"status": "error", "message": "Failed to authenticate request."},
                    status_code=401,
                )
        except KeyError as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=400)

    # Parse JSON payload
    try:
        data = await request.json()
    except Exception:
        logger.error(f"Invalid JSON received")
        return JSONResponse(
            {"status": "error", "message": "Invalid JSON"}, status_code=400
        )

    if not data:
        return JSONResponse(
            {"status": "error", "message": "Empty payload"}, status_code=400
        )

    try:
        # Parse top-level event details
        event_id = data.get("eventId")
        event_time = datetime.fromisoformat(
            data.get("eventTime").replace("Z", "+00:00")
        )
        event_data = data.get("data", {})

        # Parse condition details
        conditions = event_data.get("conditions", [])
        if not conditions:
            logger.error(f"No conditions found in webhook data: {data}")
            return JSONResponse(
                {"status": "error", "message": "Missing conditions"}, status_code=400
            )

        for condition in conditions:
            details = condition.get("details", {})
            # Determine if entry or exit
            if "geofenceEntry" in details:
                geofence_event = details.get("geofenceEntry", {})
            else:
                geofence_event = details.get("geofenceExit", {})

            vehicle_data = geofence_event.get("vehicle")
            if not vehicle_data:
                continue  # Skip conditions with no vehicle

            address = geofence_event.get("address", {})
            geofence = address.get("geofence", {})
            polygon = geofence.get("polygon", {})
            vertices = polygon.get("vertices", [])
            latitude = vertices[0].get("latitude") if vertices else None
            longitude = vertices[0].get("longitude") if vertices else None

            # Extract vehicle info
            vehicle_id = vehicle_data.get("id")
            vehicle_name = vehicle_data.get("name")

            if not (vehicle_id and vehicle_name):
                continue  # Skip invalid entries

            # Find or create vehicle
            vehicle_query = select(Vehicle).where(Vehicle.id == vehicle_id)
            result = await db.execute(vehicle_query)
            vehicle = result.scalar_one_or_none()

            if not vehicle:
                vehicle = Vehicle(
                    id=vehicle_id,
                    name=vehicle_name,
                    asset_type=vehicle_data.get("assetType", "vehicle"),
                    license_plate=vehicle_data.get("licensePlate"),
                    vin=vehicle_data.get("vin"),
                    maintenance_id=vehicle_data.get("externalIds", {}).get(
                        "maintenanceId"
                    ),
                    gateway_model=vehicle_data.get("gateway", {}).get("model"),
                    gateway_serial=vehicle_data.get("gateway", {}).get("serial"),
                )
                db.add(vehicle)
                await db.flush()  # Ensure vehicle.id is available

            # Insert geofence event (using PostgreSQL upsert)
            insert_stmt = postgresql.insert(GeofenceEvent).values(
                id=event_id,
                vehicle_id=vehicle_id,
                event_type=(
                    "geofenceEntry" if "geofenceEntry" in details else "geofenceExit"
                ),
                event_time=event_time,
                address_name=address.get("name"),
                address_formatted=address.get("formattedAddress"),
                latitude=latitude,
                longitude=longitude,
            )
            insert_stmt = insert_stmt.on_conflict_do_nothing()
            await db.execute(insert_stmt)

        await db.commit()

        # Invalidate cache for vehicles in geofence
        await FastAPICache.clear(namespace="vehicles_in_geofence")

        return {"status": "success"}

    except Exception as e:
        await db.rollback()
        logger.exception(f"Error processing webhook data: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@router.get("/api/today")
async def data_today(db: AsyncSession = Depends(get_db)):
    """Get all location data and geofence events for today."""
    now = datetime.now(timezone.utc)
    start_of_day = get_campus_start_of_day()

    # Query locations today
    locations_query = (
        select(VehicleLocation)
        .where(
            and_(
                VehicleLocation.timestamp >= start_of_day,
                VehicleLocation.timestamp <= now,
            )
        )
        .order_by(VehicleLocation.timestamp.asc())
    )
    locations_result = await db.execute(locations_query)
    locations_today = locations_result.scalars().all()

    # Query events today
    events_query = (
        select(GeofenceEvent)
        .where(
            and_(
                GeofenceEvent.event_time >= start_of_day,
                GeofenceEvent.event_time <= now,
            )
        )
        .order_by(GeofenceEvent.event_time.asc())
    )
    events_result = await db.execute(events_query)
    events_today = events_result.scalars().all()

    # Build response dict
    locations_today_dict = {}
    for location in locations_today:
        vehicle_location = {
            "latitude": location.latitude,
            "longitude": location.longitude,
            "timestamp": location.timestamp,
            "speed_mph": location.speed_mph,
            "heading_degrees": location.heading_degrees,
            "address_id": location.address_id,
        }
        if location.vehicle_id in locations_today_dict:
            locations_today_dict[location.vehicle_id]["data"].append(vehicle_location)
        else:
            locations_today_dict[location.vehicle_id] = {
                "entry": None,
                "exit": None,
                "data": [vehicle_location],
            }

    for geofence_event in events_today:
        if geofence_event.event_type == "geofenceEntry":
            if (
                "entry" not in locations_today_dict[geofence_event.vehicle_id]
            ):  # First entry
                locations_today_dict[geofence_event.vehicle_id]["entry"] = (
                    geofence_event.event_time
                )
        elif geofence_event.event_type == "geofenceExit":
            if (
                "entry" in locations_today_dict[geofence_event.vehicle_id]
            ):  # Makes sure that the vehicle already entered
                locations_today_dict[geofence_event.vehicle_id]["exit"] = (
                    geofence_event.event_time
                )

    return locations_today_dict


@router.get("/api/routes")
async def get_shuttle_routes():
    """Serve routes.json file."""
    root_dir = Path(__file__).parent.parent
    routes_file = root_dir / "data" / "routes.json"
    if routes_file.exists():
        return FileResponse(routes_file)
    raise HTTPException(status_code=404, detail="Routes file not found")


@router.get("/api/schedule")
async def get_shuttle_schedule():
    """Serve schedule.json file."""
    root_dir = Path(__file__).parent.parent
    schedule_file = root_dir / "data" / "schedule.json"
    if schedule_file.exists():
        return FileResponse(schedule_file)
    raise HTTPException(status_code=404, detail="Schedule file not found")


@router.get("/api/aggregated-schedule")
async def get_aggregated_shuttle_schedule():
    """Serve aggregated_schedule.json file."""
    root_dir = Path(__file__).parent.parent
    aggregated_file = root_dir / "data" / "aggregated_schedule.json"
    if aggregated_file.exists():
        return FileResponse(aggregated_file)
    raise HTTPException(status_code=404, detail="Aggregated schedule file not found")


@router.get("/api/matched-schedules")
@cache(expire=3600, namespace="matched_schedules")
async def get_matched_shuttle_schedules(force_recompute: bool = False):
    """
    Return cached matched schedules unless force_recompute=true,
    in which case recompute and update the cache.
    """
    try:
        # Note: With fastapi-cache2, the @cache decorator handles caching automatically
        # The force_recompute parameter would need custom cache invalidation logic
        # For now, we compute fresh data if requested

        matched = {} # Schedule.match_shuttles_to_schedules()

        return {
            "status": "success",
            "matchedSchedules": matched,
            "source": "recomputed" if force_recompute else "computed",
        }

    except Exception as e:
        logger.exception(f"Error in matched schedule endpoint: {e}")
        return JSONResponse(
            {"status": "error", "message": str(e)}, status_code=500
        )
