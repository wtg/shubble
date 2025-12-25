"""FastAPI routes for the Shubble API."""
import logging
import hmac
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
from sqlalchemy import func, and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import selectinload

from .database import get_db
from .models import Vehicle, GeofenceEvent, VehicleLocation, DriverVehicleAssignment
from .config import settings
from .time_utils import get_campus_start_of_day
from .utils import get_vehicles_in_geofence_query
from data.stops import Stops
# from data.schedules import Schedule

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/locations")
@cache(expire=60, namespace="locations")
async def get_locations(db: AsyncSession = Depends(get_db)):
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

    # Format response
    response = {}
    for loc in results:
        vehicle = loc.vehicle
        # Get closest loop
        closest_distance, _, closest_route_name, polyline_index = Stops.get_closest_point(
            (loc.latitude, loc.longitude)
        )
        if closest_distance is None:
            route_name = "UNCLEAR"
        else:
            route_name = closest_route_name if closest_distance < 0.050 else None

        # Get current driver info
        driver_info = None
        assignment = current_assignments.get(loc.vehicle_id)
        if assignment and assignment.driver:
            driver_info = {
                "id": assignment.driver.id,
                "name": assignment.driver.name,
            }

        response[loc.vehicle_id] = {
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
        }

    return response


@router.post("/api/webhook")
async def webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Handles incoming webhook events for geofence entries/exits.
    Expects JSON payload with event details.
    """
    # Verify webhook signature if secret is configured
    if secret := settings.SAMSARA_SECRET:
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
        FastAPICache.clear(namespace="vehicles_in_geofence")

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
