"""FastAPI routes for the Shubble API."""
import logging
import hmac
from hashlib import sha256
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Request, Depends, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import func, and_, select

from backend.cache import cache, soft_clear_namespace
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import selectinload
from backend.database import get_db
from backend.models import BusSchedule, BusScheduleToDaySchedule, DateToDaySchedule, Polyline, Route, RouteToBusSchedule, Stop, Vehicle, GeofenceEvent, VehicleLocation, DaySchedule
from backend.config import settings
from backend.time_utils import get_campus_start_of_day
from backend.utils import (
    get_vehicles_in_geofence,
)
from backend.fastapi.utils import (
    smart_closest_point,
    get_latest_vehicle_locations,
    get_current_driver_assignments,
    get_latest_etas,
    get_latest_velocities,
)
from shared.stops import Stops
# from shared.schedules import Schedule

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/locations")
@cache(soft_ttl=15, hard_ttl=300, lock_timeout=5.0,namespace="locations")
async def get_locations(response: Response, request: Request):
    """
    Returns the latest location for each vehicle currently inside the geofence.
    The vehicle is considered inside the geofence if its latest geofence event
    today is a 'geofenceEntry'.

    This endpoint returns raw vehicle location data without ML predictions
    or route matching. Use /api/predictions for that data.
    """
    # Get latest locations for vehicles in geofence
    # Uses cached function that returns dicts
    results = await get_latest_vehicle_locations(request.app.state.session_factory)

    # Get current driver assignments for all vehicles in results
    vehicle_ids = [loc["vehicle_id"] for loc in results]
    current_assignments = await get_current_driver_assignments(vehicle_ids, request.app.state.session_factory)

    # Format response
    response_data = {}
    oldest_timestamp = None

    for loc in results:
        vehicle = loc["vehicle"]
        # Track oldest data point for latency calculation
        # Timestamp is ISO string in cached dict, convert back to datetime
        ts = datetime.fromisoformat(loc["timestamp"])
        if oldest_timestamp is None or ts < oldest_timestamp:
            oldest_timestamp = ts

        # Get current driver info
        driver_info = None
        assignment = current_assignments.get(loc["vehicle_id"])
        # Assignment is DriverAssignmentDict (dict)
        if assignment and assignment.get("driver"):
            driver_data = assignment["driver"]
            driver_info = {
                "id": driver_data["id"],
                "name": driver_data["name"],
            }
        else:
            driver_info = None

        response_data[loc["vehicle_id"]] = {
            "name": loc["name"],
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "timestamp": loc["timestamp"], # Already ISO string
            "heading_degrees": loc["heading_degrees"],
            "speed_mph": loc["speed_mph"],
            "is_ecu_speed": loc["is_ecu_speed"],
            "formatted_location": loc["formatted_location"],
            "address_id": loc["address_id"],
            "address_name": loc["address_name"],
            "license_plate": vehicle["license_plate"],
            "vin": vehicle["vin"],
            "asset_type": vehicle["asset_type"],
            "gateway_model": vehicle["gateway_model"],
            "gateway_serial": vehicle["gateway_serial"],
            "driver": driver_info,
        }

    # Add timing metadata as HTTP headers to help frontend synchronize with Samsara API
    now = datetime.now(timezone.utc)
    data_age = (now - oldest_timestamp).total_seconds() if oldest_timestamp else None

    response.headers['X-Server-Time'] = now.isoformat()
    response.headers['X-Oldest-Data-Time'] = oldest_timestamp.isoformat() if oldest_timestamp else ''
    response.headers['X-Data-Age-Seconds'] = str(data_age) if data_age is not None else ''

    return response_data


@router.get("/api/etas")
@cache(soft_ttl=15, hard_ttl=300, lock_timeout=5.0, namespace="etas")
async def get_etas(request: Request, response: Response):
    """
    Returns ETA information for each vehicle currently inside the geofence.
    """
    # Get vehicle IDs currently in geofence using cached query
    vehicle_ids_set = await get_vehicles_in_geofence(request.app.state.session_factory)
    vehicle_ids = list(vehicle_ids_set)

    if not vehicle_ids:
        return {}

    # Get latest ETAs
    etas_dict = await get_latest_etas(vehicle_ids, request.app.state.session_factory)

    return etas_dict


@router.get("/api/velocities")
@cache(soft_ttl=15, hard_ttl=300, lock_timeout=5.0, namespace="velocities")
async def get_velocities(request: Request, response: Response):
    """
    Returns predicted velocity and route matching data for each vehicle currently inside the geofence.
    """
    # Get vehicle IDs currently in geofence using cached query
    vehicle_ids_set = await get_vehicles_in_geofence(request.app.state.session_factory)
    vehicle_ids = list(vehicle_ids_set)

    if not vehicle_ids:
        return {}

    # Get latest velocities
    predicted_dict = await get_latest_velocities(vehicle_ids, request.app.state.session_factory)

    # Get route matching data from cached dataframe
    closest_points = await smart_closest_point(vehicle_ids)

    # Format response
    response_data = {}
    for vehicle_id in vehicle_ids:
        # Get velocity data
        velocity_data = predicted_dict.get(vehicle_id)

        # Get closest point result from smart_closest_point
        closest_distance, _, closest_route_name, polyline_index, _, stop_name = closest_points.get(
            vehicle_id,
            (None, None, None, None, None, None)
        )

        route_name = closest_route_name if closest_distance is not None and closest_distance < 0.050 else None

        # Determine if vehicle is at a stop
        is_at_stop = stop_name is not None
        current_stop = stop_name if is_at_stop else None

        response_data[vehicle_id] = {
            "speed_kmh": velocity_data["speed_kmh"] if velocity_data else None,
            "timestamp": velocity_data["timestamp"] if velocity_data else None,
            "route_name": route_name,
            "polyline_index": polyline_index,
            "is_at_stop": is_at_stop,
            "current_stop": current_stop,
        }

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
        await soft_clear_namespace("vehicles_in_geofence")

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

@router.get("/api/routes-v2")
async def get_shuttle_routes2(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Route)
        .options(
            selectinload(Route.stops)
            .selectinload(Stop.departure_polyline)
        )
    )

    routes = result.scalars().all()
    response = {}
    stop_obj = {}
    path_obj = {}
  
    #Loop through routes
    for route in routes:

        # Skip ENTRY and EXIT routes which are not real shuttle routes
        if route.name.startswith("ENTRY") or route.name.startswith("EXIT"):
            continue

        stops = sorted(route.stops, key=lambda s: s.id)
       
        full_path = []
        stop_list = []

        #loop through stops to build stops list and path
        for stop in stops:
           
            stop_obj[stop.name] = {
                "latitude": stop.latitude,
                "longitude": stop.longitude,
            }

            stop_list.append(stop.name)
         
        #Build path from polylines
        for stop in stops:
            for poly in stop.departure_polyline:
                coords = [
                    [float(lat), float(lng)]
                    for lat, lng in (c.split(",") for c in poly.coordinates)
                ]
                full_path.append(coords)
            path_obj[route.name] = full_path

        route_obj = {
            "color": route.route_color,
            "stops": stop_list,
            "path": route.name,
        }
    
        response[route.name] = route_obj
    
    response["stops"] = stop_obj
    response["polylines"] = path_obj
    
       

    return response

@router.get("/api/routes")
async def get_shuttle_routes(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Route)
        .options(
            selectinload(Route.stops)
            .selectinload(Stop.departure_polyline)
        )
    )

    routes = result.scalars().all()
    response = {}

    for route in routes:
        if route.name.startswith("ENTRY") or route.name.startswith("EXIT"):
            continue

        stops = sorted(route.stops, key=lambda s: s.id)

        route_obj = {
            "COLOR": route.route_color,
            "STOPS": [],
            "POLYLINE_STOPS": [],
            "ROUTES": []
        }

        # Build stops
        for i, stop in enumerate(stops):
            stop_key = stop.name.upper().replace(" ", "_")

            route_obj["STOPS"].append(stop_key)
            route_obj["POLYLINE_STOPS"].append(stop_key)

            route_obj[stop_key] = {
                "COORDINATES": [stop.latitude, stop.longitude],
                "OFFSET": i,
                "NAME": stop.name
            }

        # Build ROUTES (polyline segments)
        for stop in stops:
            for poly in stop.departure_polyline:
                coords = [
                    [float(lat), float(lng)]
                    for lat, lng in (c.split(",") for c in poly.coordinates)
                ]
                route_obj["ROUTES"].append(coords)

        response[route.name] = route_obj

    return response

@router.get("/api/schedule")
async def get_shuttle_schedule(db: AsyncSession = Depends(get_db)):

    result = await db.execute(
        select(DaySchedule)
        .options(
            selectinload(DaySchedule.bus_schedule_to_day_schedule)
            .selectinload(BusScheduleToDaySchedule.bus_schedule)
            .selectinload(BusSchedule.route_to_bus_schedules)
            .selectinload(RouteToBusSchedule.route)
        )
    )

    day_schedules = result.scalars().all()
    response = {}

    # Day type mapping
    DAY_TYPE_MAP = {
        "MONDAY": "weekday",
        "TUESDAY": "weekday",
        "WEDNESDAY": "weekday",
        "THURSDAY": "weekday",
        "FRIDAY": "weekday",
        "SATURDAY": "saturday",
        "SUNDAY": "sunday",
    }

    # Add mappings first
    response.update(DAY_TYPE_MAP)

    # Build schedules
    for day in day_schedules:
        day_type = day.name

        if day_type not in response:
            response[day_type] = {}

        for mapping in day.bus_schedule_to_day_schedule:
            bus = mapping.bus_schedule
            bus_name = bus.name

            if bus_name not in response[day_type]:
                response[day_type][bus_name] = []

            for rbs in bus.route_to_bus_schedules:
                time_str = rbs.time.strftime("%I:%M %p")  
                route_name = rbs.route.name

                response[day_type][bus_name].append([
                    time_str,
                    route_name
                ])

    return response

@router.get("/api/week-schedule")
async def get_week_schedule(db: AsyncSession = Depends(get_db)):
    # Implementation for weeks schedule
    response = {}
    date = datetime.now().date()
    sched_obj = {}
    for i in range(7):
        date = datetime.now().date() + timedelta(days=i)

        result = await db.execute(
            select(DateToDaySchedule)
            .where(DateToDaySchedule.date == date)
            .options(
                selectinload(DateToDaySchedule.day_schedule)
                .selectinload(DaySchedule.bus_schedule_to_day_schedule)
                .selectinload(BusScheduleToDaySchedule.bus_schedule)
                .selectinload(BusSchedule.route_to_bus_schedules)
                .selectinload(RouteToBusSchedule.route)
            )
        )

        date_schedule = result.scalar_one_or_none()
  
        response[date.isoformat()] = [] 
    
        #loop through bus schedules
        for mapping in date_schedule.day_schedule.bus_schedule_to_day_schedule:

            bus = mapping.bus_schedule

            sched_obj[bus.name] = {
                "route": bus.route_to_bus_schedules[0].route.name if bus.route_to_bus_schedules else None,
                "departures": []
            }
            response[date.isoformat()].append(bus.name)
            #loop through times for this bus
            for rbs in bus.route_to_bus_schedules:
        
                sched_obj[bus.name]["departures"].append(rbs.time.strftime("%H:%M"))
        
    response["schedules"] = sched_obj
    return response

@router.get("/api/aggregated-schedule")
async def get_aggregated_shuttle_schedule():
    """Serve aggregated_schedule.json file."""
    root_dir = Path(__file__).parent.parent.parent
    aggregated_file = root_dir / "shared" / "aggregated_schedule.json"
    if aggregated_file.exists():
        return FileResponse(aggregated_file)
    raise HTTPException(status_code=404, detail="Aggregated schedule file not found")


@router.get("/api/matched-schedules")
@cache(soft_ttl=3600, hard_ttl=86400, namespace="matched_schedules")
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
