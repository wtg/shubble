"""FastAPI routes for the Shubble API."""
import logging
import hmac
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path

from typing import Any, Optional

from fastapi import APIRouter, Request, Depends, HTTPException, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from sqlalchemy import and_, select

import json

from backend.cache import cache, get_redis, soft_clear_namespace
from backend.function_timer import timed
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import selectinload
from backend.database import get_db
from backend.models import Announcement, BusSchedule, BusScheduleToDaySchedule, DaySchedule, Route, RouteToBusSchedule, Stop, Vehicle, GeofenceEvent, VehicleLocation
from backend.config import settings
from backend.time_utils import get_campus_start_of_day, dev_now
from backend.utils import (
    get_vehicles_in_geofence,
)
from backend.fastapi.utils import (
    smart_closest_point,
    get_latest_vehicle_locations,
    get_current_driver_assignments,
    get_latest_velocities,
    _load_today_gap_windows,
    _in_schedule_gap,
)
from backend.fastapi.break_archetype import predict_on_break
from shared.stops import Stops

logger = logging.getLogger(__name__)

router = APIRouter()


# SSE stream keepalive interval. If no pub/sub message arrives, emit a
# comment line so intermediate proxies don't close the connection as idle.
_SSE_KEEPALIVE_SEC = 15.0
# Common SSE response headers. Disable nginx buffering so events flush
# to the client as soon as they're yielded.
_SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


async def _build_locations_payload(session_factory) -> tuple[dict[str, Any], Optional[str]]:
    """Assemble the /api/locations response body.

    Shared by the REST handler (GET /api/locations) and the SSE handler
    (GET /api/locations/stream). Returns (response_data, oldest_iso) so
    the REST path can still set data-age headers on cache miss.
    """
    results = await get_latest_vehicle_locations(session_factory)
    vehicle_ids = [loc["vehicle_id"] for loc in results]

    current_assignments = await get_current_driver_assignments(
        vehicle_ids, session_factory
    )

    # Schedule-gap detection (Phase 1 break detection, quick task
    # 260415-oeb). Computed once per payload build and reused for every
    # vehicle. Cached per-day inside _load_today_gap_windows so this is
    # effectively free after the first call.
    gap_windows = _load_today_gap_windows(settings.CAMPUS_TZ)
    now_utc = dev_now(timezone.utc)
    in_gap_now = _in_schedule_gap(now_utc, gap_windows)

    # Archetype-based break prediction (Phase 3, quick task 260415-owx).
    # Per-vehicle flag ORed with the schedule-gap flag below. Isolated
    # try/except: if the archetype path fails, schedule-gap detection
    # still drives `on_break` so Phase 1 behavior is preserved.
    try:
        archetype_breaks = await predict_on_break(
            vehicle_ids, session_factory, settings.CAMPUS_TZ
        )
    except Exception as e:
        logger.warning(f"predict_on_break failed; falling back to schedule-gap only: {e}")
        archetype_breaks = {}

    # PERF: ISO-8601 timestamps sort lexicographically in the same order
    # as chronological order (when UTC / same timezone), so we can find
    # the oldest via string min() and only do one fromisoformat call.
    response_data: dict[str, Any] = {}
    oldest_iso = min((loc["timestamp"] for loc in results), default=None)

    for loc in results:
        vehicle = loc["vehicle"]
        vid = loc["vehicle_id"]

        driver_info = None
        assignment = current_assignments.get(vid)
        if assignment and assignment.get("driver"):
            driver_data = assignment["driver"]
            driver_info = {
                "id": driver_data["id"],
                "name": driver_data["name"],
            }

        # Phase 1 (schedule-gap, fleet-wide) OR Phase 3 (archetype-matched,
        # per-vehicle). Either fires the muted-marker UX.
        on_break = in_gap_now or bool(archetype_breaks.get(vid, False))

        response_data[vid] = {
            "name": loc["name"],
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "timestamp": loc["timestamp"],  # Already ISO string
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
            "on_break": on_break,  # Phase 1 (fleet-gap) OR Phase 3 (per-vehicle archetype)
        }

    return response_data, oldest_iso


@router.get("/api/locations")
@timed
@cache(soft_ttl=3, hard_ttl=300, lock_timeout=0.0, namespace="locations")
async def get_locations(response: Response, request: Request):
    """
    Returns the latest location for each vehicle currently inside the geofence.
    The vehicle is considered inside the geofence if its latest geofence event
    today is a 'geofenceEntry'.

    This endpoint returns raw vehicle location data without ML predictions
    or route matching. Use /api/predictions for that data.
    """
    response_data, oldest_iso = await _build_locations_payload(
        request.app.state.session_factory
    )

    # Add timing metadata as HTTP headers to help frontend synchronize with Samsara API
    now = dev_now(timezone.utc)
    data_age = (
        (now - datetime.fromisoformat(oldest_iso)).total_seconds()
        if oldest_iso else None
    )

    response.headers['X-Server-Time'] = now.isoformat()
    response.headers['X-Oldest-Data-Time'] = oldest_iso or ''
    response.headers['X-Data-Age-Seconds'] = str(data_age) if data_age is not None else ''

    return response_data


@router.get("/api/locations/stream")
async def stream_locations(request: Request):
    """
    Server-Sent Events stream of /api/locations data. Pushes a fresh
    payload whenever the worker publishes to the `shubble:locations_updated`
    Redis channel (fired after each worker cycle that ingested new GPS rows).

    Client contract: each SSE message body is the same JSON shape as
    GET /api/locations. Comment lines (`: ka`) are keepalives and should
    be ignored.
    """
    session_factory = request.app.state.session_factory
    redis = get_redis()

    async def gen():
        # Initial emit so the client doesn't wait up to _SSE_KEEPALIVE_SEC
        # or a worker cycle for their first data point.
        try:
            payload, _ = await _build_locations_payload(session_factory)
            yield f"data: {json.dumps(payload)}\n\n"
        except Exception as e:
            logger.warning(f"stream_locations initial emit failed: {e}")

        if redis is None:
            # Without Redis we can't subscribe to pub/sub. Close the stream;
            # the client will fall back to polling.
            return

        pubsub = redis.pubsub()
        try:
            await pubsub.subscribe("shubble:locations_updated")
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=_SSE_KEEPALIVE_SEC,
                    )
                except Exception as e:
                    logger.warning(f"stream_locations pubsub error: {e}")
                    break

                if msg is None:
                    # Keepalive so proxies don't time out the idle connection.
                    yield ": ka\n\n"
                    continue

                try:
                    payload, _ = await _build_locations_payload(session_factory)
                    yield f"data: {json.dumps(payload)}\n\n"
                except Exception as e:
                    logger.warning(f"stream_locations payload build failed: {e}")
        finally:
            try:
                await pubsub.unsubscribe("shubble:locations_updated")
            except Exception:
                pass
            try:
                await pubsub.close()
            except Exception:
                pass

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.get("/api/trips")
@timed
async def get_trips(request: Request, response: Response):
    """Returns per-trip ETAs. Each trip is a (route, departure_time) pair
    with its assigned shuttle's stop ETAs. See backend/worker/trips.py.
    """
    redis = get_redis()
    if redis:
        raw = await redis.get("shubble:trips_live")
        if raw:
            return json.loads(raw)
    return []


@router.get("/api/trips/stream")
async def stream_trips(request: Request):
    """
    Server-Sent Events stream of /api/trips data. Pushes the latest
    `shubble:trips_live` payload whenever the worker publishes to the
    `shubble:trips_updated` Redis channel (fired at the end of each worker
    cycle).

    Client contract: each SSE message body is the same JSON array as
    GET /api/trips. Comment lines (`: ka`) are keepalives.
    """
    redis = get_redis()

    async def _current_payload() -> str:
        """Return the SSE data frame for the current Redis trips blob."""
        if redis is None:
            return "data: []\n\n"
        try:
            raw = await redis.get("shubble:trips_live")
        except Exception as e:
            logger.warning(f"stream_trips Redis read failed: {e}")
            return "data: []\n\n"
        if raw is None:
            return "data: []\n\n"
        text = raw.decode() if isinstance(raw, (bytes, bytearray)) else raw
        return f"data: {text}\n\n"

    async def gen():
        # Initial emit so the client gets the current snapshot immediately.
        yield await _current_payload()

        if redis is None:
            return

        pubsub = redis.pubsub()
        try:
            await pubsub.subscribe("shubble:trips_updated")
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=_SSE_KEEPALIVE_SEC,
                    )
                except Exception as e:
                    logger.warning(f"stream_trips pubsub error: {e}")
                    break

                if msg is None:
                    yield ": ka\n\n"
                    continue

                yield await _current_payload()
        finally:
            try:
                await pubsub.unsubscribe("shubble:trips_updated")
            except Exception:
                pass
            try:
                await pubsub.close()
            except Exception:
                pass

    return StreamingResponse(
        gen(), media_type="text/event-stream", headers=_SSE_HEADERS
    )


@router.get("/api/velocities")
@timed
@cache(soft_ttl=3, hard_ttl=300, lock_timeout=5.0, namespace="velocities")
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

    # Get latest locations for real-time route matching fallback
    locations_dict = await get_latest_vehicle_locations(request.app.state.session_factory)
    locations_by_id = {loc["vehicle_id"]: loc for loc in locations_dict}

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

        # Fallback: real-time route matching when dataframe has no route data
        if route_name is None:
            loc = locations_by_id.get(vehicle_id)
            if loc:
                rt = Stops.get_closest_point((loc["latitude"], loc["longitude"]))
                if rt[0] is not None and rt[0] < 0.050:
                    closest_distance, _, route_name, polyline_index, _ = rt[0], rt[1], rt[2], rt[3], rt[4]

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
@timed
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
@timed
async def data_today(db: AsyncSession = Depends(get_db)):
    """Get all location data and geofence events for today."""
    now = dev_now(timezone.utc)
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

    # Build response dict. setdefault avoids the "if in dict else new"
    # branching and makes geofence-before-location edge cases safe.
    locations_today_dict: dict = {}
    for location in locations_today:
        entry = locations_today_dict.setdefault(
            location.vehicle_id, {"entry": None, "exit": None, "data": []}
        )
        entry["data"].append({
            "latitude": location.latitude,
            "longitude": location.longitude,
            "timestamp": location.timestamp,
            "speed_mph": location.speed_mph,
            "heading_degrees": location.heading_degrees,
            "address_id": location.address_id,
        })

    for geofence_event in events_today:
        # Guard against events for vehicles with no location rows — previously
        # this raised KeyError and crashed the endpoint.
        vehicle_entry = locations_today_dict.setdefault(
            geofence_event.vehicle_id, {"entry": None, "exit": None, "data": []}
        )
        if geofence_event.event_type == "geofenceEntry":
            if vehicle_entry["entry"] is None:  # first entry wins
                vehicle_entry["entry"] = geofence_event.event_time
        elif geofence_event.event_type == "geofenceExit":
            if vehicle_entry["entry"] is not None:  # only after an entry
                vehicle_entry["exit"] = geofence_event.event_time

    return locations_today_dict

@router.get("/api/routes")
@timed
async def get_shuttle_routes(db: AsyncSession = Depends(get_db)):

    result = await db.execute(
        select(Route)
        .options(
            selectinload(Route.stops)
            .selectinload(Stop.departure_polyline)
        )
    )

    routes = result.scalars().all()
    response = []

    #Loop through routes
    for route in routes:

        # Skip ENTRY and EXIT routes which are not real shuttle routes
        if route.name.startswith("ENTRY") or route.name.startswith("EXIT"):
            continue

        stops = sorted(route.stops, key=lambda s: s.id)

        stops_list = []
        full_path = []

        #loop through stops to build stops list and path
        for stop in stops:
            stops_list.append({
                "name": stop.name,
                "latitude": stop.latitude,
                "longitude": stop.longitude,
            })

        #Build path from polylines
        for stop in stops:
            for poly in stop.departure_polyline:
                coords = [
                    [float(lat), float(lng)]
                    for lat, lng in (c.split(",") for c in poly.coordinates)
                ]
                full_path.extend(coords)

        response.append({
            "name": route.name,
            "color": route.route_color,
            "stops": stops_list,
            "path": full_path,
        })

    return response

@router.get("/api/schedule")
@timed
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
    response = []

    for day in day_schedules:

        day_obj = {
            "day_type": day.name,
            "buses": []
        }

        #loop through bus schedules
        for mapping in day.bus_schedule_to_day_schedule:
            bus = mapping.bus_schedule

            bus_obj = {
                "name": bus.name,
                "departures": []
            }

            #loop through times for this bus
            for rbs in bus.route_to_bus_schedules:
               
                departure = {
                    "time": rbs.time.strftime("%H:%M"),   
                    "route": rbs.route.name   
                }

                bus_obj["departures"].append(departure)

            day_obj["buses"].append(bus_obj)

        response.append(day_obj)

    return response

@router.get("/api/aggregated-schedule")
@timed
async def get_aggregated_shuttle_schedule():
    """Serve aggregated_schedule.json file."""
    root_dir = Path(__file__).parent.parent.parent
    aggregated_file = root_dir / "shared" / "aggregated_schedule.json"
    if aggregated_file.exists():
        return FileResponse(aggregated_file)
    raise HTTPException(status_code=404, detail="Aggregated schedule file not found")


@router.get("/api/matched-schedules")
@timed
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

@router.get("/api/announcements")
@cache(soft_ttl=900, hard_ttl=3600, namespace="announcements")
async def data_announcement(db: AsyncSession = Depends(get_db)):

    # Query announcements that are active and not expired
    now = dev_now(timezone.utc)
    announcements_query = (
        select(Announcement)
        .where(
            and_(
                Announcement.active == True,
                Announcement.expires_at >= now,
            )
        )
        # Order by most recent first
        .order_by(Announcement.created_at.desc())
    )
    announcements_result = await db.execute(announcements_query)
    announcements = announcements_result.scalars().all()
    return announcements