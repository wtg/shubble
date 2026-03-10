"""Utility functions for FastAPI."""
from datetime import datetime, timezone
from sqlalchemy import func, and_, select
from sqlalchemy.orm import selectinload
from typing import Dict, Tuple, Optional, List, TypedDict, Any
import pandas as pd

from backend.cache import cache
from backend.function_timer import timed

from backend.models import VehicleLocation, DriverVehicleAssignment, ETA, PredictedLocation
from backend.cache_dataframe import get_today_dataframe
from backend.time_utils import get_campus_start_of_day
from backend.utils import get_vehicles_in_geofence, get_vehicles_in_geofence_query

import logging

logger = logging.getLogger(__name__)


class VehicleInfoDict(TypedDict):
    license_plate: Optional[str]
    vin: Optional[str]
    asset_type: str
    gateway_model: Optional[str]
    gateway_serial: Optional[str]


class VehicleLocationDict(TypedDict):
    vehicle_id: str
    name: Optional[str]
    latitude: float
    longitude: float
    timestamp: str
    heading_degrees: Optional[float]
    speed_mph: Optional[float]
    is_ecu_speed: bool
    formatted_location: Optional[str]
    address_id: Optional[str]
    address_name: Optional[str]
    vehicle: VehicleInfoDict


class DriverInfoDict(TypedDict):
    id: str
    name: str


class DriverAssignmentDict(TypedDict):
    vehicle_id: str
    driver: Optional[DriverInfoDict]


class ETADict(TypedDict):
    stop_times: dict
    timestamp: str


class VelocityDict(TypedDict):
    speed_kmh: float
    timestamp: str


@timed
@cache(soft_ttl=15, hard_ttl=300, lock_timeout=5.0, namespace="smart_closest_point")
async def smart_closest_point(
    vehicle_ids: List[str]
) -> Dict[str, Tuple[Optional[float], Optional[Tuple[float, float]], Optional[str], Optional[int], Optional[int], Optional[str]]]:
    """
    Get the closest point data for each vehicle from the cached dataframe (cached for 15 seconds).

    The cached dataframe (from get_today_dataframe) already contains preprocessed
    route matching data from the ML pipeline. This function simply retrieves the
    latest row for each vehicle and extracts the relevant columns.

    Args:
        vehicle_ids: List of vehicle IDs to get closest point data for

    Returns:
        Dictionary mapping vehicle_id to (distance, closest_point, route_name, polyline_idx, segment_idx, stop_name)
        Returns (None, None, None, None, None, None) for vehicles not found in the cache.
    """
    results = {}

    try:
        # Load cached dataframe with preprocessed route information
        df = await get_today_dataframe()

        if df.empty:
            # No cached data, return None for all vehicles
            for vehicle_id in vehicle_ids:
                results[vehicle_id] = (None, None, None, None, None, None)
            return results

        df['vehicle_id'] = df['vehicle_id'].astype(str)
        grouped = df.groupby('vehicle_id')

        # Get the latest row for each vehicle
        for vehicle_id in vehicle_ids:
            vid_str = str(vehicle_id)
            if vid_str not in grouped.groups:
                results[vehicle_id] = (None, None, None, None, None, None)
                continue

            latest = grouped.get_group(vid_str).iloc[-1]

            # Extract closest point data from the preprocessed columns
            # These columns are added by ml/data/preprocess.py pipeline
            distance = latest.get('dist_to_route')
            route_name = latest.get('route')
            polyline_idx = latest.get('polyline_idx')
            segment_idx = latest.get('segment_idx')
            closest_lat = latest.get('closest_lat')
            closest_lon = latest.get('closest_lon')
            stop_name = latest.get('stop_name')  # From stops pipeline

            # Build closest_point tuple if coordinates are available
            if closest_lat is not None and closest_lon is not None:
                closest_point = (float(closest_lat), float(closest_lon))
            else:
                closest_point = None

            # Convert polyline_idx and segment_idx to int if they exist
            if polyline_idx is not None:
                try:
                    polyline_idx = int(polyline_idx)
                except (ValueError, TypeError):
                    polyline_idx = None

            if segment_idx is not None:
                try:
                    segment_idx = int(segment_idx)
                except (ValueError, TypeError):
                    segment_idx = None

            # Convert stop_name to string or None
            if stop_name is not None and not pd.isna(stop_name):
                stop_name = str(stop_name)
            else:
                stop_name = None

            results[vehicle_id] = (distance, closest_point, route_name, polyline_idx, segment_idx, stop_name)

    except Exception as e:
        # If anything goes wrong, return None for all vehicles
        logger.error(f"Error in smart_closest_point: {e}")

        for vehicle_id in vehicle_ids:
            results[vehicle_id] = (None, None, None, None, None, None)

    return results


@timed
@cache(soft_ttl=15, hard_ttl=300, lock_timeout=0.0, namespace="locations")
async def get_latest_vehicle_locations(session_factory) -> List[VehicleLocationDict]:
    """
    Get the latest location for each vehicle currently inside the geofence.

    Args:
        session_factory: Async session factory

    Returns:
        List of VehicleLocationDict with timestamps as ISO strings
    """
    async with session_factory() as db:
        # Get query for vehicles in geofence and convert to subquery
        geofence_entries = get_vehicles_in_geofence_query().subquery()

        query = (
            select(VehicleLocation)
            .where(
                VehicleLocation.vehicle_id.in_(select(geofence_entries.c.vehicle_id)),
                VehicleLocation.timestamp >= get_campus_start_of_day(),
            )
            .order_by(
                VehicleLocation.vehicle_id,
                VehicleLocation.timestamp.desc()
            )
            .distinct(VehicleLocation.vehicle_id)
            .options(selectinload(VehicleLocation.vehicle))
        )

        result = await db.execute(query)
        locations = result.scalars().all()

        # Convert to serializable dicts
        location_dicts: List[VehicleLocationDict] = []
        for loc in locations:
            vehicle = loc.vehicle
            vehicle_dict: VehicleInfoDict = {
                "license_plate": vehicle.license_plate,
                "vin": vehicle.vin,
                "asset_type": vehicle.asset_type,
                "gateway_model": vehicle.gateway_model,
                "gateway_serial": vehicle.gateway_serial,
            }

            loc_dict: VehicleLocationDict = {
                "vehicle_id": loc.vehicle_id,
                "name": loc.name,
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "timestamp": loc.timestamp.isoformat(),
                "heading_degrees": loc.heading_degrees,
                "speed_mph": loc.speed_mph,
                "is_ecu_speed": loc.is_ecu_speed,
                "formatted_location": loc.formatted_location,
                "address_id": loc.address_id,
                "address_name": loc.address_name,
                "vehicle": vehicle_dict,
            }
            location_dicts.append(loc_dict)

        return location_dicts


@timed
@cache(soft_ttl=900, hard_ttl=3600, namespace="driver_assignments")
async def get_current_driver_assignments(
    vehicle_ids: List[str], session_factory
) -> Dict[str, DriverAssignmentDict]:
    """
    Get the current driver assignments for a list of vehicles.

    Args:
        vehicle_ids: List of vehicle IDs to get assignments for
        session_factory: Async session factory

    Returns:
        Dictionary mapping vehicle_id to DriverAssignmentDict
    """
    if not vehicle_ids:
        return {}

    async with session_factory() as db:
        assignments_query = (
            select(DriverVehicleAssignment)
            .where(
                DriverVehicleAssignment.vehicle_id.in_(vehicle_ids),
                DriverVehicleAssignment.assignment_end.is_(None),
            )
        ).options(selectinload(DriverVehicleAssignment.driver))

        assignments_result = await db.execute(assignments_query)
        assignments = assignments_result.scalars().all()

        result_dict: Dict[str, DriverAssignmentDict] = {}
        for assignment in assignments:
            driver_dict: Optional[DriverInfoDict] = None
            if assignment.driver:
                driver_dict = {
                    "id": assignment.driver.id,
                    "name": assignment.driver.name,
                }

            result_dict[assignment.vehicle_id] = {
                "vehicle_id": assignment.vehicle_id,
                "driver": driver_dict,
            }

        return result_dict


@timed
@cache(soft_ttl=15, hard_ttl=300, lock_timeout=5.0, namespace="etas")
async def get_latest_etas(vehicle_ids: List[str], session_factory) -> Dict[str, ETADict]:
    """
    Get the latest ETA for each vehicle.

    Args:
        vehicle_ids: List of vehicle IDs
        session_factory: Async session factory

    Returns:
        Dictionary mapping vehicle_id to ETADict
    """
    if not vehicle_ids:
        return {}

    async with session_factory() as db:
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

        etas_dict: Dict[str, ETADict] = {}
        for eta in etas:
            etas_dict[eta.vehicle_id] = {
                "stop_times": eta.etas,
                "timestamp": eta.timestamp.isoformat(),
            }

        return etas_dict


@timed
@cache(soft_ttl=15, hard_ttl=300, lock_timeout=5.0, namespace="velocities")
async def get_latest_velocities(vehicle_ids: List[str], session_factory) -> Dict[str, VelocityDict]:
    """
    Get the latest predicted velocity for each vehicle.

    Args:
        vehicle_ids: List of vehicle IDs
        session_factory: Async session factory

    Returns:
        Dictionary mapping vehicle_id to VelocityDict
    """
    if not vehicle_ids:
        return {}

    async with session_factory() as db:
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

        predicted_dict: Dict[str, VelocityDict] = {}
        for pred in predicted_locations:
            predicted_dict[pred.vehicle_id] = {
                "speed_kmh": pred.speed_kmh,
                "timestamp": pred.timestamp.isoformat(),
            }

        return predicted_dict


# ──────────────────────────────────────────────────────────────────────
# Response builder functions
#
# These build the full API response dicts, reusable by both the REST
# endpoints (routes.py) and the SSE stream (sse.py).
# ──────────────────────────────────────────────────────────────────────


class TimingMetadata(TypedDict):
    server_time: str
    oldest_data_time: str
    data_age_seconds: str


async def build_locations_response(
    session_factory,
) -> Tuple[dict, TimingMetadata]:
    """Build the /api/locations response data.

    Queries the latest location for each vehicle in the geofence,
    merges with driver assignment info, and computes timing metadata.

    Args:
        session_factory: Async session factory

    Returns:
        Tuple of (response_data dict, timing_metadata dict)
    """
    results = await get_latest_vehicle_locations(session_factory)

    vehicle_ids = [loc["vehicle_id"] for loc in results]
    current_assignments = await get_current_driver_assignments(
        vehicle_ids, session_factory
    )

    response_data = {}
    oldest_timestamp = None

    for loc in results:
        vehicle = loc["vehicle"]
        ts = datetime.fromisoformat(loc["timestamp"])
        if oldest_timestamp is None or ts < oldest_timestamp:
            oldest_timestamp = ts

        driver_info = None
        assignment = current_assignments.get(loc["vehicle_id"])
        if assignment and assignment.get("driver"):
            driver_data = assignment["driver"]
            driver_info = {
                "id": driver_data["id"],
                "name": driver_data["name"],
            }

        response_data[loc["vehicle_id"]] = {
            "name": loc["name"],
            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "timestamp": loc["timestamp"],
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

    now = datetime.now(timezone.utc)
    data_age = (
        (now - oldest_timestamp).total_seconds() if oldest_timestamp else None
    )

    timing: TimingMetadata = {
        "server_time": now.isoformat(),
        "oldest_data_time": oldest_timestamp.isoformat() if oldest_timestamp else "",
        "data_age_seconds": str(data_age) if data_age is not None else "",
    }

    return response_data, timing


async def build_velocities_response(session_factory) -> dict:
    """Build the /api/velocities response data.

    Queries predicted velocities and route matching data for all
    vehicles currently in the geofence.

    Args:
        session_factory: Async session factory

    Returns:
        Response data dict keyed by vehicle_id
    """
    vehicle_ids_set = await get_vehicles_in_geofence(session_factory)
    vehicle_ids = list(vehicle_ids_set)

    if not vehicle_ids:
        return {}

    predicted_dict = await get_latest_velocities(vehicle_ids, session_factory)
    closest_points = await smart_closest_point(vehicle_ids)

    response_data = {}
    for vehicle_id in vehicle_ids:
        velocity_data = predicted_dict.get(vehicle_id)

        closest_distance, _, closest_route_name, polyline_index, _, stop_name = (
            closest_points.get(
                vehicle_id, (None, None, None, None, None, None)
            )
        )

        route_name = (
            closest_route_name
            if closest_distance is not None and closest_distance < 0.050
            else None
        )

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


async def build_etas_response(session_factory) -> dict:
    """Build the /api/etas response data.

    Queries the latest ETA predictions for all vehicles currently
    in the geofence.

    Args:
        session_factory: Async session factory

    Returns:
        Response data dict keyed by vehicle_id
    """
    vehicle_ids_set = await get_vehicles_in_geofence(session_factory)
    vehicle_ids = list(vehicle_ids_set)

    if not vehicle_ids:
        return {}

    return await get_latest_etas(vehicle_ids, session_factory)
