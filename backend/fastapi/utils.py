"""Utility functions for FastAPI."""
from datetime import datetime, timezone
from sqlalchemy import func, and_, select
from sqlalchemy.orm import selectinload
from typing import Dict, Tuple, Optional, List, TypedDict
import pandas as pd

from backend.cache import cache
from backend.function_timer import timed

from backend.models import VehicleLocation, DriverVehicleAssignment, ETA, PredictedLocation
from backend.cache_dataframe import get_today_dataframe
from backend.utils import get_vehicles_in_geofence_query
from backend.time_utils import dev_now, get_campus_start_of_day
from shared.stops import Stops

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


class StopETADict(TypedDict):
    eta: str
    vehicle_id: str
    route: str
    last_arrival: Optional[str]


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

        # PERF + correctness: no longer mutating the shared cached df.
        # vehicle_id comes from the DB as String already; groupby on the
        # existing column works without forcing an astype dance.
        grouped = df.groupby('vehicle_id', sort=False)

        # Get the latest row for each vehicle
        for vehicle_id in vehicle_ids:
            vid_str = str(vehicle_id)
            if vid_str not in grouped.groups:
                results[vehicle_id] = (None, None, None, None, None, None)
                continue

            vehicle_rows = grouped.get_group(vid_str)
            latest = vehicle_rows.iloc[-1]

            # Extract closest point data from the preprocessed columns
            # These columns are added by ml/data/preprocess.py pipeline
            distance = latest.get('dist_to_route')
            route_name = latest.get('route')
            polyline_idx = latest.get('polyline_idx')
            segment_idx = latest.get('segment_idx')
            closest_lat = latest.get('closest_lat')
            closest_lon = latest.get('closest_lon')
            stop_name = latest.get('stop_name')  # From stops pipeline

            def _is_na(v):
                return v is None or (isinstance(v, float) and pd.isna(v))

            # If the latest row has no route (e.g. shuttle currently at
            # Union where multiple routes are ambiguous), walk
            # BACKWARD through recent history to find the most recent
            # row with a valid route. This eliminates the transient
            # "Vehicle X has no route_name" that appears right after a
            # shuttle starts or wraps between loops.
            if _is_na(route_name):
                # Only walk back a handful of rows — a stale route
                # from 30+ minutes ago isn't useful.
                look_back = min(len(vehicle_rows), 12)
                for i in range(2, look_back + 1):
                    prev = vehicle_rows.iloc[-i]
                    prev_route = prev.get('route')
                    if not _is_na(prev_route):
                        route_name = prev_route
                        # Also prefer the same row's closest-point
                        # info so the response is internally coherent.
                        distance = prev.get('dist_to_route')
                        polyline_idx = prev.get('polyline_idx')
                        segment_idx = prev.get('segment_idx')
                        closest_lat = prev.get('closest_lat')
                        closest_lon = prev.get('closest_lon')
                        break

            # Fallback: if still no route data, use real-time route matching
            if _is_na(route_name):
                lat = latest.get('latitude')
                lon = latest.get('longitude')
                if lat is not None and lon is not None:
                    rt_result = Stops.get_closest_point((float(lat), float(lon)))
                    if rt_result[0] is not None:
                        distance, closest_point_arr, route_name, polyline_idx, segment_idx = rt_result
                        closest_lat = closest_point_arr[0] if closest_point_arr is not None else None
                        closest_lon = closest_point_arr[1] if closest_point_arr is not None else None

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
async def get_latest_etas(vehicle_ids: List[str], session_factory) -> Dict[str, StopETADict]:
    """
    Get per-stop ETAs: for each stop, the next shuttle to arrive.

    Queries the latest ETA row per vehicle from the database, then aggregates
    across all vehicles to find the earliest future ETA for each stop.
    Also includes last_arrival timestamps from vehicle_locations.

    Args:
        vehicle_ids: List of vehicle IDs
        session_factory: Async session factory

    Returns:
        Dictionary mapping stop_key to StopETADict
    """
    if not vehicle_ids:
        return {}

    now_utc = dev_now(timezone.utc)

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

        # Aggregate per-stop: pick earliest future ETA across all vehicles
        per_stop: Dict[str, StopETADict] = {}
        for eta_row in etas:
            if not eta_row.etas:
                continue
            for stop_key, stop_data in eta_row.etas.items():
                # Handle both new format {eta, route} and legacy format (plain ISO string)
                if isinstance(stop_data, dict):
                    eta_iso = stop_data.get("eta", "")
                    route = stop_data.get("route", "")
                else:
                    eta_iso = stop_data
                    route = ""

                try:
                    eta_dt = datetime.fromisoformat(eta_iso)
                except (ValueError, TypeError):
                    continue
                if eta_dt <= now_utc:
                    continue  # Skip past ETAs

                last_arrival = stop_data.get("last_arrival") if isinstance(stop_data, dict) else None

                entry = {
                    "eta": eta_iso,
                    "vehicle_id": eta_row.vehicle_id,
                    "route": route,
                    "last_arrival": last_arrival,
                }

                # Skip route-qualified keys from DB storage (they're re-generated below)
                if ":" in stop_key:
                    continue

                # Primary key: bare stop name — keep earliest across routes
                if stop_key not in per_stop or eta_dt < datetime.fromisoformat(per_stop[stop_key]["eta"]):
                    per_stop[stop_key] = entry

                # Secondary key: route-qualified — ensures each route's ETA is preserved
                # so the frontend can show the correct one for the selected route
                if route:
                    route_key = f"{stop_key}:{route}"
                    if route_key not in per_stop or eta_dt < datetime.fromisoformat(per_stop[route_key]["eta"]):
                        per_stop[route_key] = entry

        return per_stop


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
