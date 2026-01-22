"""Utility functions for FastAPI."""
from sqlalchemy import func, and_, select
from sqlalchemy.orm import selectinload
from typing import Dict, Tuple, Optional, List, TypedDict, Any
import pandas as pd

from backend.cache import cache

from backend.models import VehicleLocation, DriverVehicleAssignment, ETA, PredictedLocation
from backend.cache_dataframe import get_today_dataframe
from backend.utils import get_vehicles_in_geofence_query


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


@cache(expire=15, namespace="smart_closest_point")
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

        # Ensure vehicle_id is string type for comparison
        df['vehicle_id'] = df['vehicle_id'].astype(str)

        # Get the latest row for each vehicle
        for vehicle_id in vehicle_ids:
            vehicle_data = df[df['vehicle_id'] == str(vehicle_id)]

            if vehicle_data.empty:
                # No data for this vehicle
                results[vehicle_id] = (None, None, None, None, None, None)
                continue

            # Get the latest row (dataframe is sorted by timestamp)
            latest = vehicle_data.iloc[-1]

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
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error in smart_closest_point: {e}")

        for vehicle_id in vehicle_ids:
            results[vehicle_id] = (None, None, None, None, None, None)

    return results


@cache(expire=15, namespace="locations")
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


@cache(expire=900, namespace="driver_assignments")
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


@cache(expire=15, namespace="etas")
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


@cache(expire=15, namespace="velocities")
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