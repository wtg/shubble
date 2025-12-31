"""Backend package - re-exports for convenient imports."""
from backend.flask import app
from backend.config import settings
from backend.database import Base, create_async_db_engine, create_session_factory, get_db
from backend.models import Vehicle, GeofenceEvent, VehicleLocation, Driver, DriverVehicleAssignment
from backend.time_utils import get_campus_start_of_day
from backend.utils import get_vehicles_in_geofence_query, get_vehicles_in_geofence

__all__ = [
    "app",
    "settings",
    "Base",
    "create_async_db_engine",
    "create_session_factory",
    "get_db",
    "Vehicle",
    "GeofenceEvent",
    "VehicleLocation",
    "Driver",
    "DriverVehicleAssignment",
    "get_campus_start_of_day",
    "get_vehicles_in_geofence_query",
    "get_vehicles_in_geofence",
]
