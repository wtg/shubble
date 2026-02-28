"""SQLAdmin model views for the Shubble db tables."""
from sqladmin import ModelView
from backend.models import (
    Vehicle,
    GeofenceEvent,
    VehicleLocation,
    Driver,
    DriverVehicleAssignment,
    ETA,
    PredictedLocation,
)


class VehicleAdmin(ModelView, model=Vehicle):
    name = "Vehicle"
    name_plural = "Vehicles"
    
    column_list = [Vehicle.id, Vehicle.name, Vehicle.asset_type, Vehicle.license_plate, Vehicle.vin]
    column_searchable_list = [Vehicle.id, Vehicle.name, Vehicle.license_plate]
    column_sortable_list = [Vehicle.id, Vehicle.name, Vehicle.asset_type]
    column_default_sort = [(Vehicle.name, False)]


class GeofenceEventAdmin(ModelView, model=GeofenceEvent):
    name = "Geofence Event"
    name_plural = "Geofence Events"
    
    column_list = [GeofenceEvent.id, GeofenceEvent.vehicle_id, GeofenceEvent.event_type, 
                   GeofenceEvent.event_time, GeofenceEvent.address_name]
    column_searchable_list = [GeofenceEvent.id, GeofenceEvent.vehicle_id, GeofenceEvent.address_name]
    column_sortable_list = [GeofenceEvent.vehicle_id, GeofenceEvent.event_type, GeofenceEvent.event_time]
    column_default_sort = [(GeofenceEvent.event_time, True)]


class VehicleLocationAdmin(ModelView, model=VehicleLocation):
    name = "Vehicle Location"
    name_plural = "Vehicle Locations"
    
    column_list = [VehicleLocation.id, VehicleLocation.vehicle_id, VehicleLocation.timestamp,
                   VehicleLocation.latitude, VehicleLocation.longitude, VehicleLocation.speed_mph]
    column_searchable_list = [VehicleLocation.vehicle_id]
    column_sortable_list = [VehicleLocation.vehicle_id, VehicleLocation.timestamp, VehicleLocation.speed_mph]
    column_default_sort = [(VehicleLocation.timestamp, True)]
    page_size = 50
    page_size_options = [25, 50, 100, 200]


class DriverAdmin(ModelView, model=Driver):
    name = "Driver"
    name_plural = "Drivers"
    
    column_list = [Driver.id, Driver.name, Driver.created_at]
    column_searchable_list = [Driver.id, Driver.name]
    column_sortable_list = [Driver.id, Driver.name, Driver.created_at]
    column_default_sort = [(Driver.name, False)]


class DriverVehicleAssignmentAdmin(ModelView, model=DriverVehicleAssignment):
    name = "Driver Assignment"
    name_plural = "Driver Assignments"
    
    column_list = [DriverVehicleAssignment.id, DriverVehicleAssignment.driver_id,
                   DriverVehicleAssignment.vehicle_id, DriverVehicleAssignment.assignment_start,
                   DriverVehicleAssignment.assignment_end]
    column_searchable_list = [DriverVehicleAssignment.driver_id, DriverVehicleAssignment.vehicle_id]
    column_sortable_list = [DriverVehicleAssignment.assignment_start, DriverVehicleAssignment.assignment_end]
    column_default_sort = [(DriverVehicleAssignment.assignment_start, True)]


class ETAAdmin(ModelView, model=ETA):
    name = "ETA"
    name_plural = "ETAs"
    
    column_list = [ETA.id, ETA.vehicle_id, ETA.timestamp, ETA.etas]
    column_searchable_list = [ETA.vehicle_id]
    column_sortable_list = [ETA.vehicle_id, ETA.timestamp]
    column_default_sort = [(ETA.timestamp, True)]
    page_size = 50
    page_size_options = [25, 50, 100, 200]


class PredictedLocationAdmin(ModelView, model=PredictedLocation):
    name = "Predicted Location"
    name_plural = "Predicted Locations"
    
    column_list = [PredictedLocation.id, PredictedLocation.vehicle_id,
                   PredictedLocation.timestamp, PredictedLocation.speed_kmh]
    column_searchable_list = [PredictedLocation.vehicle_id]
    column_sortable_list = [PredictedLocation.vehicle_id, PredictedLocation.timestamp, PredictedLocation.speed_kmh]
    column_default_sort = [(PredictedLocation.timestamp, True)]
    page_size = 50
    page_size_options = [25, 50, 100, 200]
