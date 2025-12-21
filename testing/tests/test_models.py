"""
Unit tests for database models
"""
import pytest
from datetime import datetime
from backend.models import Vehicle, GeofenceEvent, VehicleLocation, Driver, DriverVehicleAssignment


@pytest.mark.unit
def test_vehicle_creation():
    """Test Vehicle model instantiation"""
    vehicle = Vehicle(
        id="test_vehicle_1",
        name="Test Shuttle",
        asset_type="vehicle",
        license_plate="ABC123",
        vin="1234567890"
    )
    assert vehicle.id == "test_vehicle_1"
    assert vehicle.name == "Test Shuttle"
    assert vehicle.license_plate == "ABC123"


@pytest.mark.unit
def test_geofence_event_creation():
    """Test GeofenceEvent model instantiation"""
    event = GeofenceEvent(
        id="event_1",
        vehicle_id="test_vehicle_1",
        event_type="geofenceEntry",
        event_time=datetime.utcnow()
    )
    assert event.event_type == "geofenceEntry"
    assert event.vehicle_id == "test_vehicle_1"


@pytest.mark.unit
def test_vehicle_location_creation():
    """Test VehicleLocation model instantiation"""
    location = VehicleLocation(
        vehicle_id="test_vehicle_1",
        timestamp=datetime.utcnow(),
        latitude=42.7284,
        longitude=-73.6918,
        heading_degrees=90.0,
        speed_mph=15.0
    )
    assert location.latitude == 42.7284
    assert location.longitude == -73.6918
    assert location.speed_mph == 15.0


@pytest.mark.unit
def test_driver_creation():
    """Test Driver model instantiation"""
    driver = Driver(
        id="driver_1",
        name="Test Driver"
    )
    assert driver.id == "driver_1"
    assert driver.name == "Test Driver"


@pytest.mark.unit
def test_driver_vehicle_assignment_creation():
    """Test DriverVehicleAssignment model instantiation"""
    assignment = DriverVehicleAssignment(
        driver_id="driver_1",
        vehicle_id="vehicle_1",
        assignment_start=datetime.utcnow(),
        assignment_end=None
    )
    assert assignment.driver_id == "driver_1"
    assert assignment.vehicle_id == "vehicle_1"
    assert assignment.assignment_end is None
