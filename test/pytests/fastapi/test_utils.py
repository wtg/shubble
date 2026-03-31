import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from sqlalchemy import literal_column, select

# Testing 2 utils first
from backend.fastapi.utils import (
    get_latest_vehicle_locations,
    VehicleLocationDict,
)


# Mimics the async context manager of SQLAlchemy session factory
# we need support for __aenter and __aexit
# __aenter__ returns mock db
class AsyncSessionContextManager:
    def __init__(self, db_mock):
        self.db_mock = db_mock

    async def __aenter__(self):
        return self.db_mock

    async def __aexit__(self, *args):
        pass


# fixes couroutine issue with async def
def make_session_factory(db_mock):
    def session_factory():
        return AsyncSessionContextManager(db_mock)
    return session_factory

# make mock DB 
# .scalars().all() is row list similar to production rows = result.scalaras().all()
# currently holds row, in row we hold make_location info
def make_db(rows: list):
    mock_result = MagicMock()
    mock_result.scalars.return_value.all.return_value = rows

    mock_db = AsyncMock()
    mock_db.execute.return_value = mock_result
    return mock_db

# make mock query for geofence
# return a mock response that mimics the subquery used in get_latest_vehicle_locations
# select() with fake vehicle data because SQLAlchemy needs it 
def make_geofence():
    fake_subquery = select(literal_column("'fake_id'").label("vehicle_id")).subquery()
    mock_query = MagicMock()
    mock_query.subquery.return_value = fake_subquery
    return mock_query

# make mock vehicle for testing
def make_vehicle():
    v = MagicMock()
    v.license_plate = "ABC123"
    v.vin = "VIN456"
    v.asset_type = "shuttle"
    v.gateway_model = "GatewayX"
    v.gateway_serial = "Serial123"
    return v

def make_vehicle2():
    v = MagicMock()
    v.license_plate = "XYZ321"
    v.vin = "VIN001"
    v.asset_type = "shuttle"
    v.gateway_model = "GatewayX"
    v.gateway_serial = "Serial456"
    return v


# make mock vehicle location for testing
def make_location():
    loc = MagicMock()
    loc.vehicle_id = "veh1"
    loc.name = "Shuttle 1"
    loc.latitude = 42.0
    loc.longitude = -71.0
    loc.timestamp = datetime(2026, 3, 31, 12, 0, 0)
    loc.heading_degrees = 90.0
    loc.speed_mph = 25.0
    loc.is_ecu_speed = True
    loc.formatted_location = "Some address"
    loc.address_id = "addr1"
    loc.address_name = "Main Stop"
    loc.vehicle = make_vehicle()
    return loc

# make mock location 2 for testing
def make_location2():
    loc = MagicMock()
    loc.vehicle_id = "veh2"
    loc.name = "Shuttle 2"
    loc.latitude = 90.0
    loc.longitude = -90.0
    loc.timestamp = datetime(2026, 3, 31, 12, 5, 0)
    loc.heading_degrees = 180.0
    loc.speed_mph = 40.0
    loc.is_ecu_speed = False
    loc.formatted_location = "Another address"
    loc.address_id = "addr2"
    loc.address_name = "Secondary Stop"
    loc.vehicle = make_vehicle2()
    return loc


#TESTS
# set up mock db that returns one location row
# get location from get_latest_vehicle_locations and check it matches the mock location data
# patch? patch replaces the get_vehicles_in_geofence_query with our mock query that returns fake subquery with vehicle_id 'fake_id'
@pytest.mark.asyncio
async def test_get_latest_vehicle_locations():
    db = make_db([make_location(), make_location2()])
    with patch("backend.fastapi.utils.get_vehicles_in_geofence_query", return_value=make_geofence()):
        result = await get_latest_vehicle_locations(make_session_factory(db))

    assert isinstance(result, list)
    assert len(result) == 2

    # loc holds info about the vehicle loc, but also vehicle data    
    loc: VehicleLocationDict = result[0]
    assert loc["vehicle_id"] == "veh1"
    assert loc["latitude"] == 42.0
    assert loc["timestamp"] == "2026-03-31T12:00:00"
    assert loc["vehicle"]["license_plate"] == "ABC123"


    loc2: VehicleLocationDict = result[1]
    assert loc2["vehicle_id"] == "veh2"
    assert loc2["latitude"] == 90.0
    assert loc2["timestamp"] == "2026-03-31T12:05:00"
    assert loc2["vehicle"]["license_plate"] == "XYZ321"


# test empty db returns empty list
@pytest.mark.asyncio
async def test_get_latest_vehicle_locations_empty():
    db = make_db([])
    with patch("backend.fastapi.utils.get_vehicles_in_geofence_query", return_value=make_geofence()):
        result = await get_latest_vehicle_locations(make_session_factory(db))

    assert result == []




