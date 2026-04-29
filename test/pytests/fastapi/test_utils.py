import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from sqlalchemy import literal_column, select
from backend.fastapi.utils import get_latest_vehicle_locations, VehicleLocationDict
from backend.fastapi.utils import (
    get_latest_vehicle_locations,
    VehicleLocationDict,
    VehicleInfoDict,
    get_current_driver_assignments,
    get_latest_etas,
    get_latest_velocities
)

# Fixture loader to build rows/expected from JSON files
from test.pytests.fastapi.fixture_loader import (
    build_location_rows_and_expected,
    build_driver_assignment_rows_and_expected,
    build_eta_rows_and_expected,
    build_velocity_rows_and_expected,
    load_expected_drivers,
    load_expected_vehicles,
)

# Mimics the async context manager of SQLAlchemy session factory used in utils functions, but returns a mock DB instead of a real session.
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


# Pass parameters to make_vehicle() to create different vehicles for testing or basic default
def make_vehicle(
    license_plate="ABC123",
    vin="VIN456",
    asset_type="shuttle",
    gateway_model="GatewayX",
    gateway_serial="Serial123",
):
    v = MagicMock()
    v.license_plate = license_plate
    v.vin = vin
    v.asset_type = asset_type
    v.gateway_model = gateway_model
    v.gateway_serial = gateway_serial
    return v



def make_location(
    vehicle_id="veh1",
    name="Shuttle 1",
    latitude=42.0,
    longitude=-71.0,
    timestamp=datetime(2026, 3, 31, 12, 0, 0),
    heading_degrees=90.0,
    speed_mph=25.0,
    is_ecu_speed=True,
    formatted_location="Some address",
    address_id="addr1",
    address_name="Main Stop",
    vehicle=None,
):
    # Pass a make_vehicle() result via the vehicle param
    loc = MagicMock()
    loc.vehicle_id = vehicle_id
    loc.name = name
    loc.latitude = latitude
    loc.longitude = longitude
    loc.timestamp = timestamp
    loc.heading_degrees = heading_degrees
    loc.speed_mph = speed_mph
    loc.is_ecu_speed = is_ecu_speed
    loc.formatted_location = formatted_location
    loc.address_id = address_id
    loc.address_name = address_name
    loc.vehicle = vehicle if vehicle is not None else make_vehicle()
    return loc

# TEST - Location tests
# fed into test_get_latest_vehicle_locations pytest
LOCATION_CASES = []
rows, expected = build_location_rows_and_expected()
LOCATION_CASES.append(pytest.param(rows, expected, id="from_fixtures"))


# Driver cases built from JSON fixtures
Driver_Cases = []
rows_all, expected_map = build_driver_assignment_rows_and_expected()
for vid, info in expected_map.items():
    def make_factory(rows_snapshot=rows_all, v=vid):
        return lambda: [r for r in rows_snapshot if r.vehicle_id == v]

    Driver_Cases.append(pytest.param([vid], make_factory(), {vid: info}, id=f"driver_{vid}"))

# Above are legacy test cases, new system uses testdata folder which holds json files for each utility tested
# insert test cases in xyz.json, view fixture_loader.py for how they are loaded and built into test cases, and then add test function below that uses the cases
# results are stored in the expected folder as xyz.json and loaded in fixture_loader.py for assertion in test function below

@pytest.mark.asyncio
@pytest.mark.parametrize("v_ids, rows_factory, expected", Driver_Cases)
async def test_confirm_driver_info_dict(v_ids, rows_factory, expected):
    # Execute the lambda to "create" vehicles and drivers, then get the rows
    rows = rows_factory()
    
    db = make_db(rows)
    # Testing get_current_driver_assignments specifically
    result = await get_current_driver_assignments(v_ids, make_session_factory(db))
    
    assert result == expected

# Tests 
@pytest.mark.asyncio
@pytest.mark.parametrize("rows, expected", LOCATION_CASES)
async def test_get_latest_vehicle_locations(rows, expected):
    # Parametrized: verifies get_latest_vehicle_locations returns the exact
    # data that was inserted into the mock DB, for each set of test cases.
    db = make_db(rows)
    with patch("backend.fastapi.utils.get_vehicles_in_geofence_query", return_value=make_geofence()):
        result = await get_latest_vehicle_locations(make_session_factory(db))

    # Check for exact amount of shuttles
    assert len(result) == len(expected)

    # Check each returned shuttle dict matches the expected dict for that index.
    for actual_loc, expected_loc in zip(result, expected):
        assert actual_loc == expected_loc


@pytest.mark.asyncio
async def test_get_latest_vehicle_locations_empty():
    # Empty DB should return an empty list.
    db = make_db([])
    with patch("backend.fastapi.utils.get_vehicles_in_geofence_query", return_value=make_geofence()):
        result = await get_latest_vehicle_locations(make_session_factory(db))

    assert result == []


@pytest.mark.asyncio
async def test_three_shuttles_last_inserted_is_distinct():
    # Insert three shuttles; assert the last one (Shuttle North) is present
    # and correct, and that the first two do not equal it.
    shuttle_east = make_location(
        vehicle_id="veh_east",
        name="Shuttle East",
        latitude=43.1,
        longitude=-72.5,
        timestamp=datetime(2026, 4, 1, 9, 0, 0),
        vehicle=make_vehicle(license_plate="EAST01", vin="VIN_EAST"),
    )
    shuttle_west = make_location(
        vehicle_id="veh_west",
        name="Shuttle West",
        latitude=41.2,
        longitude=-73.1,
        timestamp=datetime(2026, 4, 1, 9, 10, 0),
        vehicle=make_vehicle(license_plate="WEST01", vin="VIN_WEST"),
    )
    shuttle_north = make_location(
        vehicle_id="veh_north",
        name="Shuttle North",
        latitude=44.8,
        longitude=-70.3,
        timestamp=datetime(2026, 4, 1, 9, 20, 0),
        vehicle=make_vehicle(license_plate="NORTH01", vin="VIN_NORTH"),
    )

    db = make_db([shuttle_east, shuttle_west, shuttle_north])
    with patch("backend.fastapi.utils.get_vehicles_in_geofence_query", return_value=make_geofence()):
        result = await get_latest_vehicle_locations(make_session_factory(db))

    assert len(result) == 3

    first = result[0]   # shuttle_east check
    second = result[1]  # shuttle_west check
    last = result[2]    # shuttle_north check

    # Last inserted shuttle is present with correct data
    assert last["vehicle_id"] == "veh_north"
    assert last["vehicle"]["license_plate"] == "NORTH01"
    assert last["timestamp"] == "2026-04-01T09:20:00"

    # Unique shuttles
    assert first != last
    assert first != second
    assert second != last



ETA_CASES = []
rows_all, expected_map = build_eta_rows_and_expected()
for vid, info in expected_map.items():
    rows_for_vid = [r for r in rows_all if r.vehicle_id == vid]
    ETA_CASES.append(pytest.param([vid], rows_for_vid, {vid: info}, id=f"eta_{vid}"))
if len(expected_map) >= 2:
    vids = list(expected_map.keys())[:2]
    rows_multi = [r for r in rows_all if r.vehicle_id in vids]
    ETA_CASES.append(pytest.param(vids, rows_multi, {v: expected_map[v] for v in vids}, id="eta_multi"))

@pytest.mark.asyncio
@pytest.mark.parametrize("vehicle_ids, rows, expected", ETA_CASES)
async def test_get_latest_etas(vehicle_ids, rows, expected):
    db = make_db(rows)

    result = await get_latest_etas(vehicle_ids, make_session_factory(db))

    assert result == expected

@pytest.mark.asyncio
async def test_get_latest_etas_empty_vehicle_ids():
    db = make_db([])

    result = await get_latest_etas([], make_session_factory(db))

    assert result == {}



VELOCITY_CASES = []
rows_all, expected_map = build_velocity_rows_and_expected()
for vid, info in expected_map.items():
    rows_for_vid = [r for r in rows_all if r.vehicle_id == vid]
    VELOCITY_CASES.append(pytest.param([vid], rows_for_vid, {vid: info}, id=f"vel_{vid}"))
if len(expected_map) >= 2:
    vids = list(expected_map.keys())[:2]
    rows_multi = [r for r in rows_all if r.vehicle_id in vids]
    VELOCITY_CASES.append(pytest.param(vids, rows_multi, {v: expected_map[v] for v in vids}, id="vel_multi"))


@pytest.mark.asyncio
@pytest.mark.parametrize("vehicle_ids, rows, expected", VELOCITY_CASES)
async def test_get_latest_velocities(vehicle_ids, rows, expected):
    db = make_db(rows)

    result = await get_latest_velocities(vehicle_ids, make_session_factory(db))

    assert result == expected

@pytest.mark.asyncio
async def test_get_latest_velocities_empty_vehicle_ids():
    db = make_db([])

    result = await get_latest_velocities([], make_session_factory(db))

    assert result == {}

