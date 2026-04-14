import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from sqlalchemy import literal_column, select
# from the db utils.py import 
from backend.fastapi.utils import get_latest_vehicle_locations, VehicleLocationDict
# Testing 2 utils first
from backend.fastapi.utils import (
    get_latest_vehicle_locations,
    VehicleLocationDict,
    VehicleInfoDict,
    get_current_driver_assignments, # Added for driver tests
    # make mock data in database
    # use get_locatest


    # assert
    # continue tests with empty, write read, write write read getting latest
    # no write read
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


# class VehicleInfoDict(TypedDict):
#     license_plate: Optional[str]
#     vin: Optional[str]
#     asset_type: str
#     gateway_model: Optional[str]
#     gateway_serial: Optional[str]

# Pass parameters to make_vehicle() to create different vehicles for testing or basic default
# DONE
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


# class VehicleLocationDict(TypedDict):
#     vehicle_id: str
#     name: Optional[str]
#     latitude: float
#     longitude: float
#     timestamp: str
#     heading_degrees: Optional[float]
#     speed_mph: Optional[float]
#     is_ecu_speed: bool
#     formatted_location: Optional[str]
#     address_id: Optional[str]
#     address_name: Optional[str]
#     vehicle: VehicleInfoDict

#DONE
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

# class DriverInfoDict(TypedDict):
#     id: str
#     name: str

#TEST WIP
def make_driver(
    id="driver1",
    name="Alice Smith",
):
    d = MagicMock()
    d.id = id
    d.name = name
    return d

# class DriverAssignmentDict(TypedDict):
#     vehicle_id: str
#     driver: Optional[DriverInfoDict]

#TEST WIP
def make_driver_assignment(
    vehicle_id="veh0", # default veh0
    driver=None,  # Pass a make_driver() result via the driver param
):
    da = MagicMock()
    da.vehicle_id = vehicle_id
    da.driver = driver if driver is not None else make_driver()
    return da


# class ETADict(TypedDict):
#     stop_times: dict
#     timestamp: str

#TEST WIP
def make_eta(
    stop_times={"stop1": "2026-04-01T10:00:00"},
    timestamp=datetime(2026, 4, 1, 9, 0, 0),
):
    eta = MagicMock()
    eta.stop_times = stop_times
    eta.timestamp = timestamp
    return eta


# class VelocityDict(TypedDict):
#     speed_kmh: float
#     timestamp: str

#TEST WIP
def make_velocity(
    speed_kmh=40.0,
    timestamp=datetime(2026, 4, 1, 9, 0, 0),
):
    v = MagicMock()
    v.speed_kmh = speed_kmh
    v.timestamp = timestamp
    return v


# TEST - Location tests
# fed into test_get_latest_vehicle_locations pytest
LOCATION_CASES = [
    pytest.param(
        # rows fed into the mock DB
        [
            make_location(
                vehicle_id="veh1",
                name="Shuttle 1",
                latitude=42.0,
                longitude=-71.0,
                timestamp=datetime(2026, 3, 31, 12, 0, 0),
                vehicle=make_vehicle(license_plate="ABC123", vin="VIN456"),
            ),
            make_location(
                vehicle_id="veh2",
                name="Shuttle 2",
                latitude=90.0,
                longitude=-90.0,
                timestamp=datetime(2026, 3, 31, 12, 5, 0),
                heading_degrees=180.0,
                speed_mph=40.0,
                is_ecu_speed=False,
                formatted_location="Another address",
                address_id="addr2",
                address_name="Secondary Stop",
                vehicle=make_vehicle(license_plate="XYZ321", vin="VIN001", gateway_serial="Serial456"),
            ),
        ],
        # expected dicts
        [
            {
                "vehicle_id": "veh1",
                "name": "Shuttle 1",
                "latitude": 42.0,
                "longitude": -71.0,
                "timestamp": "2026-03-31T12:00:00",
                "heading_degrees": 90.0,
                "speed_mph": 25.0,
                "is_ecu_speed": True,
                "formatted_location": "Some address",
                "address_id": "addr1",
                "address_name": "Main Stop",
                "vehicle": {
                    "license_plate": "ABC123",
                    "vin": "VIN456",
                    "asset_type": "shuttle",
                    "gateway_model": "GatewayX",
                    "gateway_serial": "Serial123",
                },
            },
            {
                "vehicle_id": "veh2",
                "name": "Shuttle 2",
                "latitude": 90.0,
                "longitude": -90.0,
                "timestamp": "2026-03-31T12:05:00",
                "heading_degrees": 180.0,
                "speed_mph": 40.0,
                "is_ecu_speed": False,
                "formatted_location": "Another address",
                "address_id": "addr2",
                "address_name": "Secondary Stop",
                "vehicle": {
                    "license_plate": "XYZ321",
                    "vin": "VIN001",
                    "asset_type": "shuttle",
                    "gateway_model": "GatewayX",
                    "gateway_serial": "Serial456",
                },
            },
        ],
        id="two_shuttles",
    ),
    pytest.param(
        [
            make_location(
                vehicle_id="veh3",
                name="Shuttle 3",
                latitude=0.0,
                longitude=0.0,
                timestamp=datetime(2026, 1, 1, 8, 30, 0),
                heading_degrees=None,
                speed_mph=0.0,
                is_ecu_speed=False,
                formatted_location=None,
                address_id=None,
                address_name=None,
                vehicle=make_vehicle(license_plate=None, vin=None, gateway_model=None, gateway_serial=None),
            ),
        ],
        [
            {
                "vehicle_id": "veh3",
                "name": "Shuttle 3",
                "latitude": 0.0,
                "longitude": 0.0,
                "timestamp": "2026-01-01T08:30:00",
                "heading_degrees": None,
                "speed_mph": 0.0,
                "is_ecu_speed": False,
                "formatted_location": None,
                "address_id": None,
                "address_name": None,
                "vehicle": {
                    "license_plate": None,
                    "vin": None,
                    "asset_type": "shuttle",
                    "gateway_model": None,
                    "gateway_serial": None,
                },
            },
        ],
        id="single_shuttle_nulls",
    ),
    pytest.param(
        [
            make_location(
                vehicle_id="veh_east",
                name="Shuttle East",
                latitude=43.1,
                longitude=-72.5,
                timestamp=datetime(2026, 4, 1, 9, 0, 0),
                vehicle=make_vehicle(license_plate="EAST01", vin="VIN_EAST"),
            ),
            make_location(
                vehicle_id="veh_west",
                name="Shuttle West",
                latitude=41.2,
                longitude=-73.1,
                timestamp=datetime(2026, 4, 1, 9, 10, 0),
                vehicle=make_vehicle(license_plate="WEST01", vin="VIN_WEST"),
            ),
            make_location(
                vehicle_id="veh_north",
                name="Shuttle North",
                latitude=44.8,
                longitude=-70.3,
                timestamp=datetime(2026, 4, 1, 9, 20, 0),
                vehicle=make_vehicle(license_plate="NORTH01", vin="VIN_NORTH"),
            ),
        ],
        [
            {
                "vehicle_id": "veh_east",
                "name": "Shuttle East",
                "latitude": 43.1,
                "longitude": -72.5,
                "timestamp": "2026-04-01T09:00:00",
                "heading_degrees": 90.0,
                "speed_mph": 25.0,
                "is_ecu_speed": True,
                "formatted_location": "Some address",
                "address_id": "addr1",
                "address_name": "Main Stop",
                "vehicle": {
                    "license_plate": "EAST01",
                    "vin": "VIN_EAST",
                    "asset_type": "shuttle",
                    "gateway_model": "GatewayX",
                    "gateway_serial": "Serial123",
                },
            },
            {
                "vehicle_id": "veh_west",
                "name": "Shuttle West",
                "latitude": 41.2,
                "longitude": -73.1,
                "timestamp": "2026-04-01T09:10:00",
                "heading_degrees": 90.0,
                "speed_mph": 25.0,
                "is_ecu_speed": True,
                "formatted_location": "Some address",
                "address_id": "addr1",
                "address_name": "Main Stop",
                "vehicle": {
                    "license_plate": "WEST01",
                    "vin": "VIN_WEST",
                    "asset_type": "shuttle",
                    "gateway_model": "GatewayX",
                    "gateway_serial": "Serial123",
                },
            },
            {
                "vehicle_id": "veh_north",
                "name": "Shuttle North",
                "latitude": 44.8,
                "longitude": -70.3,
                "timestamp": "2026-04-01T09:20:00",
                "heading_degrees": 90.0,
                "speed_mph": 25.0,
                "is_ecu_speed": True,
                "formatted_location": "Some address",
                "address_id": "addr1",
                "address_name": "Main Stop",
                "vehicle": {
                    "license_plate": "NORTH01",
                    "vin": "VIN_NORTH",
                    "asset_type": "shuttle",
                    "gateway_model": "GatewayX",
                    "gateway_serial": "Serial123",
                },
            },
        ],
        id="three_shuttles_last_is_latest",
    ),
]


#WIP

# class DriverInfoDict(TypedDict):
#     id: str
#     name: str

#TEST WIP
def make_driver(
    id="driver1",
    name="Alice Smith",
):
    d = MagicMock()
    d.id = id
    d.name = name
    return d

# UPDATED Driver_Cases
Driver_Cases = [
    pytest.param(
        ["shuttle_01"], # vehicle_ids to query
        lambda: (
            # Create the vehicle
            (shuttle_1 := make_vehicle(license_plate="TEST-01", vin="VIN-01")),
            # Create the driver
            (driver_1 := make_driver(id="d1", name="John Doe")),
            # Create the assignment linking the vehicle_id to the driver object
            [make_driver_assignment(vehicle_id="shuttle_01", driver=driver_1)]
        )[-1], # Return only the list of rows for the mock DB
        {
            "shuttle_01": {
                "vehicle_id": "shuttle_01",
                "driver": {"id": "d1", "name": "John Doe"}
            }
        },
        id="1driver_assignment",
    ),

    pytest.param(
        ["shuttle_east", "shuttle_west"],
        lambda: (
            (east_veh := make_vehicle(license_plate="EAST-1")),
            (west_veh := make_vehicle(license_plate="WEST-1")),
            (jane := make_driver(id="d2", name="Jane Doe")),
            (bob := make_driver(id="d3", name="Bob Driver")),
            [
                make_driver_assignment(vehicle_id="shuttle_east", driver=jane),
                make_driver_assignment(vehicle_id="shuttle_west", driver=bob)
            ]
        )[-1],
        {
            "shuttle_east": {"vehicle_id": "shuttle_east", "driver": {"id": "d2", "name": "Jane Doe"}},
            "shuttle_west": {"vehicle_id": "shuttle_west", "driver": {"id": "d3", "name": "Bob Driver"}}
        },
        id="multi_driver_assignment",
    ),

    pytest.param(
        ["bot_veh"],
        lambda: (
            (bot_shuttle := make_vehicle(license_plate="BOT-999")),
            (bot_user := make_driver(id="bot_01", name="Shubble Bot")),
            [make_driver_assignment(vehicle_id="bot_veh", driver=bot_user)]
        )[-1],
        {
            "bot_veh": {
                "vehicle_id": "bot_veh",
                "driver": {"id": "bot_01", "name": "Shubble Bot"}
            }
        },
        id="driver_test_bot",
    )
]

@pytest.mark.asyncio
@pytest.mark.parametrize("v_ids, rows_factory, expected", Driver_Cases)
async def test_confirm_driver_info_dict(v_ids, rows_factory, expected):
    # Execute the lambda to "create" vehicles and drivers, then get the rows
    rows = rows_factory()
    
    db = make_db(rows)
    # Testing get_current_driver_assignments specifically
    result = await get_current_driver_assignments(v_ids, make_session_factory(db))
    
    assert result == expected

#Tests to do 
# async def smart_closest_point(
#     vehicle_ids: List[str]
# ) -> Dict[str, Tuple[Optional[float], Optional[Tuple[float, float]], Optional[str], Optional[int], Optional[int], Optional[str]]]:

# async def get_current_driver_assignments(
#     vehicle_ids: List[str], session_factory
# ) -> Dict[str, DriverAssignmentDict]:

# async def get_latest_etas(vehicle_ids: List[str], session_factory) -> Dict[str, ETADict]:

#async def get_latest_velocities(vehicle_ids: List[str], session_factory) -> Dict[str, VelocityDict]:

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


    #Work on velocities and etas next