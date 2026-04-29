import json
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock
from types import SimpleNamespace

BASE = Path(__file__).parent


def load_json(name):
    # Prefer testdata/ subfolder, fall back to module folder
    candidate = BASE / "testdata" / name
    if candidate.exists():
        return json.loads(candidate.read_text())
    return json.loads((BASE / name).read_text())


def _to_dt(s):
    return datetime.fromisoformat(s) if s is not None else None


def build_vehicle_map():
    data = load_json("vehicledata.json")
    out = {}
    for vid, v in data.items():
        m = SimpleNamespace(
            license_plate=v.get("license_plate"),
            vin=v.get("vin"),
            asset_type=v.get("asset_type"),
            gateway_model=v.get("gateway_model"),
            gateway_serial=v.get("gateway_serial"),
        )
        out[vid] = m
    return out


def build_driver_map():
    data = load_json("driverdata.json")
    out = {}
    for key, d in data.items():
        m = SimpleNamespace(id=d.get("id"), name=d.get("name"))
        out[key] = m
    return out


def build_location_rows_and_expected():
    vehicles = build_vehicle_map()
    data = load_json("locationdata.json")
    if isinstance(data, dict):
        data = list(data.values())
    rows = []
    expected = []
    for item in data:
        loc = MagicMock()
        loc.vehicle_id = item.get("vehicle_id")
        loc.name = item.get("name")
        loc.latitude = item.get("latitude")
        loc.longitude = item.get("longitude")
        loc.timestamp = _to_dt(item.get("timestamp"))
        loc.heading_degrees = item.get("heading_degrees")
        loc.speed_mph = item.get("speed_mph")
        loc.is_ecu_speed = item.get("is_ecu_speed")
        loc.formatted_location = item.get("formatted_location")
        loc.address_id = item.get("address_id")
        loc.address_name = item.get("address_name")
        # vehicle reference inside location data
        vehicle_ref = item.get("vehicle_ref") or item.get("vehicle_id")
        loc.vehicle = vehicles.get(vehicle_ref)
        rows.append(loc)

        vehicle_obj = loc.vehicle
        vehicle_dict = None
        if vehicle_obj is not None:
            vehicle_dict = {
                "license_plate": getattr(vehicle_obj, "license_plate", None),
                "vin": getattr(vehicle_obj, "vin", None),
                "asset_type": getattr(vehicle_obj, "asset_type", "shuttle"),
                "gateway_model": getattr(vehicle_obj, "gateway_model", None),
                "gateway_serial": getattr(vehicle_obj, "gateway_serial", None),
            }

        expected.append(
            {
                "vehicle_id": loc.vehicle_id,
                "name": loc.name,
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "timestamp": loc.timestamp.isoformat() if loc.timestamp is not None else None,
                "heading_degrees": loc.heading_degrees,
                "speed_mph": loc.speed_mph,
                "is_ecu_speed": loc.is_ecu_speed,
                "formatted_location": loc.formatted_location,
                "address_id": loc.address_id,
                "address_name": loc.address_name,
                "vehicle": vehicle_dict,
            }
        )

    return rows, expected


def build_driver_assignment_rows_and_expected():
    drivers = build_driver_map()
    data = load_json("driver_assignmentdata.json")
    if isinstance(data, dict):
        data = list(data.values())
    rows = []
    expected = {}
    for item in data:
        vehicle_id = item.get("vehicle_id")
        driver_ref = item.get("driver_ref")
        driver_obj = drivers.get(driver_ref)

        da = SimpleNamespace(vehicle_id=vehicle_id, driver=driver_obj)
        rows.append(da)

        expected[vehicle_id] = {
            "vehicle_id": vehicle_id,
            "driver": {"id": getattr(driver_obj, "id", None), "name": getattr(driver_obj, "name", None)}
            if driver_obj is not None
            else None,
        }

    return rows, expected


def build_eta_rows_and_expected():
    data = load_json("etadata.json")
    if isinstance(data, dict):
        data = list(data.values())
    rows = []
    expected = {}
    for item in data:
        eta = SimpleNamespace(vehicle_id=item.get("vehicle_id"), etas=item.get("etas"), timestamp=_to_dt(item.get("timestamp")))
        rows.append(eta)
        expected[eta.vehicle_id] = {"stop_times": eta.etas, "timestamp": eta.timestamp.isoformat()}
    return rows, expected


def build_velocity_rows_and_expected():
    data = load_json("velocitydata.json")
    if isinstance(data, dict):
        data = list(data.values())
    rows = []
    expected = {}
    for item in data:
        v = SimpleNamespace(vehicle_id=item.get("vehicle_id"), speed_kmh=item.get("speed_kmh"), timestamp=_to_dt(item.get("timestamp")))
        rows.append(v)
        expected[v.vehicle_id] = {"speed_kmh": v.speed_kmh, "timestamp": v.timestamp.isoformat()}
    return rows, expected


def load_expected(name):
    p = BASE / "expected" / name
    return json.loads(p.read_text())


def load_expected_vehicle_locations():
    return load_expected("expected_vehicle_locations.json")


def load_expected_driver_assignments():
    return load_expected("expected_driver_assignments.json")


def load_expected_etas():
    return load_expected("expected_etas.json")


def load_expected_velocities():
    return load_expected("expected_velocities.json")


def load_expected_drivers():
    return load_expected("expected_drivers.json")


def load_expected_vehicles():
    return load_expected("expected_vehicles.json")
