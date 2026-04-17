"""
Shuttle schedule simulation engine with Gaussian noise.

Simulates shuttle arrivals/departures with:
- Gaussian noise: mu=0, sigma=90 seconds, clamped to +/-3 minutes
- Union stop constraint: STUDENT_UNION and STUDENT_UNION_RETURN may
  only depart on time or late (never early)
- Multiple routes with 8-12 stops each
- Generates 50+ trips across all routes over a 2-hour window
"""

import hashlib
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Load route data
SHARED_DIR = Path(__file__).parent.parent.parent / "shared"

with open(SHARED_DIR / "routes.json") as f:
    ROUTES_DATA = json.load(f)

with open(SHARED_DIR / "aggregated_schedule.json") as f:
    AGGREGATED_SCHEDULE = json.load(f)

# Constants
NOISE_MU = 0.0         # Mean delay (seconds)
NOISE_SIGMA = 90.0     # Std dev (seconds)
NOISE_CLAMP = 180.0    # Max |delay| = 3 minutes

# Union stop keys — these stops may ONLY depart on time or late
UNION_STOPS = {"STUDENT_UNION", "STUDENT_UNION_RETURN"}

# Active routes with enough stops
ACTIVE_ROUTES = ["NORTH", "WEST"]


@dataclass
class StopArrival:
    """A single stop arrival/departure within a trip."""
    trip_id: str
    route: str
    stop_key: str
    stop_name: str
    stop_index: int
    scheduled_time: datetime
    actual_time: datetime
    delay_seconds: float
    is_union_stop: bool

    @property
    def delay_minutes(self) -> float:
        return self.delay_seconds / 60.0

    @property
    def is_early(self) -> bool:
        return self.delay_seconds < 0

    @property
    def is_late(self) -> bool:
        return self.delay_seconds > 0

    @property
    def is_on_time(self) -> bool:
        return self.delay_seconds == 0


@dataclass
class SimulatedTrip:
    """A complete trip: one shuttle loop on one route."""
    trip_id: str
    route: str
    shuttle_id: str
    departure_time: datetime
    stops: list[StopArrival] = field(default_factory=list)

    @property
    def num_stops(self) -> int:
        return len(self.stops)


@dataclass
class SimulationResult:
    """Complete simulation output."""
    trips: list[SimulatedTrip]
    routes_used: list[str]
    window_start: datetime
    window_end: datetime
    total_stop_arrivals: int
    rng_seed: int

    @property
    def num_trips(self) -> int:
        return len(self.trips)

    def all_arrivals(self) -> list[StopArrival]:
        """Flat list of all stop arrivals across all trips."""
        return [stop for trip in self.trips for stop in trip.stops]

    def arrivals_at_stop(self, stop_key: str) -> list[StopArrival]:
        """All arrivals at a specific stop."""
        return [s for s in self.all_arrivals() if s.stop_key == stop_key]

    def union_arrivals(self) -> list[StopArrival]:
        """All arrivals at Union stops."""
        return [s for s in self.all_arrivals() if s.is_union_stop]

    def delays(self) -> list[float]:
        """All delay values in seconds."""
        return [s.delay_seconds for s in self.all_arrivals()]


def generate_delay(rng: np.random.Generator, is_union: bool = False) -> float:
    """
    Generate a Gaussian delay in seconds.

    Args:
        rng: NumPy random generator
        is_union: If True, clamp negative delays to 0 (Union stop constraint)

    Returns:
        Delay in seconds (positive = late, negative = early, 0 = on time)
    """
    delay = rng.normal(NOISE_MU, NOISE_SIGMA)

    # Clamp to +/- 3 minutes
    delay = max(-NOISE_CLAMP, min(NOISE_CLAMP, delay))

    # Union stop constraint: never early
    if is_union and delay < 0:
        delay = 0.0

    return delay


def get_route_stops(route_name: str) -> list[dict]:
    """
    Get stops for a route with their offsets.

    Returns list of dicts: [{"key": "STUDENT_UNION", "name": "Student Union", "offset_min": 0}, ...]
    """
    route = ROUTES_DATA.get(route_name)
    if not route or "STOPS" not in route:
        return []

    stops = []
    for stop_key in route["STOPS"]:
        stop_data = route.get(stop_key, {})
        stops.append({
            "key": stop_key,
            "name": stop_data.get("NAME", stop_key),
            "offset_min": stop_data.get("OFFSET", 0),
        })
    return stops


def parse_schedule_time(time_str: str, base_date: datetime) -> datetime:
    """Parse '9:00 AM' into a datetime on base_date."""
    parsed = datetime.strptime(time_str.strip(), "%I:%M %p")
    result = base_date.replace(
        hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0
    )
    # "12:00 AM" means midnight of next day
    if time_str.strip() == "12:00 AM":
        result += timedelta(days=1)
    return result


def run_simulation(
    seed: int = 42,
    window_hours: float = 2.0,
    min_trips: int = 50,
    day_index: int = 1,  # Monday by default
    base_time: Optional[datetime] = None,
) -> SimulationResult:
    """
    Run a full shuttle schedule simulation.

    Args:
        seed: Random seed for reproducibility
        window_hours: Simulation window in hours
        min_trips: Minimum number of trips to generate
        day_index: Day of week (0=Sun, 1=Mon, ..., 6=Sat)
        base_time: Base datetime for the simulation (default: today at 9:00 AM ET)

    Returns:
        SimulationResult with all trips and stop arrivals
    """
    rng = np.random.default_rng(seed)

    if base_time is None:
        # Use a future base time so generate_api_payload produces future ETAs
        base_time = datetime.now(timezone.utc) + timedelta(minutes=5)
        base_time = base_time.replace(second=0, microsecond=0)

    window_start = base_time
    window_end = base_time + timedelta(hours=window_hours)

    # Get schedule for the day
    day_schedule = AGGREGATED_SCHEDULE[day_index]

    trips: list[SimulatedTrip] = []
    shuttle_counter = 1
    trip_counter = 0

    for route_name in ACTIVE_ROUTES:
        if route_name not in day_schedule:
            continue

        time_strings = day_schedule[route_name]
        stops = get_route_stops(route_name)

        if not stops:
            continue

        # Parse departure times within window
        departure_times = []
        for ts in time_strings:
            dt = parse_schedule_time(ts, base_time)
            if window_start <= dt <= window_end:
                departure_times.append(dt)

        # Ensure enough trips per route by extending with 10-min spacing
        target_per_route = max(min_trips // len(ACTIVE_ROUTES) + 1, 26)
        while len(departure_times) < target_per_route:
            if departure_times:
                last = departure_times[-1]
                next_dep = last + timedelta(minutes=10)
                departure_times.append(next_dep)
            else:
                departure_times.append(window_start)

        # Create trips
        for dep_time in departure_times:
            shuttle_id = f"SIM-{shuttle_counter:03d}"
            # Deterministic trip ID from seed + counter
            trip_id = hashlib.md5(f"{seed}-{trip_counter}".encode()).hexdigest()[:8]
            shuttle_counter += 1
            trip_counter += 1

            trip = SimulatedTrip(
                trip_id=trip_id,
                route=route_name,
                shuttle_id=shuttle_id,
                departure_time=dep_time,
            )

            for idx, stop in enumerate(stops):
                scheduled = dep_time + timedelta(minutes=stop["offset_min"])
                is_union = stop["key"] in UNION_STOPS
                delay = generate_delay(rng, is_union=is_union)
                actual = scheduled + timedelta(seconds=delay)

                arrival = StopArrival(
                    trip_id=trip_id,
                    route=route_name,
                    stop_key=stop["key"],
                    stop_name=stop["name"],
                    stop_index=idx,
                    scheduled_time=scheduled,
                    actual_time=actual,
                    delay_seconds=delay,
                    is_union_stop=is_union,
                )
                trip.stops.append(arrival)

            trips.append(trip)

    total_arrivals = sum(t.num_stops for t in trips)

    return SimulationResult(
        trips=trips,
        routes_used=ACTIVE_ROUTES,
        window_start=window_start,
        window_end=window_end,
        total_stop_arrivals=total_arrivals,
        rng_seed=seed,
    )


def generate_api_payload(sim: SimulationResult) -> dict:
    """
    Convert simulation results into the format expected by the /api/etas endpoint.

    Returns dict matching StopETAMap format: {stop_key: {eta, vehicle_id, route, last_arrival}}
    """
    now = datetime.now(timezone.utc)
    result = {}

    for trip in sim.trips:
        for stop in trip.stops:
            if stop.actual_time <= now:
                # Past arrival — record as last_arrival
                existing = result.get(stop.stop_key)
                if existing is None or (existing.get("last_arrival") is None):
                    if stop.stop_key not in result:
                        result[stop.stop_key] = {
                            "eta": None,
                            "vehicle_id": None,
                            "route": stop.route,
                            "last_arrival": stop.actual_time.isoformat(),
                        }
                    else:
                        result[stop.stop_key]["last_arrival"] = stop.actual_time.isoformat()
            else:
                # Future arrival — pick earliest
                existing = result.get(stop.stop_key)
                if existing is None or existing.get("eta") is None or stop.actual_time.isoformat() < existing["eta"]:
                    result[stop.stop_key] = {
                        "eta": stop.actual_time.isoformat(),
                        "vehicle_id": trip.shuttle_id,
                        "route": stop.route,
                        "last_arrival": result.get(stop.stop_key, {}).get("last_arrival"),
                    }

    return result


def generate_locations_payload(sim: SimulationResult) -> dict:
    """
    Convert simulation into /api/locations format for active shuttles.
    """
    now = datetime.now(timezone.utc)
    result = {}

    for trip in sim.trips:
        # Find the "current" stop (most recent past arrival)
        current_stop = None
        for stop in trip.stops:
            if stop.actual_time <= now:
                current_stop = stop
            else:
                break

        if current_stop is None:
            continue

        # Get coordinates from route data
        route = ROUTES_DATA.get(trip.route, {})
        stop_data = route.get(current_stop.stop_key, {})
        coords = stop_data.get("COORDINATES", [42.73, -73.68])

        result[trip.shuttle_id] = {
            "name": trip.shuttle_id,
            "latitude": coords[0],
            "longitude": coords[1],
            "timestamp": current_stop.actual_time.isoformat(),
            "heading_degrees": 0,
            "speed_mph": 20,
            "is_ecu_speed": False,
            "formatted_location": f"Near {current_stop.stop_name}",
            "address_id": "sim",
            "address_name": "Campus",
            "license_plate": f"SIM-{trip.shuttle_id[-3:]}",
            "vin": f"SIM{trip.shuttle_id[-3:]}",
            "asset_type": "vehicle",
            "gateway_model": "SIM",
            "gateway_serial": "SIM",
            "driver": None,
        }

    return result
