"""
seed_db.py

Seeds the database using routes.json + schedule.json.

Run it with:
    uv run python backend/seed_db.py
"""

import asyncio
import json
from datetime import date, datetime, time, timezone, timedelta
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import create_async_db_engine, create_session_factory
from backend.models import (
    BusSchedule,
    BusScheduleToDaySchedule,
    DateToDaySchedule,
    DaySchedule,
    Polyline,
    Route,
    RouteToBusSchedule,
    Stop,
)

# file paths
HERE = Path(__file__).resolve().parent
ROUTES_JSON = HERE / "../shared/routes.json"
SCHEDULE_JSON = HERE / "../shared/schedule.json"

# day index -> schedule key
DOW_TO_KEY = {
    0: "MONDAY",
    1: "TUESDAY",
    2: "WEDNESDAY",
    3: "THURSDAY",
    4: "FRIDAY",
    5: "SATURDAY",
    6: "SUNDAY",
}


def parse_time(time_str: str) -> time:
    return datetime.strptime(time_str.strip(), "%I:%M %p").time()


def to_utc(t: time) -> time:
    return t.replace(tzinfo=timezone.utc)


async def seed(session: AsyncSession) -> None:
    # wipe everything clean first
    await session.execute(text("""
        TRUNCATE TABLE
            route_to_bus_schedules,
            bus_schedules_to_day_schedules,
            date_to_day_schedules,
            polylines,
            stops,
            routes,
            bus_schedules,
            day_schedules
        RESTART IDENTITY CASCADE;
    """))

    # load JSON
    with open(ROUTES_JSON, encoding="utf-8") as f:
        routes_data = json.load(f)

    with open(SCHEDULE_JSON, encoding="utf-8") as f:
        schedule_data = json.load(f)

   
    # Routes + Stops
    route_objs: dict[str, Route] = {}

    for route_key, route_info in routes_data.items():
        route = Route(
            name=route_key,
            route_color=route_info["COLOR"],
        )
        session.add(route)
        await session.flush()

        route_objs[route_key] = route

        stop_keys = route_info["STOPS"]
        polyline_segments = route_info["ROUTES"]

        stop_map: dict[str, Stop] = {}

        # create stops
        for stop_key in stop_keys:
            stop_def = route_info[stop_key]
            lat, lng = stop_def["COORDINATES"]

            stop = Stop(
                name=stop_def["NAME"],
                latitude=lat,
                longitude=lng,
                route_id=route.id,
            )
            session.add(stop)
            await session.flush()

            stop_map[stop_key] = stop

        # Polylines
        for i, coords in enumerate(polyline_segments):
            dep = stop_map[stop_keys[i]]
            arr = stop_map[stop_keys[i + 1]]

            session.add(Polyline(
                departure_stop_id=dep.id,
                arrival_stop_id=arr.id,
                coordinates=[f"{lat},{lng}" for lat, lng in coords],
            ))

    await session.flush()

    # DaySchedules
    day_schedule_names = set()
    day_map = {}

    for key, value in schedule_data.items():
        if isinstance(value, str):
            day_map[key] = value
            day_schedule_names.add(value)

    day_schedule_objs: dict[str, DaySchedule] = {}

    for name in day_schedule_names:
        ds = DaySchedule(name=name)
        session.add(ds)
        await session.flush()
        day_schedule_objs[name] = ds

    # Map every date in 2026 -> DaySchedule
    current = date(2026, 1, 1)
    end = date(2026, 12, 31)

    while current <= end:
        dow_key = DOW_TO_KEY[current.weekday()]
        ds_name = day_map.get(dow_key)

        if ds_name and ds_name in day_schedule_objs:
            session.add(DateToDaySchedule(
                day_schedule_id=day_schedule_objs[ds_name].id,
                date=current,
            ))

        current += timedelta(days=1)

    await session.flush()

 
    # BusSchedules + mappings
    for ds_name, ds_obj in day_schedule_objs.items():
        bus_schedules = schedule_data[ds_name]

        for bus_name, departures in bus_schedules.items():
            bus = BusSchedule(name=bus_name)
            session.add(bus)
            await session.flush()

            # link bus -> day schedule
            session.add(BusScheduleToDaySchedule(
                bus_schedule_id=bus.id,
                day_schedule_id=ds_obj.id,
            ))

            # add route + time entries
            for time_str, route_key in departures:
                route = route_objs.get(route_key)

                if not route:
                    print(f"WARNING: route '{route_key}' missing in routes.json")
                    continue

                session.add(RouteToBusSchedule(
                    route_id=route.id,
                    bus_schedule_id=bus.id,
                    time=to_utc(parse_time(time_str)),
                ))

    await session.flush()


async def main():
    engine = create_async_db_engine(settings.DATABASE_URL, echo=False)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        try:
            await seed(session)
            await session.commit()
            print("Seeding complete.")
        except Exception as e:
            await session.rollback()
            print(f"Seeding failed: {e}")
            raise
        finally:
            await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())