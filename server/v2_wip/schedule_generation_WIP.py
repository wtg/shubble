import json
from datetime import datetime, timedelta

def parse_interval(interval_str):
    if interval_str.endswith("m"):
        return timedelta(minutes=int(interval_str[:-1]))
    if interval_str.endswith("h"):
        return timedelta(hours=int(interval_str[:-1]))
    raise ValueError("Unsupported interval format")

def generate_schedule_for_day(day, route):
    today_dt = datetime.fromisoformat(day).date()
    day_type = "weekday" if today_dt.weekday() < 5 else "weekend"

    route_defs = route["routes"]
    results = {}

    for route_key, route_info in route_defs.items():
        schedule_list = {}
        stops = route_info["stops"]
        travel_times = route_info["travel_times"]
        sched_blocks = route_info["schedule"].get(day_type, [])

        if len(travel_times) != len(stops) - 1:
            raise ValueError(f"Travel times length mismatch for route {route_key}")
            #todo, alternatively, instead of raising an error, set travel_time to interval / len(stops
            # eventually, we can have data team update average travel times

        # pre-create lists for each stop
        for stop in stops:
            schedule_list[stop] = []

        for block in sched_blocks:
            # Parse "09:00" style times
            start_t = datetime.strptime(block["start"], "%H:%M").time()
            end_t = datetime.strptime(block["end"], "%H:%M").time()
            interval = parse_interval(block["interval"])
            excludes = set(block.get("excludes", []))

            # Merge with the chosen date
            start = datetime.combine(today_dt, start_t)
            end = datetime.combine(today_dt, end_t)

            # iterate departures (a departure represents the vehicle leaving the first stop)
            departure = start
            while departure <= end:
                # for each stop compute arrival = departure + cumulative travel
                cumulative = timedelta()
                # first stop arrival = departure (no travel)
                for idx, stop in enumerate(stops):
                    if idx > 0:
                        cumulative += timedelta(minutes=travel_times[idx - 1])
                    arrival = departure + cumulative
                    if arrival <= end and stop not in excludes:
                        schedule_list[stop].append(arrival.strftime("%H:%M"))
                departure += interval

        results[route_key] = {
            "title": route_info["title"],
            "schedule": schedule_list
        }

    return results


# import schedule_config from 'schedule_config.json' instead
with open("schedule_config.json", "r") as f:
    schedule_config = json.load(f)

final = generate_schedule_for_day("2025-10-20", schedule_config)

pretty = json.dumps(final, indent=2)

with open("schedule_output.json", "w") as f:
    json.dump(final, f, indent=2)

print(pretty)

