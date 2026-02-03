import numpy as np
from datetime import datetime, timezone

"""
Schedule - stores a list of routes with times
Schedule:
    schedule_id (PK, auto-increment)
    bus_name (AM WEST Bus 1)
    route_name
    list of schedule for day

Route - stores a list of stops
Route:
    route_id (PK, auto-increment)
    route_name (NORTH)
    color (hex color)
    list_of_stops

    
Stop - stores a stop's information
Stops:
    stop_id (PK, auto-increment)
    stop_name (UNION)
    latitude
    longitude

Polyline - stores a list of polylines that connect stops
    line_id (PK, auto-increment)
    route_id (FK)
    departure_stop (sting)
    arrival_stop (string)
    list of (lat, long)

DateSchedule - matches days to schedules
    day_id (PK, auto-increment)
    day_name (Monday,etc)
    schedule_id (FK)
    route_id (FK)
"""
