"""
Stop coordinates for the online KNN pipeline (Student Union and Colonie only).

We do NOT read shared/routes.json or use any other stops. Only these two
stops are used for ETA; old scripts that compute "ETA to next stop" over
all route stops are not used here.
"""
from typing import List, Tuple

# Student Union and Colonie (lat, lon) — fixed for this test only
STUDENT_UNION_WEST: Tuple[float, float] = (42.730_309_579_363_59, -73.676_536_791_629_25)
COLONIE_WEST: Tuple[float, float] = (42.737_048, -73.670_397)  # from routes.json

# The only two stops used for online KNN ETA (no other stops from any route)
STOPS_STUDENT_UNION_COLONIE: List[Tuple[float, float]] = [
    STUDENT_UNION_WEST,
    COLONIE_WEST,
]
