import json
import numpy as np
import math
from pathlib import Path

class Stops:
    # Get the directory where this script is located
    _script_dir = Path(__file__).parent

    with open(_script_dir / 'routes.json', 'r') as f:
        routes_data = json.load(f)

    with open(_script_dir / 'schedule.json', 'r') as f:
        schedule_data = json.load(f)

    # get active routes from schedule
    active_routes = set()
    for day in ['SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY']:
        if day in schedule_data:
            schedule_name = schedule_data[day]
            for bus_schedule in schedule_data[schedule_name].values():
                for time, route_name in bus_schedule:
                    active_routes.add(route_name)

    polylines = {}
    for route_name in active_routes:
        route = routes_data.get(route_name)
        polylines[route_name] = []
        for polyline in route.get('ROUTES', []):
            polylines[route_name].append(np.array(polyline))

    # Pre-calculate segments for vectorized operations
    _route_name_indices = sorted(list(active_routes))
    _all_segments = []

    # Build segments in sorted order of route_idx (implied by _route_name_indices order)
    for r_idx, route_name in enumerate(_route_name_indices):
        route_polys = polylines[route_name]

        for p_idx, poly in enumerate(route_polys):
            if len(poly) < 2:
                # Handle single point as a degenerate segment
                if len(poly) == 1:
                    row = [poly[0][0], poly[0][1], poly[0][0], poly[0][1], r_idx, p_idx]
                    _all_segments.append(np.array([row]))
                continue

            starts = poly[:-1]
            ends = poly[1:]
            n_segs = len(starts)

            # Create array for this polyline's segments
            # columns: lat1, lon1, lat2, lon2, route_idx, poly_idx
            seg_arr = np.column_stack([
                starts,
                ends,
                np.full(n_segs, r_idx),
                np.full(n_segs, p_idx)
            ])
            _all_segments.append(seg_arr)

    if _all_segments:
        polylines_np = np.vstack(_all_segments)
    else:
        polylines_np = np.empty((0, 6))

    @classmethod
    def get_closest_point(cls, origin_point, threshold=0.020, ambiguous=False):
        """
        Find the closest point on any polyline to the given origin point.
        :param origin_point: A tuple or list with (latitude, longitude) coordinates.
        :return: A tuple with the closest point (latitude, longitude), distance to that point,
                route name, and polyline index.
        """
        if cls.polylines_np.shape[0] == 0:
            return None, None, None, None

        point = np.array(origin_point)

        # Extract segment start/end points
        p1 = cls.polylines_np[:, 0:2]
        p2 = cls.polylines_np[:, 2:4]

        # Vector from p1 to p2
        diffs = p2 - p1

        # Squared length of segments (in degree space)
        lengths_sq = np.sum(diffs**2, axis=1)

        # Avoid division by zero
        nonzero_mask = lengths_sq > 1e-12

        # Project point onto lines (parameter t)
        # t = dot(point - p1, diffs) / lengths_sq
        t = np.sum((point - p1) * diffs, axis=1)

        # Initialize t with 0 for zero-length segments (closest point is p1)
        # Apply division only for nonzero segments
        t = np.divide(t, lengths_sq, out=np.zeros_like(t), where=nonzero_mask)

        # Clamp t to segment [0, 1]
        t = np.clip(t, 0, 1)

        # Calculate closest points on the segments
        closest_points = p1 + t[:, np.newaxis] * diffs

        # Calculate Haversine distances to all closest points
        dists = haversine_vectorized(point, closest_points)

        # We need to find the best point *per polyline* to match original logic
        # Data is sorted by route_idx then poly_idx.
        # Find boundaries where (route_idx, poly_idx) changes.
        # columns 4 and 5 are route_idx, poly_idx

        # Helper to identify groups
        # We can just iterate through the segments since we need to aggregate.
        # But looping 10k segments is slow in python.
        # Faster: identify unique group boundaries.

        group_ids = cls.polylines_np[:, 4] * 10000 + cls.polylines_np[:, 5]
        # Find indices where group_ids change
        # Prepend valid index 0
        changes = np.concatenate(([True], group_ids[1:] != group_ids[:-1]))
        boundary_indices = np.nonzero(changes)[0]
        # Append end index for easier slicing
        boundary_indices = np.concatenate((boundary_indices, [len(dists)]))

        closest_data = []

        # Loop over each polyline group
        for i in range(len(boundary_indices) - 1):
            start = boundary_indices[i]
            end = boundary_indices[i+1]

            # Slice for this polyline
            sub_dists = dists[start:end]

            # Find min index in this slice
            min_local_idx = np.argmin(sub_dists)
            min_dist = sub_dists[min_local_idx]
            global_idx = start + min_local_idx

            # Retrieve metadata
            r_idx = int(cls.polylines_np[global_idx, 4])
            p_idx = int(cls.polylines_np[global_idx, 5])
            route_name = cls._route_name_indices[r_idx]

            closest_point = closest_points[global_idx]

            closest_data.append((min_dist, closest_point, route_name, p_idx))

        # Find the overall closest point
        if closest_data:
            closest_routes = sorted(closest_data, key=lambda x: x[0])
            # Check if closest route is significantly closer than others
            if not ambiguous and len(closest_routes) > 1 and haversine(closest_routes[0][1], closest_routes[1][1]) < threshold:
                # If not significantly closer (ambiguous), return None
                return None, None, None, None
            return closest_routes[0]
        return None, None, None, None

    @classmethod
    def is_at_stop(cls, origin_point, threshold=0.020):
        """
        Check if the given point is close enough to any stop.
        :param origin_point: A tuple or list with (latitude, longitude) coordinates.
        :param threshold: Distance threshold to consider as "at stop".
        :return: A tuple with (the route name if close enough, otherwise None,
                the stop name if close enough, otherwise None).
        """
        for route_name, route in cls.routes_data.items():
            for stop in route.get('STOPS', []):
                stop_point = np.array(route[stop]['COORDINATES'])

                distance = haversine(tuple(origin_point), tuple(stop_point))
                if distance < threshold:
                    return route_name, stop
        return None, None

def haversine(coord1, coord2):
    """
    Calculate the great-circle distance between two points on the Earth
    using the Haversine formula.

    Parameters:
        coord1: (lat1, lon1) in decimal degrees
        coord2: (lat2, lon2) in decimal degrees

    Returns:
        Distance in kilometers.
    """
    # Earth radius in kilometers
    R = 6371.0

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert degrees to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def haversine_vectorized(coords1, coords2):
    """
    Vectorized haversine distance between two sets of coordinates.

    Parameters
    ----------
    coords1 : array_like, shape (N, 2)
        Array of (lat, lon) pairs in decimal degrees.
    coords2 : array_like, shape (N, 2)
        Array of (lat, lon) pairs in decimal degrees.

    Returns
    -------
    distances : ndarray, shape (N,)
        Great-circle distances in kilometers.
    """
    # Accept either single (lat,lon) pairs or arrays of pairs. Normalize to 2-D arrays.
    coords1 = np.atleast_2d(np.asarray(coords1, dtype=float))
    coords2 = np.atleast_2d(np.asarray(coords2, dtype=float))

    # Earth radius in kilometers
    R = 6371.0

    lat1 = np.radians(coords1[:, 0])
    lon1 = np.radians(coords1[:, 1])
    lat2 = np.radians(coords2[:, 0])
    lon2 = np.radians(coords2[:, 1])

    dphi = lat2 - lat1
    dlambda = lon2 - lon1

    a = np.sin(dphi / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c
