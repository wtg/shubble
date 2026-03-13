import json
import numpy as np
import math
from pathlib import Path


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

    # Pre-calculate all stops for vectorized operations
    _all_stops = []
    _stop_route_names = []
    _stop_names = []

    for route_name, route in routes_data.items():
        for stop in route.get('STOPS', []):
            stop_coords = route[stop]['COORDINATES']
            _all_stops.append(stop_coords)
            _stop_route_names.append(route_name)
            _stop_names.append(stop)

    if _all_stops:
        stops_np = np.array(_all_stops)  # Shape: (N, 2) where columns are [lat, lon]
    else:
        stops_np = np.empty((0, 2))

    # Pre-calculate cumulative distances for each polyline
    # Maps (route_name, polyline_idx) -> array of cumulative distances at each point
    _polyline_cumulative_distances = {}
    _polyline_total_lengths = {}

    for route_name in active_routes:
        route_polys = polylines[route_name]
        for p_idx, poly in enumerate(route_polys):
            if len(poly) < 2:
                # Single point polyline
                _polyline_cumulative_distances[(route_name, p_idx)] = np.array([0.0])
                _polyline_total_lengths[(route_name, p_idx)] = 0.0
                continue

            # Calculate segment distances using vectorized haversine
            # poly[:-1] are segment starts, poly[1:] are segment ends
            segment_distances = haversine_vectorized(poly[:-1], poly[1:])

            # Calculate cumulative distances
            cumulative_dists = np.concatenate(([0.0], np.cumsum(segment_distances)))

            _polyline_cumulative_distances[(route_name, p_idx)] = cumulative_dists
            _polyline_total_lengths[(route_name, p_idx)] = cumulative_dists[-1]

    @classmethod
    def get_closest_point(cls, origin_point, threshold=0.020, target_polyline=None):
        """
        Find the closest point on any polyline to the given origin point.
        :param origin_point: A tuple or list with (latitude, longitude) coordinates.
        :param threshold: Distance threshold in km (not currently used in logic).
        :param target_polyline: Optional tuple (route_name, polyline_idx) to filter to a specific polyline.
                                If provided, only considers points on that polyline.
        :return: A tuple with (distance, closest_point_coords, route_name, polyline_idx, segment_idx).
                 The segment_idx indicates which segment of the polyline (0-indexed) contains
                 the closest point, i.e., the point is on the line segment between
                 polyline[segment_idx] and polyline[segment_idx+1].
        """
        if cls.polylines_np.shape[0] == 0:
            return None, None, None, None, None

        point = np.array(origin_point)

        # Filter to target polyline if specified
        segments_to_search = cls.polylines_np
        if target_polyline is not None:
            target_route_name, target_polyline_idx = target_polyline

            # Find the route index for the target route name
            if target_route_name not in cls._route_name_indices:
                return None, None, None, None, None

            target_route_idx = cls._route_name_indices.index(target_route_name)

            # Filter segments to only those matching the target route and polyline
            mask = (cls.polylines_np[:, 4] == target_route_idx) & (cls.polylines_np[:, 5] == target_polyline_idx)
            segments_to_search = cls.polylines_np[mask]

            if segments_to_search.shape[0] == 0:
                return None, None, None, None, None

        # Extract segment start/end points
        p1 = segments_to_search[:, 0:2]
        p2 = segments_to_search[:, 2:4]

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

        group_ids = segments_to_search[:, 4] * 10000 + segments_to_search[:, 5]
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
            r_idx = int(segments_to_search[global_idx, 4])
            p_idx = int(segments_to_search[global_idx, 5])
            route_name = cls._route_name_indices[r_idx]

            closest_point = closest_points[global_idx]

            # min_local_idx is the segment index within this polyline
            segment_idx = min_local_idx

            closest_data.append((min_dist, closest_point, route_name, p_idx, segment_idx))

        # Find the overall closest point
        if closest_data:
            closest_routes = sorted(closest_data, key=lambda x: x[0])
            # If target_polyline is specified, skip ambiguity check (only one polyline to consider)
            if target_polyline is None:
                # Check if closest route is significantly closer than others
                if len(closest_routes) > 1 and closest_routes[1][0] - closest_routes[0][0] < threshold:
                    # If not significantly closer (ambiguous), return None
                    return None, None, None, None, None
            return closest_routes[0]
        return None, None, None, None, None
    @classmethod
    def is_at_stop(cls, origin_point, threshold=0.020):
        """
        Check if the given point is close enough to any stop.
        :param origin_point: A tuple or list with (latitude, longitude) coordinates.
        :param threshold: Distance threshold to consider as "at stop".
        :return: A tuple with (the route name if close enough, otherwise None,
                the stop name if close enough, otherwise None).
        """
        if cls.stops_np.shape[0] == 0:
            return None, None

        point = np.array(origin_point)

        # Calculate distances to all stops at once using vectorized haversine
        distances = haversine_vectorized(point, cls.stops_np)

        # Find the minimum distance
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]

        # Check if the closest stop is within threshold
        if min_distance < threshold:
            return cls._stop_route_names[min_idx], cls._stop_names[min_idx]

        return None, None

    @classmethod
    def get_polyline_distances(cls, origin_point, closest_point_result=None):
        """
        Calculate distance into polyline and distance from end of polyline.

        :param origin_point: A tuple or list with (latitude, longitude) coordinates.
        :param closest_point_result: Optional tuple from get_closest_point().
                                      If None, get_closest_point() will be called.
        :return: A tuple with (distance_from_start, distance_to_end, total_length).
                 All distances in kilometers. Returns (None, None, None) if point
                 cannot be matched to a polyline.

        Example:
            >>> point = (42.7284, -73.6788)
            >>> dist_from_start, dist_to_end, total = Stops.get_polyline_distances(point)
            >>> print(f"Point is {dist_from_start:.3f} km into the route")
            >>> print(f"Point has {dist_to_end:.3f} km remaining")
        """
        # Get closest point info if not provided
        if closest_point_result is None:
            closest_point_result = cls.get_closest_point(origin_point)

        distance, closest_coords, route_name, polyline_idx, segment_idx = closest_point_result

        # Check if we got a valid result
        if route_name is None or polyline_idx is None or segment_idx is None:
            return None, None, None

        # try to convert polyline_idx and segment_idx to int
        try:
            polyline_idx = int(polyline_idx)
            segment_idx = int(segment_idx)
        except (ValueError, TypeError):
            return None, None, None

        # Get the polyline
        key = (route_name, polyline_idx)
        if key not in cls._polyline_cumulative_distances:
            return None, None, None

        cumulative_dists = cls._polyline_cumulative_distances[key]
        total_length = cls._polyline_total_lengths[key]
        poly = cls.polylines[route_name][polyline_idx]

        # Handle edge case: single point polyline
        if len(poly) < 2:
            return 0.0, 0.0, 0.0

        # Get cumulative distance at the start of the segment
        dist_at_segment_start = cumulative_dists[segment_idx]

        # Calculate distance from segment start to closest point
        segment_start = poly[segment_idx]
        segment_end = poly[segment_idx + 1]

        # Calculate the partial distance along the segment
        # We need to project the closest_coords onto the segment to find how far along it is
        partial_dist = haversine(segment_start, closest_coords)

        # Total distance from start of polyline
        distance_from_start = dist_at_segment_start + partial_dist

        # Clamp to [0, total_length]: closest point is computed by linear interpolation
        # in (lat, lon) space, so partial_dist can exceed the segment's haversine length
        # and make distance_from_start > total_length (and dist_to_end negative).
        distance_from_start = max(0.0, min(distance_from_start, total_length))

        # Distance to end of polyline
        distance_to_end = total_length - distance_from_start

        return distance_from_start, distance_to_end, total_length
