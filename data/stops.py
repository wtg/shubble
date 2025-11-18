import json
import math
from pathlib import Path
import numpy as np

class Stops:
    with open('data/routes.json', 'r') as f:
        routes_data = json.load(f)

    with open('data/schedule.json', 'r') as f:
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

    @classmethod
    def get_closest_point(cls, origin_point):
        """
        Find the closest point on any polyline to the given origin point.
        :param origin_point: A tuple or list with (latitude, longitude) coordinates.
        :return: A tuple with the closest point (latitude, longitude), distance to that point,
                route name, and polyline index.
        """
        point = np.array(origin_point)

        closest_data = []
        for route_name, polylines in cls.polylines.items():
            for index, polyline in enumerate(polylines):
                if len(polyline) < 2:
                    # not enough points to form a segment, just check distance to the point itself
                    closest_data.append((np.linalg.norm(point - np.array(polyline[0])), np.array(polyline[0]), route_name, 0))
                    continue

                # Build segments
                lines = np.array([polyline[:-1], polyline[1:]])
                diffs = lines[1, :] - lines[0, :]
                lengths = np.linalg.norm(diffs, axis=1)

                # Handle zero-length segments (duplicate points)
                nonzero_mask = lengths > 0
                if np.any(~nonzero_mask):
                    # check distance directly to these points
                    zero_points = lines[0, ~nonzero_mask]
                    zero_distances = haversine_vectorized(point[np.newaxis, :], zero_points)
                    min_idx = np.argmin(zero_distances)
                    closest_data.append((zero_distances[min_idx], zero_points[min_idx], route_name, min_idx))

                if not np.any(nonzero_mask):
                    # all segments are zero-length, already handled
                    continue

                diffs_normalized = diffs[nonzero_mask] / lengths[nonzero_mask, np.newaxis]
                projections = np.sum((point - lines[0, nonzero_mask]) * diffs_normalized, axis=1)
                projections = np.clip(projections, 0, lengths[nonzero_mask])
                closest_points = lines[0, nonzero_mask] + projections[:, np.newaxis] * diffs_normalized
                distances = haversine_vectorized(point[np.newaxis, :], closest_points)

                min_index = np.argmin(distances)
                closest_data.append((distances[min_index], closest_points[min_index], route_name, index))

        # Find the overall closest point
        if closest_data:
            closest_routes = sorted(closest_data, key=lambda x: x[0])
            # Check if closest route is significantly closer than others
            if len(closest_routes) > 1 and haversine(closest_routes[0][1], closest_routes[1][1]) < 0.050:
                # If not significantly closer, return None to indicate ambiguity
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
    @staticmethod
    def get_segment_id(route_name, current_polyline_index):
        """
        Determines the 'From_A_To_B' segment ID based on the vehicle's 
        current index along the route path.
        """
            
        if route_name not in Stops.routes_data:
            return None

        route_data = Stops.routes_data[route_name]
        # Get list of stops dicts: [{'name': 'Union', 'indices': [0, 1]}, ...]
        stops_list = route_data.get('stops_list', [])
        
        if not stops_list:
            return None

        # We need to find where current_polyline_index fits.
        # We assume stops_list is ordered by the route direction.
        
        last_stop_name = stops_list[-1]['name'] # Default to last if wrapping around
        next_stop_name = stops_list[0]['name']  # Default to first if wrapping around

        # Iterate to find the specific interval
        for i, stop in enumerate(stops_list):
            # Get the 'entry' index of this stop (usually the first index in its list)
            stop_idx = stop['indices'][0]
            
            if stop_idx > current_polyline_index:
                # We found the stop *ahead* of us
                next_stop_name = stop['name']
                # The one before us is the previous in the list (or the very last one if i=0)
                prev_idx = i - 1 if i > 0 else -1
                last_stop_name = stops_list[prev_idx]['name']
                break
            
            # If we haven't found a stop > index, then this stop is currently the "last one passed"
            # We continue the loop. If the loop finishes, it means we are past the last stop
            # and heading towards the first stop (wrap around).

        return f"From_{last_stop_name}_To_{next_stop_name}"

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
    R = 6371.0 # Earth radius in kilometers
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def haversine_vectorized(coords1, coords2):
    """
    Vectorized haversine distance between a single point and an array of points.

    Parameters
    ----------
    coords1 : array_like, shape (1, 2)
        Single (lat, lon) pair in decimal degrees.
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

    R = 6371.0 # Earth radius in kilometers
    lat1 = np.radians(coords1[:, 0])
    lon1 = np.radians(coords1[:, 1])
    lat2 = np.radians(coords2[:, 0])
    lon2 = np.radians(coords2[:, 1])

    dphi = lat2 - lat1
    dlambda = lon2 - lon1

    a = np.sin(dphi / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# --- Stops Class ---

class Stops:
    """
    Handles loading, processing, and querying of bus route and stop data.
    """
    # Resolve data files relative to this module so imports work from any CWD.
    _BASE_DIR = Path(__file__).resolve().parent
    _ROUTES_PATH = _BASE_DIR / 'routes.json'
    _SCHEDULE_PATH = _BASE_DIR / 'schedule.json'

    with _ROUTES_PATH.open('r', encoding='utf-8') as f:
        routes_data = json.load(f)

    with _SCHEDULE_PATH.open('r', encoding='utf-8') as f:
        schedule_data = json.load(f)

    # get active routes from schedule
    active_routes = set()
    for day in ['SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY']:
        if day in schedule_data:
            schedule_name = schedule_data[day]
            for bus_schedule in schedule_data[schedule_name].values():
                for time, route_name in bus_schedule:
                    active_routes.add(route_name)

    # Pre-process polylines into NumPy arrays for efficient calculation
    polylines = {}
    for route_name in active_routes:
        route = routes_data.get(route_name)
        if not route:
            continue
        polylines[route_name] = []
        for polyline in route.get('ROUTES', []):
            polylines[route_name].append(np.array(polyline))

    @classmethod
    def _find_best_match_unfiltered(cls, point_np):
        """
        Private helper to find the single closest node to a point,
        with no filtering or ambiguity checks.
        
        :param point_np: A NumPy array with (latitude, longitude) coordinates.
        :return: A tuple (distance, point, route_name, polyline_index) or None.
        """
        all_closest_points = []
        for route_name, polylines in cls.polylines.items():
            for poly_index, polyline in enumerate(polylines):
                if polyline.size == 0:
                    continue

                distances = haversine_vectorized(point_np[np.newaxis, :], polyline)
                min_dist_idx = np.argmin(distances)
                
                closest_point_data = (
                    distances[min_dist_idx],
                    polyline[min_dist_idx],
                    route_name,
                    poly_index
                )
                all_closest_points.append(closest_point_data)
        
        if not all_closest_points:
            return None
            
        all_closest_points.sort(key=lambda x: x[0])
        return all_closest_points[0]

    @classmethod
    def get_closest_point(cls, origin_point, last_point=None):
        """
        Find the closest point on any polyline to the given origin point.
        
        This method uses a "closest node" approximation. If ambiguity is
        found (e.g., at an intersection), it uses the 'last_point' to
        determine which route the vehicle is currently on.

        :param origin_point: A NumPy array with (latitude, longitude) coordinates.
        :param last_point: (Optional) A NumPy array for the vehicle's last 
                           known (latitude, longitude) to resolve ambiguity.
        :return: A tuple with (distance, point, route_name, polyline_index)
                 Returns (None, None, None, None) if the point is too far
                 from any route or if the route is ambiguous and cannot be
                 resolved.
        """
        
        # --- Constants for logic clarity (in kilometers) ---
        MAX_VALID_DISTANCE_KM = 0.020  # Max distance to be "on" a route (20m)
        AMBIGUITY_DELTA_KM = 0.025   # Max delta to be considered ambiguous (25m)

        point_np = np.array(origin_point)
        all_closest_points = [] # Stores best match (dist, point, route, poly_idx) for EACH polyline

        # 1. Find the closest node for every polyline
        for route_name, polylines in cls.polylines.items():
            for poly_index, polyline in enumerate(polylines):
                
                # Skip any polylines that have no points
                if polyline.size == 0:
                    continue

                # Calculate Haversine distance from the origin to *every node* in this polyline
                distances = haversine_vectorized(point_np[np.newaxis, :], polyline)
                
                # Find the index of the node that is closest to the origin
                min_dist_idx = np.argmin(distances)
                
                # Store the metadata for this polyline's best match
                closest_point_data = (
                    distances[min_dist_idx],      # The minimum distance (km)
                    polyline[min_dist_idx],       # The coordinate [lat, lon] of that node
                    route_name,                   # The route name (e.g., "NORTH")
                    poly_index                    # The index of this polyline within the route
                )
                all_closest_points.append(closest_point_data)

        # 2. Find the single best match from all candidates
        if not all_closest_points:
            # Handle edge case where no polylines exist in the data
            return None, None, None, None

        # Sort all results by distance (index 0) to find the overall best
        all_closest_points.sort(key=lambda x: x[0])
        best_match = all_closest_points[0]

        # 3. Apply Filters
        
        # --- FILTER 1: Proximity Check ---
        # If the *best* match is still too far away (e.g., > 20m),
        # then the vehicle is not considered to be on any route.
        if best_match[0] > MAX_VALID_DISTANCE_KM:
            return None, None, None, None

        # --- FILTER 2: Ambiguity Check ---
        # Find all other matches that are almost as good as the best one
        ambiguous_matches = [best_match]
        for other_match in all_closest_points[1:]:
            if other_match[0] < (best_match[0] + AMBIGUITY_DELTA_KM):
                ambiguous_matches.append(other_match)
            else:
                # List is sorted, so we can stop searching
                break
        
        # Check if there is route ambiguity (matches from *different* routes)
        routes_in_set = set(match[2] for match in ambiguous_matches)
        
        if len(routes_in_set) == 1:
            # No ambiguity, all close matches are on the same route.
            return best_match
        
        # --- AMBIGUITY RESOLUTION ---
        # We have multiple routes that are very close.
        
        if last_point is None:
            # We have no history, so we cannot resolve the ambiguity.
            return None, None, None, None
        
        # We have a last_point, find its route and polyline
        last_match = cls._find_best_match_unfiltered(np.array(last_point))
        
        if last_match is None:
            # Could not determine last route, cannot resolve ambiguity.
            return None, None, None, None
            
        last_route_name = last_match[2]
        last_poly_index = last_match[3]
        
        # Check if any of the ambiguous matches stay on the *exact same* polyline
        for match in ambiguous_matches:
            if match[2] == last_route_name and match[3] == last_poly_index:
                return match # Found a perfect continuation
        
        # Check if any matches are on the *next logical* polyline of the same route
        for match in ambiguous_matches:
            if match[2] == last_route_name and match[3] == (last_poly_index + 1):
                return match # Found a continuation to the next segment
        
        # Check if any matches are on the *same route* at all (e.g., looping back)
        for match in ambiguous_matches:
            if match[2] == last_route_name:
                return match # Found a continuation on the same route
        
        # If the vehicle's last point doesn't help (e.g., it just
        # turned onto a new route), we cannot be certain.
        return None, None, None, None

    @classmethod
    def is_at_stop(cls, origin_point, threshold=0.020):
        """
        Check if the given point is close enough to any stop.
        
        :param origin_point: A tuple or list with (latitude, longitude) coordinates.
        :param threshold: Distance threshold (in km) to consider as "at stop".
        :return: A tuple with (the route name if close enough, otherwise None,
                 the stop name if close enough, otherwise None).
        """
        origin_point_np = np.asarray(origin_point, dtype=float)
        
        for route_name, route in cls.routes_data.items():
            for stop_key in route.get('STOPS', []):
                
                # Ensure the stop key exists as a main entry
                if stop_key not in route:
                    continue 
                    
                stop_point = np.array(route[stop_key]['COORDINATES'])
                distance = haversine_vectorized(origin_point_np, [stop_point])[0]
                
                if distance < threshold:
                    # Return the route and the stop's key (e.g., "STUDENT_UNION")
                    return route_name, stop_key
        
        # If no stop is found within the threshold
        return None, None

