import json
import math
from pathlib import Path
import numpy as np

# --- Helper Functions ---

def haversine(coord1, coord2):
    """Calculate the great-circle distance between two points on the Earth."""
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
    """Vectorized haversine distance."""
    coords1 = np.atleast_2d(np.asarray(coords1, dtype=float))
    coords2 = np.atleast_2d(np.asarray(coords2, dtype=float))

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


# --- Stops Class ---

class Stops:
    """
    Handles loading, processing, and querying of bus route and stop data.
    """
    _BASE_DIR = Path(__file__).resolve().parent
    _ROUTES_PATH = _BASE_DIR / 'routes.json'
    _SCHEDULE_PATH = _BASE_DIR / 'schedule.json'

    with _ROUTES_PATH.open('r', encoding='utf-8') as f:
        routes_data = json.load(f)

    with _SCHEDULE_PATH.open('r', encoding='utf-8') as f:
        schedule_data = json.load(f)

    # Cache for stop indices: { 'ROUTE_NAME': [ (index, 'StopName'), ... ] }
    _stop_indices_cache = {}

    # Get active routes from schedule
    active_routes = set()
    for day in ['SUNDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY']:
        if day in schedule_data:
            schedule_name = schedule_data[day]
            for bus_schedule in schedule_data[schedule_name].values():
                for time, route_name in bus_schedule:
                    active_routes.add(route_name)

    # Pre-process polylines into NumPy arrays
    polylines = {}
    for route_name in active_routes:
        route = routes_data.get(route_name)
        if not route:
            continue
        polylines[route_name] = []
        for polyline in route.get('ROUTES', []):
            polylines[route_name].append(np.array(polyline))

    @staticmethod
    def get_route_sequence(route_name):
        """
        Returns a tuple: (all_stops, public_stops)
        - all_stops: Includes ghosts (used for calculation)
        - public_stops: Only real stops (used for final JSON output)
        """
        # Safety check if the route exists
        if route_name in Stops.routes_data:
            data = Stops.routes_data[route_name]
            # Return tuple of (Polyline List, Public Stop List)
            return data.get('POLYLINE_STOPS', []), data.get('STOPS', [])

        return [], []
    @classmethod
    def _find_best_match_unfiltered(cls, point_np):
        """Private helper to find the single closest node to a point."""
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
        Returns (distance, point, route_name, polyline_index).
        """
        MAX_VALID_DISTANCE_KM = 0.040  # Increased to 40m to be safe
        AMBIGUITY_DELTA_KM = 0.025

        point_np = np.array(origin_point)
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
            return None, None, None, None

        all_closest_points.sort(key=lambda x: x[0])
        best_match = all_closest_points[0]

        if best_match[0] > MAX_VALID_DISTANCE_KM:
            return None, None, None, None

        # Ambiguity Logic
        ambiguous_matches = [best_match]
        for other_match in all_closest_points[1:]:
            if other_match[0] < (best_match[0] + AMBIGUITY_DELTA_KM):
                ambiguous_matches.append(other_match)
            else:
                break
        
        routes_in_set = set(match[2] for match in ambiguous_matches)
        if len(routes_in_set) == 1:
            return best_match
        
        if last_point is None:
            return None, None, None, None
        
        last_match = cls._find_best_match_unfiltered(np.array(last_point))
        if last_match is None:
            return None, None, None, None
            
        last_route_name = last_match[2]
        last_poly_index = last_match[3]
        
        # Preference logic
        for match in ambiguous_matches:
            if match[2] == last_route_name and match[3] == last_poly_index:
                return match 
        for match in ambiguous_matches:
            if match[2] == last_route_name and match[3] == (last_poly_index + 1):
                return match 
        for match in ambiguous_matches:
            if match[2] == last_route_name:
                return match 
        
        return None, None, None, None

    @classmethod
    def is_at_stop(cls, origin_point, threshold=0.020):
        origin_point_np = np.asarray(origin_point, dtype=float)
        for route_name, route in cls.routes_data.items():
            for stop_key in route.get('STOPS', []):
                if stop_key not in route:
                    continue 
                stop_point = np.array(route[stop_key]['COORDINATES'])
                distance = haversine_vectorized(origin_point_np, [stop_point])[0]
                if distance < threshold:
                    return route_name, stop_key
        return None, None

    # --- NEW: CACHE BUILDER ---
    @classmethod
    def _build_stop_cache(cls, route_name):
        """
        Dynamically finds where each stop is located on the route's polyline.
        Stores a sorted list of (polyline_index, stop_name).
        """
        if route_name not in cls.routes_data or route_name not in cls.polylines:
            return []

        route_data = cls.routes_data[route_name]
        polylines = cls.polylines[route_name]
        
        stop_indices = []

        # Iterate over all stops in this route
        for stop_key in route_data.get('STOPS', []):
            if stop_key not in route_data:
                continue
            
            # Get Stop Coordinates
            stop_coords = np.array(route_data[stop_key]['COORDINATES'])
            
            # Find which polyline index this stop snaps to
            best_poly_index = -1
            min_dist = float('inf')
            
            for poly_index, polyline in enumerate(polylines):
                if polyline.size == 0: continue
                
                dists = haversine_vectorized(stop_coords[np.newaxis, :], polyline)
                local_min = np.min(dists)
                
                if local_min < min_dist:
                    min_dist = local_min
                    best_poly_index = poly_index
            
            if best_poly_index != -1:
                stop_indices.append((best_poly_index, stop_key))

        # Sort stops by their position on the route
        stop_indices.sort(key=lambda x: x[0])
        
        cls._stop_indices_cache[route_name] = stop_indices
        return stop_indices

    # --- FIXED: Get Segment ID ---
    @classmethod
    def get_segment_id(cls, route_name, current_polyline_index):
        """
        Determines the 'From_A_To_B' segment ID based on the vehicle's 
        current index along the route path.
        """
        # 1. Build cache if missing
        if route_name not in cls._stop_indices_cache:
            cls._build_stop_cache(route_name)
        
        stops_list = cls._stop_indices_cache.get(route_name, [])
        
        if not stops_list:
            return None # Could not find any stops for this route

        # 2. Find where the current index fits
        # stops_list looks like: [(5, 'Union'), (15, 'Sage'), (25, 'Blitman')]
        
        # Default to wrap-around (Bus is after the last stop, heading to first)
        last_stop_name = stops_list[-1][1] 
        next_stop_name = stops_list[0][1]  

        for i, (stop_index, stop_name) in enumerate(stops_list):
            # If the stop's index is greater than vehicle's index, 
            # then this 'stop' is the NEXT stop.
            if stop_index > current_polyline_index:
                next_stop_name = stop_name
                
                # The previous stop is the one before this (or the last one if we are at start)
                prev_idx = i - 1 if i > 0 else -1
                last_stop_name = stops_list[prev_idx][1]
                break
            
        return f"From_{last_stop_name}_To_{next_stop_name}"
    
    @staticmethod
    def get_next_stop_name(route_name, polyline_index):
        """
        Returns the name of the next stop on the route based on current polyline index.
        """
        if route_name not in Stops.routes_data:
            return None
        
        route_data = Stops.routes_data[route_name]
        
        # 1. Collect all stops and their offsets
        stop_offsets = []
        # We check both the STOPS list and the stop definitions
        # to ensure we get the offsets correct.
        for stop_key, stop_info in route_data.items():
            if isinstance(stop_info, dict) and 'OFFSET' in stop_info and 'NAME' in stop_info:
                # We use the stop_key (e.g., 'STUDENT_UNION') as the identifier
                stop_offsets.append((stop_info['OFFSET'], stop_key))
        
        # 2. Sort by offset (order of appearance on route)
        stop_offsets.sort()
        
        # 3. Find the first stop with an offset greater than our current index
        for offset, stop_key in stop_offsets:
            if offset > polyline_index:
                return stop_key
        
        # 4. If we are past the last stop, the "next" stop is the first one (Loop)
        if stop_offsets:
             return stop_offsets[0][1]
             
        return None
    
    @staticmethod
    def get_stop_coords(route_name, stop_name):
        """Returns (lat, lon) for a given stop name."""
        if route_name in Stops.routes_data:
            # The stop definition (e.g. "STUDENT_UNION": {...}) is directly 
            # under the route object, NOT inside the 'STOPS' list.
            route_data = Stops.routes_data[route_name]
            
            if stop_name in route_data:
                coords = route_data[stop_name].get('COORDINATES')
                if coords and len(coords) >= 2:
                    return coords[0], coords[1]
                    
        return None, None