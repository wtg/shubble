import json
import math

# --- Helper Functions (Unchanged) ---

def get_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance between two lat/lon points in meters."""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def load_json_file(filename):
    """
    Helper function to load a JSON file.
    In a real module, you might pass the data directly
    instead of loading from a file.
    """
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Required file '{filename}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: File '{filename}' contains invalid JSON.")
        return None

def find_closest_point_on_segment(p_lat, p_lon, a_lat, a_lon, b_lat, b_lon):
    """
    Finds the closest point (projection) on a line segment (a-b) to a point (p).
    Returns a tuple:
    (distance_from_p_to_projection, distance_from_a_to_projection)
    """
    R = 6371000
    avg_lat_rad = math.radians((a_lat + b_lat) / 2.0)
    cos_avg_lat = math.cos(avg_lat_rad)
    
    x_a = R * math.radians(a_lon) * cos_avg_lat
    y_a = R * math.radians(a_lat)
    x_b = R * math.radians(b_lon) * cos_avg_lat
    y_b = R * math.radians(b_lat)
    x_p = R * math.radians(p_lon) * cos_avg_lat
    y_p = R * math.radians(p_lat)
    
    v_x = x_b - x_a
    v_y = y_b - y_a
    w_x = x_p - x_a
    w_y = y_p - y_a
    
    mag_v_sq = v_x**2 + v_y**2
    
    if mag_v_sq == 0:
        dist_p_to_a = get_haversine_distance(p_lat, p_lon, a_lat, a_lon)
        return (dist_p_to_a, 0.0)

    dot_wv = w_x * v_x + w_y * v_y
    t = dot_wv / mag_v_sq
    
    if t < 0.0:
        dist_p_to_a = get_haversine_distance(p_lat, p_lon, a_lat, a_lon)
        return (dist_p_to_a, 0.0)
    if t > 1.0:
        dist_p_to_b = get_haversine_distance(p_lat, p_lon, b_lat, b_lon)
        dist_a_to_b = get_haversine_distance(a_lat, a_lon, b_lat, b_lon)
        return (dist_p_to_b, dist_a_to_b)

    proj_x = x_a + t * v_x
    proj_y = y_a + t * v_y
    
    proj_lon = math.degrees(proj_x / (R * cos_avg_lat))
    proj_lat = math.degrees(proj_y / R)
    
    dist_p_to_proj = get_haversine_distance(p_lat, p_lon, proj_lat, proj_lon)
    dist_a_to_proj = get_haversine_distance(a_lat, a_lon, proj_lat, proj_lon)

    return (dist_p_to_proj, dist_a_to_proj)


def find_current_route_distance(current_point, route_data, cumulative_data):
    """
    Finds the user's effective distance along the route by "snapping"
    them to the closest *point on the polyline path* (a line segment).
    """
    segments = route_data['ROUTES']
    polyline_stops = route_data['STOPS']
    
    min_snap_distance = float('inf')
    best_route_distance = 0.0
    
    p_lat, p_lon = current_point

    for i in range(len(segments)):
        segment_polyline = segments[i]
        start_of_segment_key = polyline_stops[i]
        
        cumulative_dist_at_start_of_segment = cumulative_data[start_of_segment_key]
        distance_into_this_segment = 0.0 

        for k in range(len(segment_polyline) - 1):
            p1 = segment_polyline[k]
            p2 = segment_polyline[k+1]
            a_lat, a_lon = p1
            b_lat, b_lon = p2

            (snap_dist, dist_from_p1_to_proj) = find_closest_point_on_segment(
                p_lat, p_lon, a_lat, a_lon, b_lat, b_lon
            )

            if snap_dist < min_snap_distance:
                min_snap_distance = snap_dist
                best_route_distance = (
                    cumulative_dist_at_start_of_segment +
                    distance_into_this_segment +
                    dist_from_p1_to_proj
                )
            
            distance_into_this_segment += get_haversine_distance(a_lat, a_lon, b_lat, b_lon)

    return best_route_distance, min_snap_distance


# --- INTERNAL CALCULATION FUNCTION ---

def _calculate_loop_distances(current_point, route_data, cumulative_data):
    """
    Internal function to calculate forward-looping distance to all stops.
    
    Returns:
        tuple: (results_list, current_route_distance, snap_distance)
        Returns (None, None, None) on error.
    """
    
    # 1. Get total route length
    try:
        polyline_stops = route_data['STOPS']
        if not polyline_stops:
            print("Error: POLYLINE_STOPS is empty.")
            return None, None, None
        last_stop_key = polyline_stops[-1]
        if last_stop_key not in cumulative_data:
            print(f"Error: Last polyline stop '{last_stop_key}' not in cumulative data.")
            return None, None, None
        total_route_length = cumulative_data[last_stop_key]
    except KeyError:
        print("Error: 'POLYLINE_STOPS' key missing from route data.")
        return None, None, None
    
    if total_route_length == 0:
        print("Error: Total route length is 0. Cannot calculate loop distances.")
        return None, None, None

    # 2. Find how far along the route the user is
    current_route_distance, snap_dist = find_current_route_distance(
        current_point, route_data, cumulative_data
    )
    
    if current_route_distance is None:
        print("Could not calculate current position.")
        return None, None, None

    # 3. Get the list of *official* stops
    try:
        official_stops = route_data['STOPS']
    except KeyError:
        print("Error: 'STOPS' key missing from route data.")
        return None, None, None
        
    results = []

    # 4. Calculate *forward-looping* distance to all stops
    for stop_key in official_stops:
        if stop_key not in cumulative_data:
            print(f"Warning: Stop '{stop_key}' not in cumulative data. Skipping.")
            continue
            
        stop_total_distance = cumulative_data[stop_key]
        stop_name = route_data[stop_key]['NAME']
        
        distance_m = 0.0
        if stop_total_distance >= current_route_distance:
            # Stop is ahead on the current lap
            distance_m = stop_total_distance - current_route_distance
        else:
            # Stop is "behind", so calculate the loop-around distance
            distance_to_end = total_route_length - current_route_distance
            distance_from_start = stop_total_distance
            distance_m = distance_to_end + distance_from_start
            
        results.append({
            "name": stop_name, 
            "distance_m": distance_m
        })

    # 5. Sort results by the calculated future distance
    results.sort(key=lambda x: x['distance_m'])
    
    return results, current_route_distance, snap_dist

def calculate_distances_for_route(current_point, route_name, all_routes_data, all_cumulative_data):
    """
    Calculates the forward-looping distance to all official stops
    from a given current point for a specific route.

    This is the main function to be imported by other modules.

    Args:
        current_point (list): [latitude, longitude]
        route_name (str): The key for the route (e.g., "NORTH", "WEST")
        all_routes_data (dict): The entire loaded 'routes.json' file content.
        all_cumulative_data (dict): The entire loaded 'cumulative_distances.json' file content.

    Returns:
        tuple: (results_list, current_route_distance, snap_distance)
               - results_list: List of dicts [{"name": str, "distance_m": float}]
               - current_route_distance: Float
               - snap_distance: Float
        Returns (None, None, None) on error (e.g., invalid route_name).
    """
    
    # 1. Get the specific data for the requested route
    try:
        route_data = all_routes_data[route_name]
    except KeyError:
        print(f"Error: Route '{route_name}' not found in routes data.")
        return None, None, None
        
    try:
        cumulative_data = all_cumulative_data[route_name]
    except KeyError:
        print(f"Error: Route '{route_name}' not found in cumulative distances data.")
        return None, None, None

    # 2. Call the internal calculator with the specific route data
    return _calculate_loop_distances(current_point, route_data, cumulative_data)


# --- MAIN EXECUTION (for testing) ---

def main():
    """
    Loads data and runs the calculation function for testing.
    This part is not executed when the file is imported.
    """
    
    # --- THIS IS YOUR INPUT ---
    current_point = [42.731300, -73.666500] # Example: Near "ECAV"
    route_to_test = "NORTH"
    
    # 1. Load the pre-processed data
    print("Loading data files...")
    all_cumulative_data = load_json_file('data/cumulative_distances.json')
    all_routes_data = load_json_file('data/routes.json')
    
    if not all_cumulative_data or not all_routes_data:
        print("Failed to load data. Exiting.")
        return 

    # 2. Call the new importable function
    print(f"Calculating '{route_to_test}' distances from: {current_point}\n")
    results, current_dist, snap_dist = calculate_distances_for_route(
        current_point, route_to_test, all_routes_data, all_cumulative_data
    )
    
    # 3. Print the results
    if results:
        print(f"Snapped to route {snap_dist:.2f}m away.")
        print(f"Current position is {current_dist:.2f}m along the route.\n")
        print("--- Distances to All Future Stops (Looping) ---")
        for item in results:
            print(f"  - {item['name']}: {item['distance_m']:.2f} m")
    else:
        print(f"Could not calculate distances for route '{route_to_test}'.")

if __name__ == "__main__":
    main()