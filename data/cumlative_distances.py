import json
import math

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

def process_route(route_name, route_data):
    """
    Calculates cumulative distances for every stop in POLYLINE_STOPS
    by summing the distances of the high-resolution polylines in ROUTES.
    """
    print(f"Processing route: {route_name}...")
    
    try:
        polyline_stops = route_data['STOPS']
        segments = route_data['ROUTES']
    except KeyError as e:
        print(f"Error: Route data is missing required key: {e}")
        return None

    if len(segments) != len(polyline_stops) - 1:
        print(f"Warning: Mismatch in data!")
        print(f"  POLYLINE_STOPS has {len(polyline_stops)} stops.")
        print(f"  ROUTES has {len(segments)} segments.")
        print("  This implies the input JSON is incomplete.")
        print("  Processing will be based on the shorter list.")

    cumulative_distances = {}
    total_distance = 0.0

    # The first stop is at distance 0
    start_stop_key = polyline_stops[0]
    cumulative_distances[start_stop_key] = 0.0

    # Iterate over each high-resolution segment
    # (e.g., STUDENT_UNION to COLONIE, then COLONIE to GHOST_STOP_1, etc.)
    for i in range(len(segments)):
        segment_polyline = segments[i]
        segment_distance = 0.0

        # Calculate the length of this single segment
        for j in range(len(segment_polyline) - 1):
            p1 = segment_polyline[j]
            p2 = segment_polyline[j+1]
            segment_distance += get_haversine_distance(p1[0], p1[1], p2[0], p2[1])
        
        # Add to the total route distance
        total_distance += segment_distance
        
        # Assign this total distance to the *destination* stop of the segment
        destination_stop_key = polyline_stops[i+1]
        cumulative_distances[destination_stop_key] = total_distance

    print(f"Processed {len(cumulative_distances)} polyline stops.")
    print(f"Total route distance: {total_distance:.2f} meters.")
    return cumulative_distances

def main():
    try:
        with open('data/routes.json', 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print("Error: 'routes.json' not found.")
        print("Please save your route data to this file.")
        return
    except json.JSONDecodeError:
        print("Error: 'route_data.json' contains invalid JSON.")
        return

    # Assuming you want to process the 'NORTH' route
    route_name = 'NORTH'
    if route_name not in all_data:
        print(f"Error: Route '{route_name}' not found in JSON data.")
        return
        
    north_route_data = all_data[route_name]
    west_route_data = all_data["WEST"]
    
    # Generate the cumulative distances
    north_distances = process_route(route_name, north_route_data)
    west_distances = process_route("WEST", west_route_data)
    if north_distances and west_distances:
        # Save the results to a new file
        output_filename = 'data/cumulative_distances.json'
        with open(output_filename, 'w') as f:
            json.dump({"NORTH": north_distances, "WEST": west_distances}, f, indent=2)
        print(f"Successfully created '{output_filename}'")

if __name__ == "__main__":
    main()