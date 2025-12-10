import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.neighbors import BallTree

DATA_DIR = Path(__file__).resolve().parent
DATASET_PATH = DATA_DIR / 'shubble_october.csv'
ROUTES_PATH = DATA_DIR / 'routes.json'
OUTPUT_PATH = DATA_DIR / 'inter_stop_times.json'

def get_stop_coordinates():
    """
    Parses your specific routes.json structure.
    Iterates through 'NORTH', 'WEST' etc. to find all unique stop definitions.
    """
    print("Parsing routes.json...")
    with open(ROUTES_PATH, 'r') as f:
        routes_data = json.load(f)

    stop_records = []
    seen_stops = set()

    # Iterate over top-level keys (NORTH, WEST, possibly others)
    for route_name, route_data in routes_data.items():
        if not isinstance(route_data, dict): continue
        
        # Iterate over keys inside the route (STUDENT_UNION, COLONIE, etc.)
        for key, value in route_data.items():
            # We identify a "Stop" object if it has COORDINATES and NAME
            if isinstance(value, dict) and 'COORDINATES' in value and 'NAME' in value:
                stop_id = key  # e.g., "STUDENT_UNION" or "GHOST_STOP_1"
                
                # Deduplicate (e.g. Student Union appears in both North and West)
                if stop_id in seen_stops:
                    continue
                
                coords = value['COORDINATES']
                # Ensure we have valid lat/lon
                if len(coords) >= 2:
                    stop_records.append({
                        'stop_id': stop_id,   # Used for lookup
                        'name': value['NAME'], # Display name
                        'lat': coords[0],
                        'lon': coords[1]
                    })
                    seen_stops.add(stop_id)

    return pd.DataFrame(stop_records)

def fast_identify_stops(vehicle_df, stops_df, threshold_meters=50):
    """
    Uses a BallTree to find the closest stop for 600k points instantly.
    """
    print(f"Building spatial tree for {len(stops_df)} unique stops...")
    
    # 1. Convert everything to Radians for Haversine metric
    vehicle_rad = np.radians(vehicle_df[['latitude', 'longitude']].values)
    stops_rad = np.radians(stops_df[['lat', 'lon']].values)
    
    # 2. Build Tree
    tree = BallTree(stops_rad, metric='haversine')
    
    # 3. Query (k=1 finds the single closest stop)
    print(f"Querying tree for {len(vehicle_df)} vehicle positions...")
    distances_rad, indices = tree.query(vehicle_rad, k=1)
    
    # 4. Convert back to meters (Earth radius approx 6371km)
    distances_meters = distances_rad.flatten() * 6371000
    
    # 5. Filter
    # Get the stop_ids corresponding to the nearest indices
    nearest_stop_ids = stops_df.iloc[indices.flatten()]['stop_id'].values
    
    # If distance > threshold, it's not a stop (set to NaN)
    # We use numpy.where for speed
    final_stop_ids = np.where(
        distances_meters <= threshold_meters,
        nearest_stop_ids,
        np.nan
    )
    
    return final_stop_ids

def extract_inter_stop_times():
    print(f"Loading dataset from {DATASET_PATH}...")
    
    # Optimize CSV load
    df = pd.read_csv(DATASET_PATH, usecols=['vehicle_id', 'timestamp', 'latitude', 'longitude'])
    
    # --- FIX: Use mixed format for messy timestamps ---
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    
    df['latitude'] = df['latitude'].astype('float32')
    df['longitude'] = df['longitude'].astype('float32')
    df.sort_values(by=['vehicle_id', 'timestamp'], inplace=True)

    # 1. Get Stop Definitions
    stops_df = get_stop_coordinates()
    if stops_df.empty:
        print("Error: No stops found. Check routes.json path and structure.")
        return

    # 2. Vectorized Stop Identification
    df['stop_id'] = fast_identify_stops(df, stops_df, threshold_meters=40) # 40-50m is usually good

    # 3. Calculate Transitions (Stop A -> Stop B)
    print("Calculating segment durations...")
    
    # Filter down to just rows where we are AT a stop
    stops_only = df.dropna(subset=['stop_id']).copy()
    
    # Get the next stop for this specific vehicle
    stops_only['next_stop_id'] = stops_only.groupby('vehicle_id')['stop_id'].shift(-1)
    stops_only['next_stop_time'] = stops_only.groupby('vehicle_id')['timestamp'].shift(-1)
    
    # Keep only rows where the stop CHANGED (Movement from A -> B)
    # (We ignore rows where Next Stop == Current Stop)
    transitions = stops_only[stops_only['stop_id'] != stops_only['next_stop_id']].dropna(subset=['next_stop_id'])
    
    # Calculate duration
    transitions['duration'] = (transitions['next_stop_time'] - transitions['timestamp']).dt.total_seconds()
    
    # 4. Clean Data (Filter logic)
    # Remove glitches (< 30s) and massive breaks (> 30 mins)
    valid_transitions = transitions[
        (transitions['duration'] > 30) & 
        (transitions['duration'] < 1800)
    ].copy()

    # 5. Aggregate Medians
    # Group by [Start_Stop, End_Stop] to get average time
    print("Aggregating medians...")
    stats = valid_transitions.groupby(['stop_id', 'next_stop_id'])['duration'].median().reset_index()

    # 6. Format Output (Nested JSON)
    # Structure: { "START_STOP_ID": { "END_STOP_ID": seconds } }
    # Note: We aren't grouping by 'Route' (North/West) here because a segment
    # like 'Union' -> 'Colonie' takes the same time regardless of what you call the route.
    output = {}
    
    for _, row in stats.iterrows():
        start = row['stop_id']
        end = row['next_stop_id']
        seconds = int(row['duration'])
        
        if start not in output: output[start] = {}
        output[start][end] = seconds

    # 7. Save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Done! {len(stats)} segments saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_inter_stop_times()