import matplotlib
matplotlib.use('Agg')

from pathlib import Path

import pandas as pd
import numpy as np
from data.stops import Stops
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from data.get_distances import calculate_distances_for_route, load_json_file
from sklearn.preprocessing import MinMaxScaler
import gc

DATA_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_DIR.parent
ROUTES_JSON_PATH = DATA_DIR / 'routes.json'
CUMULATIVE_JSON_PATH = DATA_DIR / 'cumulative_distances.json'
DATASET_PATH = DATA_DIR / 'shubble_october.csv'
PLOTS_DIR = PROJECT_ROOT / 'plots'

_ROUTES_DATA_CACHE = None
_CUMULATIVE_DATA_CACHE = None


def _get_route_metadata():
    """
    Lazily load and cache the heavy route JSON payloads so they only need
    to be deserialized once per process invocation.
    """
    global _ROUTES_DATA_CACHE, _CUMULATIVE_DATA_CACHE

    if _ROUTES_DATA_CACHE is None:
        _ROUTES_DATA_CACHE = load_json_file(ROUTES_JSON_PATH)
    if _CUMULATIVE_DATA_CACHE is None:
        _CUMULATIVE_DATA_CACHE = load_json_file(CUMULATIVE_JSON_PATH)

    return _ROUTES_DATA_CACHE, _CUMULATIVE_DATA_CACHE

def process_vehicle_data(df):
    """
    Main processing pipeline for raw vehicle location data.
    
    This function applies stop detection, calculates route info, 
    engineers features (ETA, lags, distance), and separates 
    the final data into distinct route groups.

    Args:
        df (pd.DataFrame): Raw DataFrame containing at least
                           ['latitude', 'longitude', 'timestamp', 'vehicle_id'].

    Returns:
        tuple: (separated_dataframes, processed_df)
            - separated_dataframes (dict): A dictionary where keys are 
              (route_name, segment_id) and values are the
              corresponding DataFrames.
            - processed_df (pd.DataFrame): The single, fully processed 
              DataFrame.
    """
    
    # 1. Apply expensive I/O operations (Stops module) ONCE.
    print("Initial sort by vehicle and time...")
    df.sort_values(by=['vehicle_id', 'timestamp'], inplace=True)
    df = _get_stop_and_route_info(df)
    
    # 2. Clean, filter, and sort the data.
    df = _clean_and_enrich_data(df)
    
    # --- NEW: Create temporal stop-to-stop segments ---
    df = _create_segments(df)
    
    # 3. Engineer all new features.
    df = _create_features(df)
    
    # 4. Separate the final data into groups.
    separated_dataframes = _separate_dataframes(df)
    
    return separated_dataframes, df

# --- Private Helper Functions ---

def _get_stop_and_route_info(df):
    """
    Applies the external 'Stops' module functions to get stop
    and route data for each data point.
    
    This version provides the last known point to help resolve
    ambiguity at intersections.
    
    Assumes df is *already sorted* by vehicle_id and timestamp.
    """
    print("Applying stop detection...")
    threshold = 0.050  # 50 meters
    
    # --- NEW: Create last_point columns ---
    df['last_latitude'] = df.groupby('vehicle_id')['latitude'].shift(1)
    df['last_longitude'] = df.groupby('vehicle_id')['longitude'].shift(1)
    
    # Get stop status (e.g., (True, "Stop Name") or (False, None))
    df['stop_info_tuple'] = df.apply(
        lambda row: Stops.is_at_stop(np.array([[row['latitude'], row['longitude']]]), threshold=threshold), 
        axis=1
    )
    
    # --- MODIFIED: Create a helper to pass last_point ---
    def get_route_info(row):
        origin_point = np.array([row['latitude'], row['longitude']])
        last_point = None
        
        # Check if the shifted values are valid (not NaN)
        if pd.notna(row['last_latitude']):
            last_point = np.array([row['last_latitude'], row['last_longitude']])
        
        return Stops.get_closest_point(origin_point, last_point=last_point)

    df['route_info_tuple'] = df.apply(get_route_info, axis=1)
    
    # Drop helper columns
    df.drop(columns=['last_latitude', 'last_longitude'], inplace=True)
    
    return df

def _clean_and_enrich_data(df):
    """
    Filters bad data, extracts tupled information into columns,
    and sorts the DataFrame for time-series analysis.
    """
    print("Cleaning and enriching data...")
    
    # Filter out any rows where the route info call failed.
    df = df[df['route_info_tuple'].apply(lambda x: x is not None and x[0] is not None)].copy()
    print("Here here")
    # Extract data from tuples into dedicated columns
    df['distance_to_route_point'] = df['route_info_tuple'].apply(lambda x: x[0])
    df['coordinates_on_route'] = df['route_info_tuple'].apply(lambda x: x[1])
    df['route_name'] = df['route_info_tuple'].apply(lambda x: x[2])
    # We keep the original polyline_index as it might be a useful feature,
    # but we will not use it for grouping.
    df['polyline_index'] = df['route_info_tuple'].apply(lambda x: x[3]) 
    df['stop_name'] = df['stop_info_tuple'].apply(lambda x: x[1] if (isinstance(x, (list, tuple, np.ndarray)) and len(x) > 1) else None)
    
    # Standardize 'MOVING' status
    df['stop_name'].fillna('MOVING', inplace=True)
    
    # Clean and validate timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    
    # Prune data before a specific start date.
    df = df[df['timestamp'] >= '2025-08-28'].copy()
    
    # **Critical Step**: Sort by vehicle and time
    # This sort is now essential for the segment logic
    df.sort_values(by=['vehicle_id', 'route_name', 'timestamp'], inplace=True)
    
    # Clean up temporary columns
    df.drop(columns=['route_info_tuple', 'stop_info_tuple'], inplace=True)
    
    return df

def _create_segments(df):
    """
    Creates temporal stop-to-stop segment identifiers.
    ...
    """
    print("Creating stop-to-stop segments...")
    
    # 1. Identify the last non-moving stop name
    df['last_stop'] = df['stop_name'].where(df['stop_name'] != 'MOVING')
    df['last_stop'] = df.groupby(['vehicle_id', 'route_name'])['last_stop'].ffill()

    # 2. Identify the next non-moving stop name
    df['next_stop'] = df['stop_name'].where(df['stop_name'] != 'MOVING')
    df['next_stop'] = df.groupby(['vehicle_id', 'route_name'])['next_stop'].bfill()

    # 3. Create the new segment_id
    df['segment_id'] = 'AT_' + df['stop_name']
    moving_mask = (df['stop_name'] == 'MOVING')
    df.loc[moving_mask, 'segment_id'] = 'From_' + df['last_stop'] + '_To_' + df['next_stop']
    
    # --- NEW FIX ---
    # 4. Re-classify noisy 'MOVING' segments (e.g., From_A_To_A)
    # These are usually just GPS drift at a stop.
    noisy_segment_mask = (
        (df['stop_name'] == 'MOVING') &
        (df['last_stop'] == df['next_stop']) &
        (df['last_stop'].notna()) # Make sure it's not a NaN fill
    )
    # Re-label these as if they were 'AT_STOP'
    df.loc[noisy_segment_mask, 'segment_id'] = 'AT_' + df['last_stop']
    # --- END FIX ---

    # 5. Clean up
    df.dropna(subset=['last_stop', 'next_stop', 'segment_id'], inplace=True)
    df.drop(columns=['last_stop', 'next_stop'], inplace=True)
    
    print("Segment creation complete.")
    return df

def _create_features(df):
    """
    Applies all feature engineering functions to the cleaned DataFrame.
    """
    print("Engineering features (ETA, Lags, Distance, Time)...")
    
    # 1. Create time-based features (ETA)
    df = _create_eta(df)
    
    # 2. Prune outliers based on calculated ETA (e.g., > 10 min)
    df = df[(df['ETA_seconds'] <= 600) & (df['ETA_seconds'] >= 0)].copy()
    
    # Print max/min ETA for verification
    if not df.empty:
        print(f"Max ETA after pruning: {df['ETA_seconds'].max():.2f} seconds")
        print(f"Min ETA after pruning: {df['ETA_seconds'].min():.2f} seconds")
    else:
        print("No data remaining after ETA pruning.")
        return df

    # 3. Create time-based features (day of week, time of day)
    df = _create_time_features(df)
    
    df = _create_distance_to_stop(df)
    
    # 4. Create location-based features (lags)
    df = _create_lagged_features(df)
    
    return df

def _create_eta(df):
    """
    Calculates the ETA (in seconds) to the *next actual stop*,
    ignoring 'MOVING' blocks, for each vehicle.
    Assumes df is sorted by ['vehicle_id', 'timestamp'].
    """
    
    # Identify the arrival timestamp for *actual stops* only
    df['stop_arrival_time'] = df['timestamp'].where(df['stop_name'] != 'MOVING')

    # Identify the start of a new stop group.
    # We must group by route as well, in case a vehicle switches routes
    is_new_stop_group = (
        (df['stop_name'] != df['stop_name'].shift()) | 
        (df['vehicle_id'] != df['vehicle_id'].shift()) |
        (df['route_name'] != df['route_name'].shift()) # Added route check
    )
    
    # Get the timestamp only for the *first* entry of a new stop
    df['next_stop_timestamp'] = df['stop_arrival_time'].where(is_new_stop_group)

    # For each vehicle/route, shift the "next stop arrival time" up.
    df['next_stop_timestamp'] = df.groupby(['vehicle_id', 'route_name'])['next_stop_timestamp'].shift(-1)

    # Propagate the next stop time backward (backward fill).
    df['next_stop_timestamp'] = df.groupby(['vehicle_id', 'route_name'])['next_stop_timestamp'].bfill()

    # Calculate ETA in seconds
    df['ETA_seconds'] = (df['next_stop_timestamp'] - df['timestamp']).dt.total_seconds()
    
    # Clean up helper columns
    df.drop(columns=['stop_arrival_time', 'next_stop_timestamp'], inplace=True)
    
    return df

def _create_time_features(df):
    """
    Creates vectorized, cyclical time-based features from the timestamp.
    Assumes 'timestamp' is already a datetime object.
    """
    dt = df['timestamp'].dt
    
    seconds_in_day = 24 * 60 * 60
    time_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    df['time_sin'] = np.sin(2 * np.pi * time_seconds / seconds_in_day)
    df['time_cos'] = np.cos(2 * np.pi * time_seconds / seconds_in_day)

    days_in_week = 7
    df['day_sin'] = np.sin(2 * np.pi * dt.dayofweek / days_in_week)
    df['day_cos'] = np.cos(2 * np.pi * dt.dayofweek / days_in_week)
    
    return df

def _create_lagged_features(df):
    """
    Creates lagged features and deltas.
    Assumes df is sorted by ['vehicle_id', 'timestamp'].
    """
    
    # Group by vehicle/route to prevent lagging data across groups
    grouped = df.groupby(['vehicle_id', 'route_name'])
    
    # --- Create conditional distance lags ---
    max_distance = df['distance_to_next_stop']
    
    for i in [1, 2, 3, 4]:
        lagged_dist_0 = grouped['distance_to_next_stop'].shift(i)
        lagged_dist_1 = grouped['distance_to_next_stop_1'].shift(i)
        
        condition = lagged_dist_0 > max_distance
        current_lag_col = lagged_dist_0.where(condition, lagged_dist_1)
        
        df[f'distance_to_next_stop_t_minus_{i}'] = current_lag_col
        max_distance = current_lag_col.where(current_lag_col.notna(), max_distance)
        
    # --- Create lat/lon lags ---
    for i in [1, 2, 3, 4]:
        df[f'lat_t_minus_{i}'] = grouped['latitude'].shift(i)
        df[f'lon_t_minus_{i}'] = grouped['longitude'].shift(i)

    # --- Add Delta Features ---
    df['lat_delta_1'] = df['latitude'] - df['lat_t_minus_1']
    df['lon_delta_1'] = df['longitude'] - df['lon_t_minus_1']
        
    return df

def _create_distance_to_stop(df):
    """
    Calculates the distance (e.g., meters) to the next two stops.
    Uses the external 'calculate_distances_for_route' function.
    """
    # Load route data ONCE
    all_routes_data, all_cumulative_data = _get_route_metadata()

    if all_cumulative_data is None or all_routes_data is None:
        print("Error: Could not load route data. Distances will be NaN.")
        df['distance_to_next_stop'] = np.nan
        df['distance_to_next_stop_1'] = np.nan # Assign NaN to both
        return df

    def get_distances(row):
        """
        Gets the distance to the next stop (index 0) and the one after (index 1).
        Returns a tuple (dist_0, dist_1).
        """
        route_name = row['route_name']
        if route_name not in all_routes_data or route_name not in all_cumulative_data:
            return np.nan, np.nan # Return a tuple of NaNs
        
        try:
            dist_list, _, _ = calculate_distances_for_route(
                [row['latitude'], row['longitude']], 
                route_name,
                all_routes_data, 
                all_cumulative_data
            )
            
            dist_0 = np.nan
            dist_1 = np.nan
            
            if dist_list:
                dist_0 = dist_list[0]['distance_m']
                if len(dist_list) > 1:
                    dist_1 = dist_list[1]['distance_m']
            
            return dist_0, dist_1
        
        except Exception as e:
            print(f"Row error: {e}")
            return np.nan, np.nan 

    df[['distance_to_next_stop', 'distance_to_next_stop_1']] = df.apply(
        get_distances, 
        axis=1, 
        result_type='expand'
    )
    
    return df

# --- MODIFIED FUNCTION ---
def _separate_dataframes(df):
    """
    Groups the fully processed DataFrame by route and segment_id.
    """
    print("Grouping data into separate DataFrames by segment...")
    separated_dataframes = {}
    
    # --- MODIFIED: Group by segment_id instead of polyline_index ---
    grouped = df.groupby(['route_name', 'segment_id'])
    
    for (route, segment), group_df in grouped:
        # Create a copy to avoid SettingWithCopyWarning
        separated_dataframes[(route, segment)] = group_df.copy()
        
    return separated_dataframes
 
def _sanitize_filename(name):
    """Removes invalid characters for a file name."""
    name = str(name).replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
    return "".join(c for c in name if c.isalnum() or c in ('_', '.')).rstrip()

# --- MODIFIED FUNCTION ---
def create_model(df, route_name, segment_id, model_type='KNN'):
    """
    Creates and evaluates a model for a specific subset of data.
    
    -- MODIFIED to use segment_id and remove redundant plots --
    
    Args:
        df (pd.DataFrame): The data subset to model.
        route_name (str): The name of the route for titles.
        segment_id (str): The segment identifier (e.g., "From_A_To_B").
        model_type (str): 'KNN' or 'RF' (Random Forest).
    """
    
    # Drop rows with NaNs created by lagging features
    df_clean = df.dropna()
    
    features = [
        'distance_to_next_stop_t_minus_1', 'distance_to_next_stop_t_minus_2', 'distance_to_next_stop_t_minus_3', 'distance_to_next_stop_t_minus_4',
        'distance_to_next_stop',
        'lat_delta_1', 'lon_delta_1',
        'time_sin', 'time_cos', # Added cos back for completeness
        'day_sin', 'day_cos'    # Added cos back for completeness
    ]

    available_features = [f for f in features if f in df_clean.columns]
    
    if len(available_features) < 3: 
        print("Not enough valid features to create a model.")
        return

    X = df_clean[available_features]
    y = df_clean['ETA_seconds'] 

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if X_train.empty or y_train.empty:
        print("Not enough data to train the model after splitting.")
        return

    # --- Model selection logic ---
    if model_type == 'KNN':
        print(f"[{route_name} - {segment_id}] Training KNeighborsRegressor...")
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = KNeighborsRegressor(n_neighbors=3, weights='distance')
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    elif model_type == 'RF':
        print(f"[{route_name} - {segment_id}] Training RandomForestRegressor...")
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"OOB Score: {model.oob_score_:.4f}")

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'KNN' or 'RF'.")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Model RMSE: {rmse:.2f} seconds")
    
    if not X_test.empty:
        print(f"Example Prediction (unscaled features): \n{X_test.iloc[0].to_string()}")
        print(f"Y-test: {y_test.iloc[0]:.0f}, Predicted: {y_pred[0]:.0f}")
        
    # --- PLOT 1: All Data (which is now specific to the segment) ---
    # --- MODIFIED: Use segment_id in title and filename ---
    plot_title = f"{model_type} - {route_name}\n({segment_id})\nActual vs Predicted (Last 100)"
    plt.figure(figsize=(10, 7)) # Made figure taller for new title
    plt.plot(y_test.iloc[-100:].values, label='Actual ETA', marker='o')
    plt.plot(y_pred[-100:], label='Predicted ETA', marker='x', linestyle='--')
    plt.xlabel('Sample Index') 
    plt.ylabel('ETA (seconds)')
    plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.tight_layout() # Adjust plot to prevent title overlap
    
    filename = _sanitize_filename(f"1_all_data_{route_name}_{segment_id}")
    plt.savefig(PLOTS_DIR / f'{filename}.png')
    plt.close()

    # --- PLOTS 2 & 3 REMOVED ---
    # The new grouping by segment_id makes these plots redundant.
    # A group is now *either* 100% 'MOVING' or 100% 'AT_STOP'.
    # Plot 1 now serves the purpose of both.
    
    # --- PLOT 4: Scatter Plot (At Stop vs. Moving) ---
    # This plot is still useful to see the *location* of the data
    # for this specific segment.
    df['at_stop'] = (df['stop_name'] != 'MOVING').astype(int)
    plot_title = f"{route_name}\n({segment_id}) - Vehicle Locations"

    plt.figure(figsize=(10, 7))
    plt.scatter(df['longitude'], df['latitude'], c=df['at_stop'], cmap='coolwarm', alpha=0.5, s=5)
    plt.colorbar(label='At Stop (1=Yes, 0=No)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(plot_title)
    plt.grid()
    plt.tight_layout()

    # --- MODIFIED: Use segment_id in filename ---
    filename = _sanitize_filename(f"4_locations_{route_name}_{segment_id}")
    plt.savefig(PLOTS_DIR / f'{filename}.png')
    plt.close()

    print(f"Model Root Mean Squared Error (RMSE): {rmse:.2f} seconds")
    
# --- MAIN FUNCTION (MODIFIED) ---
def main():
    
    MODEL_TO_USE = 'RF'  # Options: 'KNN' or 'RF'
    
    print("hello")
    try:
        with DATASET_PATH.open('r') as f:
            df = pd.read_csv(f, nrows=200000)
            
    except FileNotFoundError:
        print(f"Error: '{DATASET_PATH}' not found.")
        return
        
    print(f"Original data columns: {df.columns.to_list()}")
        
    print("Optimizing data types...")
    df['latitude'] = df['latitude'].astype('float32')
    df['longitude'] = df['longitude'].astype('float32')
    df['speed_mph'] = df['speed_mph'].astype('float32')
    df['heading_degrees'] = df['heading_degrees'].astype('float32')
    df['vehicle_id'] = df['vehicle_id'].astype('category')
    print("Data types optimized.")
    
    separated_dataframes, processed_df = process_vehicle_data(df)
    
    print("\n--- Processing Complete ---")
    
    print(f"Created {len(separated_dataframes)} separate dataframes.")
    print(f"Segment Keys: {list(separated_dataframes.keys())}")
    
    del processed_df
    gc.collect() 
    print("Freed memory by deleting full processed DataFrame.")
    
    if separated_dataframes:
        first_key = list(separated_dataframes.keys())[0]
        print(f"\n--- Head of group {first_key} ---")
        print(separated_dataframes[first_key][['timestamp', 'stop_name', 'segment_id', 'ETA_seconds']].head())
            
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- NEW: Get route stop lists for validation ---
    # This global cache was populated by process_vehicle_data()
    global _ROUTES_DATA_CACHE 
    if _ROUTES_DATA_CACHE is None:
        print("ERROR: Route cache is not populated. Cannot validate segments.")
        # Fallback: create an empty dict to avoid crashing
        route_stop_lists = {}
    else:
        # Create a simple lookup: { route_name: [STOP1, STOP2, ...] }
        route_stop_lists = {
            route: data["STOPS"] 
            for route, data in _ROUTES_DATA_CACHE.items() 
            if "STOPS" in data
        }
    # --- END NEW ---
    
    # --- MODIFIED: Loop over segment_id ---
    print("\n--- Creating Models for Each Group ---")
    for (route_name, segment_id), group_df in separated_dataframes.items():
        print(f"\n--- Modeling for Route: {route_name}, Segment: {segment_id} ---")
        
        # --- FILTER 1: Skip 'AT_STOP' segments ---
        if segment_id.startswith('AT_'):
            print("Skipping model (Segment is 'AT_STOP')")
            continue
        
        # --- FILTER 2: Skip 'From_A_To_A' noisy segments ---
        try:
            parts = segment_id.split('_')
            if parts[0] == 'From' and parts[1] == parts[3]:
                print(f"Skipping model (Segment start/end is the same: {segment_id})")
                continue
        except IndexError:
            pass # Not a 'From_To' segment, let it pass

        # --- NEW FILTER 3: Skip 'From_A_To_C' (skipped stop) segments ---
        if segment_id.startswith('From_') and route_name in route_stop_lists:
            try:
                parts = segment_id.split('_')
                from_stop = parts[1]
                to_stop = parts[3]
                
                stop_list = route_stop_lists[route_name]
                
                # Check if both stops are in the official route list
                if from_stop in stop_list and to_stop in stop_list:
                    from_index = stop_list.index(from_stop)
                    to_index = stop_list.index(to_stop)
                    
                    # Check if the 'to' stop is more than 1 stop after the 'from' stop
                    if (to_index - from_index) > 1:
                        print(f"Skipping model (Segment skips stops. From: {from_stop} (idx {from_index}) "
                              f"To: {to_stop} (idx {to_index}))")
                        continue
                
            except (IndexError, ValueError) as e:
                # IndexError: segment_id not in 'From_X_To_Y' format
                # ValueError: stop_name not in the official list
                print(f"Warning: Could not validate segment '{segment_id}' for route '{route_name}'. Error: {e}")
                # We'll let it pass and try to model it anyway.
                pass
        # --- END NEW FILTER ---

        if len(group_df) > 50:
            create_model(group_df, route_name, segment_id, model_type=MODEL_TO_USE)
        else:
            print(f"Skipping model (not enough data: {len(group_df)} rows)")
    
    print(f"\nAll plots have been saved to '{PLOTS_DIR}' directory.")

if __name__ == '__main__':
    main()