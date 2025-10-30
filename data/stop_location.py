import pandas as pd
import numpy as np
from data.stops import Stops
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor  # <-- NEW: Added Random Forest
from sklearn.metrics import mean_squared_error
import json
import matplotlib.pyplot as plt
from data.get_distances import calculate_distances_for_route, load_json_file
from sklearn.preprocessing import MinMaxScaler

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
              (route_name, polyline_index) and values are the
              corresponding DataFrames.
            - processed_df (pd.DataFrame): The single, fully processed 
              DataFrame.
    """
    
    # 1. Apply expensive I/O operations (Stops module) ONCE.
    # --- MODIFIED: Ensure data is sorted *first* for the ambiguity logic ---
    print("Initial sort by vehicle and time...")
    df.sort_values(by=['vehicle_id', 'timestamp'], inplace=True)
    df = _get_stop_and_route_info(df)
    
    # 2. Clean, filter, and sort the data.
    df = _clean_and_enrich_data(df)
    
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
    # This assumes df is already sorted by vehicle_id, timestamp
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
    df['polyline_index'] = df['route_info_tuple'].apply(lambda x: x[3])
    df['stop_name'] = df['stop_info_tuple'].apply(lambda x: x[1] if (isinstance(x, (list, tuple, np.ndarray)) and len(x) > 1) else None)
    
    # Standardize 'MOVING' status
    df['stop_name'].fillna('MOVING', inplace=True)
    
    # Clean and validate timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    
    # Prune data before a specific start date.
    df = df[df['timestamp'] >= '2025-08-28'].copy()
    
    # **Critical Step**: Sort by vehicle and time (redundant but safe)
    df.sort_values(by=['vehicle_id', 'timestamp'], inplace=True)
    
    # Clean up temporary columns
    df.drop(columns=['route_info_tuple', 'stop_info_tuple'], inplace=True)
    
    return df

def _create_features(df):
    """
    Applies all feature engineering functions to the cleaned DataFrame.
    """
    print("Engineering features (ETA, Lags, Distance, Time)...")
    
    # 1. Create time-based features (ETA)
    df = _create_eta(df)
    
    # 2. Prune outliers based on calculated ETA (e.g., > 30 min)
    # Using 1800 seconds (30 min) as per your code.
    # Also remove negative ETAs which can result from stale data
    df = df[(df['ETA_seconds'] <= 1800) & (df['ETA_seconds'] >= 0)].copy()
    
    # Print max/min ETA for verification and their rows
    print(f"Max ETA after pruning: {df['ETA_seconds'].max():.2f} seconds")
    print(f"Min ETA after pruning: {df['ETA_seconds'].min():.2f} seconds")
    # print("Rows with Max ETA:")
    # print(df[df['ETA_seconds'] == df['ETA_seconds'].max()][['vehicle_id', 'timestamp', 'ETA_seconds', 'stop_name', 'latitude', 'longitude', 'route_name', 'polyline_index', 'distance_to_route_point', 'coordinates_on_route']])
    # print("Rows with Min ETA:")
    # print(df[df['ETA_seconds'] == df['ETA_seconds'].min()][['vehicle_id', 'timestamp', 'ETA_seconds', 'stop_name', 'latitude', 'longitude']])

    # 3. Create time-based features (day of week, time of day)
    df = _create_time_features(df)
    
    # 4. Create location-based features (lags)
    df = _create_lagged_features(df)
    
    # 5. Create distance-based features
    df = _create_distance_to_stop(df)
    
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
    is_new_stop_group = (
        (df['stop_name'] != df['stop_name'].shift()) | 
        (df['vehicle_id'] != df['vehicle_id'].shift())
    )
    
    # Get the timestamp only for the *first* entry of a new stop
    df['next_stop_timestamp'] = df['stop_arrival_time'].where(is_new_stop_group)

    # For each vehicle, shift the "next stop arrival time" up.
    df['next_stop_timestamp'] = df.groupby('vehicle_id')['next_stop_timestamp'].shift(-1)

    # Propagate the next stop time backward (backward fill).
    df['next_stop_timestamp'] = df.groupby('vehicle_id')['next_stop_timestamp'].bfill()

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
    
    # --- FIX for timestamp_time_seconds ---
    seconds_in_day = 24 * 60 * 60
    time_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
    df['time_sin'] = np.sin(2 * np.pi * time_seconds / seconds_in_day)
    df['time_cos'] = np.cos(2 * np.pi * time_seconds / seconds_in_day)

    # --- FIX for timestamp_day_of_week ---
    days_in_week = 7
    df['day_sin'] = np.sin(2 * np.pi * dt.dayofweek / days_in_week)
    df['day_cos'] = np.cos(2 * np.pi * dt.dayofweek / days_in_week)
    
    return df

def _create_lagged_features(df):
    """
    Creates 6 new fields for the last 3 location points
    and delta features for the most recent change.
    Assumes df is sorted by ['vehicle_id', 'timestamp'].
    """
    
    # Group by vehicle to prevent lagging data from one vehicle to another
    grouped = df.groupby('vehicle_id')
    
    for i in [1, 2, 3]:
        df[f'lat_t_minus_{i}'] = grouped['latitude'].shift(i)
        df[f'lon_t_minus_{i}'] = grouped['longitude'].shift(i)
        
    # --- NEW: Add Delta Features ---
    # These explicitly state the most recent change in position
    df['lat_delta_1'] = df['latitude'] - df['lat_t_minus_1']
    df['lon_delta_1'] = df['longitude'] - df['lon_t_minus_1']
        
    return df

def _create_distance_to_stop(df):
    """
    Calculates the distance (e.g., meters) to the next stop.
    Uses the external 'calculate_distances_for_route' function.
    """
    # Load route data ONCE
    all_cumulative_data = load_json_file('data/cumulative_distances.json')
    all_routes_data = load_json_file('data/routes.json')

    if all_cumulative_data is None or all_routes_data is None:
        print("Error: Could not load route data. Distances will be NaN.")
        df['distance_to_next_stop'] = np.nan
        return df

    # This operation is row-wise and does not depend on sorting.
    def get_dist(row):
        route_name = row['route_name']
        if route_name not in all_routes_data or route_name not in all_cumulative_data:
            return None  # Or np.nan
        
        try:
            # Call the imported function with the correct route data for this row
            dist_list, _, _ = calculate_distances_for_route(
                [row['latitude'], row['longitude']], 
                route_name,
                all_routes_data, 
                all_cumulative_data
            )
            # find_current_route_distance returns (results_list, current_dist, snap_dist)
            # We want the distance to the *next* stop, which is the first item in the list.
            if dist_list:
                return dist_list[0]['distance_m']
            else:
                return np.nan 
        except Exception as e:
            print(f"Row error: {e}")
            return np.nan

    df['distance_to_next_stop'] = df.apply(get_dist, axis=1)
    return df

def _separate_dataframes(df):
    """
    Groups the fully processed DataFrame by route and polyline index.
    """
    print("Grouping data into separate DataFrames...")
    separated_dataframes = {}
    grouped = df.groupby(['route_name', 'polyline_index'])
    
    for (route, poly_index), group_df in grouped:
        # Create a copy to avoid SettingWithCopyWarning
        separated_dataframes[(route, poly_index)] = group_df.copy()
        
    return separated_dataframes
 
# --- MODIFIED FUNCTION ---
def create_model(df, route_name, polyline_index, model_type='KNN'):
    """
    Creates and evaluates a model for a specific subset of data.
    
    Args:
        df (pd.DataFrame): The data subset to model.
        route_name (str): The name of the route for titles.
        polyline_index (int): The polyline index for titles.
        model_type (str): 'KNN' or 'RF' (Random Forest).
    """
    
    # Drop rows with NaNs created by lagging features
    df_clean = df.dropna()
    
    # --- MODIFIED: Updated feature list ---
    features = [
        'latitude', 'longitude', 
        'lat_t_minus_1', 'lon_t_minus_1', 
        'lat_t_minus_2', 'lon_t_minus_2', 
        'lat_t_minus_3', 'lon_t_minus_3',
        'lat_delta_1', 'lon_delta_1',         # Added delta features
        'distance_to_next_stop', 'speed_mph', 'heading_degrees', 
        'time_sin', 'time_cos',               # Added cyclical features
        'day_sin', 'day_cos'                  # Added cyclical features
    ]

    # Only use features that actually exist in the cleaned df
    available_features = [f for f in features if f in df_clean.columns]
    
    if len(available_features) < 3: # Not enough features to model
        print("Not enough valid features to create a model.")
        return

    X = df_clean[available_features]
    y = df_clean['ETA_seconds'] 

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    if X_train.empty or y_train.empty:
        print("Not enough data to train the model after splitting.")
        return

    # --- NEW: Model selection logic ---
    if model_type == 'KNN':
        print(f"[{route_name} - {polyline_index}] Training KNeighborsRegressor...")
        
        # 1. Initialize and apply the scaler
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 2. Tuned KNN model
        model = KNeighborsRegressor(n_neighbors=5, weights='distance')
        
        # 3. Fit and predict using SCALED data
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    elif model_type == 'RF':
        print(f"[{route_name} - {polyline_index}] Training RandomForestRegressor...")
        
        # 1. RF doesn't need scaling
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True)
        
        # 2. Fit and predict using ORIGINAL data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"OOB Score: {model.oob_score_:.4f}")

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'KNN' or 'RF'.")
    # --------------------------------------------

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Model RMSE: {rmse:.2f} seconds")
    
    # To print the example, you can still use the unscaled X_test
    print(f"Example Prediction (unscaled features): \n{X_test.iloc[0].to_string()}")
    print(f"Y-test: {y_test.iloc[0]:.0f}, Predicted: {y_pred[0]:.0f}")
      
    # --- PLOT 1: Original (All Data) ---
    plot_title = f"{model_type} - {route_name} (Poly {polyline_index}) - Actual vs Predicted (Last 100)"
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.iloc[-100:].values, label='Actual ETA (All)', marker='o')
    plt.plot(y_pred[-100:], label='Predicted ETA (All)', marker='x')
    plt.xlabel('Sample Index') 
    plt.ylabel('ETA (seconds)')
    plt.title(plot_title)
    plt.legend()
    plt.grid()

    # --- Logic to filter test data for new plots ---
    test_stop_names = df_clean.loc[y_test.index, 'stop_name']
    is_moving_mask = (test_stop_names == 'MOVING')
    is_at_stop_mask = (test_stop_names != 'MOVING')
    
    y_test_moving = y_test[is_moving_mask]
    y_test_at_stop = y_test[is_at_stop_mask]
    y_pred_moving = y_pred[is_moving_mask.values]
    y_pred_at_stop = y_pred[is_at_stop_mask.values]
    
    # --- PLOT 2: NEW (Moving Data Only) ---
    if len(y_test_moving) > 0:
        num_moving_samples = min(100, len(y_test_moving)) # Get up to 100 samples
        plot_title = f"{model_type} - {route_name} (Poly {polyline_index}) - 'Moving' (Last {num_moving_samples})"
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_moving.iloc[-num_moving_samples:].values, label='Actual ETA (Moving)', marker='o', color='blue')
        plt.plot(y_pred_moving[-num_moving_samples:], label='Predicted ETA (Moving)', marker='x', color='cyan', linestyle='--')
        plt.xlabel('Sample Index') 
        plt.ylabel('ETA (seconds)')
        plt.title(plot_title)
        plt.legend()
        plt.grid()
    else:
        print("No 'MOVING' samples in test set to plot.")

    # --- PLOT 3: NEW (At Stop Data Only) ---
    if len(y_test_at_stop) > 0:
        num_at_stop_samples = min(100, len(y_test_at_stop)) # Get up to 100 samples
        plot_title = f"{model_type} - {route_name} (Poly {polyline_index}) - 'At Stop' (Last {num_at_stop_samples})"
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_at_stop.iloc[-num_at_stop_samples:].values, label='Actual ETA (At Stop)', marker='o', color='red')
        plt.plot(y_pred_at_stop[-num_at_stop_samples:], label='Predicted ETA (At Stop)', marker='x', color='magenta', linestyle='--')
        plt.xlabel('Sample Index') 
        plt.ylabel('ETA (seconds)')
        plt.title(plot_title)
        plt.legend()
        plt.grid()
    else:
        print("No 'At Stop' samples in test set to plot.")
        
    
    # --- PLOT 4: Scatter Plot (At Stop vs. Moving) ---
    df['at_stop'] = (df['stop_name'] != 'MOVING').astype(int)
    plot_title = f"{route_name} (Poly {polyline_index}) - Vehicle Locations"

    plt.figure(figsize=(10, 6))
    plt.scatter(df['longitude'], df['latitude'], c=df['at_stop'], cmap='coolwarm', alpha=0.5, s=5)
    plt.colorbar(label='At Stop (1=Yes, 0=No)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(plot_title)
    plt.grid()

    print(f"Model Root Mean Squared Error (RMSE): {rmse:.2f} seconds")
    # plt.show() is called once at the end of main()
    
# --- MAIN FUNCTION (MODIFIED) ---
def main():
    
    # --- NEW: Set your model type here ---
    MODEL_TO_USE = 'RF'  # Options: 'KNN' or 'RF'
    
    print("hello")
    try:
        with open('data/data2.csv', 'r') as f:
            df = pd.read_csv(f)
    except FileNotFoundError:
        print("Error: 'data/data2.csv' not found.")
        return
        
    print(f"Original data columns: {df.columns.to_list()}")
     
    separated_dataframes, processed_df = process_vehicle_data(df)
    
    print("\n--- Processing Complete ---")
    
    print(f"Created {len(separated_dataframes)} separate dataframes.")
    print(f"Keys: {list(separated_dataframes.keys())}")
    
    # Example: Print the head of the first separated dataframe
    if separated_dataframes:
        first_key = list(separated_dataframes.keys())[0]
        print(f"\n--- Head of group {first_key} ---")
        print(separated_dataframes[first_key].head())
        # print(separated_dataframes[first_key][['address_id', 'address_name']] )
        # print(separated_dataframes[first_key].columns.to_list())
            
    print(f"\n--- Head of FULL processed DataFrame ---")
    print(processed_df.head())
    
    
    # --- NEW: Loop and create a model for each group ---
    print("\n--- Creating Models for Each Group ---")
    for (route_name, poly_index), group_df in separated_dataframes.items():
        print(f"\n--- Modeling for Route: {route_name}, Polyline Index: {poly_index} ---")
        
        # Add a check to ensure there's enough data to model
        if len(group_df) > 50: # Arbitrary threshold for train/test split
            create_model(group_df, route_name, poly_index, model_type=MODEL_TO_USE)
        else:
            print(f"Skipping model (not enough data: {len(group_df)} rows)")
    
    # --- NEW: Show all plots at the very end ---
    print("\nDisplaying all model plots...")
    plt.show()

if __name__ == '__main__':
    main()
