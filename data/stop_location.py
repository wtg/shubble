import matplotlib
matplotlib.use('Agg')

from pathlib import Path

import pandas as pd
import numpy as np
from data.stops import Stops
from sklearn.model_selection import train_test_split
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
MODELS_DIR = PROJECT_ROOT / 'models'

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
                           ['latitude', 'longitude', 'timestamp', 'vehicle_id',
                            'speed_mph', 'heading_degrees'].

    Returns:
        tuple: (separated_dataframes, processed_df)
    """
    
    # 1. Apply expensive I/O operations (Stops module) ONCE.
    print("Initial sort by vehicle and time...")
    df.sort_values(by=['vehicle_id', 'timestamp'], inplace=True)
    df = _get_stop_and_route_info(df)
    
    # 2. Clean, filter, and sort the data.
    df = _clean_and_enrich_data(df)
    
    # --- Create temporal stop-to-stop segments ---
    df = _create_segments(df)
    
    # 3. Engineer all new features.
    df = _create_features(df)
    
    # 4. Separate the final data into groups.
    separated_dataframes = _separate_dataframes(df)
    
    return separated_dataframes, df

# --- Private Helper Functions ---

def _get_stop_and_route_info(df):
    # (This function is unchanged)
    print("Applying stop detection...")
    threshold = 0.050  # 50 meters
    
    df['last_latitude'] = df.groupby('vehicle_id')['latitude'].shift(1)
    df['last_longitude'] = df.groupby('vehicle_id')['longitude'].shift(1)
    
    df['stop_info_tuple'] = df.apply(
        lambda row: Stops.is_at_stop(np.array([[row['latitude'], row['longitude']]]), threshold=threshold), 
        axis=1
    )
    
    def get_route_info(row):
        origin_point = np.array([row['latitude'], row['longitude']])
        last_point = None
        
        if pd.notna(row['last_latitude']):
            last_point = np.array([row['last_latitude'], row['last_longitude']])
        
        return Stops.get_closest_point(origin_point, last_point=last_point)

    df['route_info_tuple'] = df.apply(get_route_info, axis=1)
    
    df.drop(columns=['last_latitude', 'last_longitude'], inplace=True)
    
    return df

def _clean_and_enrich_data(df):
    # (This function is unchanged)
    print("Cleaning and enriching data...")
    
    df = df[df['route_info_tuple'].apply(lambda x: x is not None and x[0] is not None)].copy()
    
    df['distance_to_route_point'] = df['route_info_tuple'].apply(lambda x: x[0])
    df['coordinates_on_route'] = df['route_info_tuple'].apply(lambda x: x[1])
    df['route_name'] = df['route_info_tuple'].apply(lambda x: x[2])
    df['polyline_index'] = df['route_info_tuple'].apply(lambda x: x[3]) 
    df['stop_name'] = df['stop_info_tuple'].apply(lambda x: x[1] if (isinstance(x, (list, tuple, np.ndarray)) and len(x) > 1) else None)
    
    df['stop_name'].fillna('MOVING', inplace=True)
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['timestamp'], inplace=True)
    
    df = df[df['timestamp'] >= '2025-08-28'].copy()
    
    df.sort_values(by=['vehicle_id', 'route_name', 'timestamp'], inplace=True)
    
    df.drop(columns=['route_info_tuple', 'stop_info_tuple'], inplace=True)
    
    return df

def _create_segments(df):
    # (This function is unchanged)
    print("Creating stop-to-stop segments...")
    
    df['last_stop'] = df['stop_name'].where(df['stop_name'] != 'MOVING')
    df['last_stop'] = df.groupby(['vehicle_id', 'route_name'])['last_stop'].ffill()

    df['next_stop'] = df['stop_name'].where(df['stop_name'] != 'MOVING')
    df['next_stop'] = df.groupby(['vehicle_id', 'route_name'])['next_stop'].bfill()

    df['segment_id'] = 'AT_' + df['stop_name']
    moving_mask = (df['stop_name'] == 'MOVING')
    df.loc[moving_mask, 'segment_id'] = 'From_' + df['last_stop'] + '_To_' + df['next_stop']
    
    noisy_segment_mask = (
        (df['stop_name'] == 'MOVING') &
        (df['last_stop'] == df['next_stop']) &
        (df['last_stop'].notna())
    )
    df.loc[noisy_segment_mask, 'segment_id'] = 'AT_' + df['last_stop']

    df.dropna(subset=['last_stop', 'next_stop', 'segment_id'], inplace=True)
    df.drop(columns=['last_stop', 'next_stop'], inplace=True)
    
    print("Segment creation complete.")
    return df

def _create_features(df):
    # (This function is unchanged)
    print("Engineering features (ETA, Lags, Distance, Time)...")
    
    # 1. Create time-based features (ETA)
    df = _create_eta(df)
    
    # 2. Prune outliers
    df = df[(df['ETA_seconds'] <= 600) & (df['ETA_seconds'] >= 0)].copy()
    
    if df.empty:
        print("No data remaining after ETA pruning.")
        return df

    # 3. Create time-based features (day of week, time of day)
    df = _create_time_features(df)
    
    df = _create_distance_to_stop(df)
    
    # 4. Create location-based features (lags)
    # This will now create lags for speed and heading as well
    df = _create_lagged_features(df)
    
    return df

def _create_eta(df):
    # (This function is unchanged)
    df['stop_arrival_time'] = df['timestamp'].where(df['stop_name'] != 'MOVING')

    is_new_stop_group = (
        (df['stop_name'] != df['stop_name'].shift()) | 
        (df['vehicle_id'] != df['vehicle_id'].shift()) |
        (df['route_name'] != df['route_name'].shift())
    )
    
    df['next_stop_timestamp'] = df['stop_arrival_time'].where(is_new_stop_group)

    df['next_stop_timestamp'] = df.groupby(['vehicle_id', 'route_name'])['next_stop_timestamp'].shift(-1)
    df['next_stop_timestamp'] = df.groupby(['vehicle_id', 'route_name'])['next_stop_timestamp'].bfill()

    df['ETA_seconds'] = (df['next_stop_timestamp'] - df['timestamp']).dt.total_seconds()
    
    df.drop(columns=['stop_arrival_time', 'next_stop_timestamp'], inplace=True)
    
    return df

def _create_time_features(df):
    # (This function is unchanged)
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
    --- MODIFIED to include speed and heading lags ---
    """
    
    print("Creating lagged features for distance, speed, and heading...")
    
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
        
    # --- Create lat/lon, speed, and heading lags ---
    for i in [1, 2, 3, 4]:
        df[f'lat_t_minus_{i}'] = grouped['latitude'].shift(i)
        df[f'lon_t_minus_{i}'] = grouped['longitude'].shift(i)
        df[f'speed_t_minus_{i}'] = grouped['speed_mph'].shift(i)
        df[f'heading_t_minus_{i}'] = grouped['heading_degrees'].shift(i)

    # --- Add Delta Features ---
    df['lat_delta_1'] = df['latitude'] - df['lat_t_minus_1']
    df['lon_delta_1'] = df['longitude'] - df['lon_t_minus_1']
    df['speed_delta_1'] = df['speed_mph'] - df['speed_t_minus_1']
        
    return df

def _create_distance_to_stop(df):
    # (This function is unchanged)
    all_routes_data, all_cumulative_data = _get_route_metadata()

    if all_cumulative_data is None or all_routes_data is None:
        print("Error: Could not load route data. Distances will be NaN.")
        df['distance_to_next_stop'] = np.nan
        df['distance_to_next_stop_1'] = np.nan
        return df

    def get_distances(row):
        route_name = row['route_name']
        if route_name not in all_routes_data or route_name not in all_cumulative_data:
            return np.nan, np.nan
        
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

def _separate_dataframes(df):
    # (This function is unchanged)
    print("Grouping data into separate DataFrames by segment...")
    separated_dataframes = {}
    
    grouped = df.groupby(['route_name', 'segment_id'])
    
    for (route, segment), group_df in grouped:
        separated_dataframes[(route, segment)] = group_df.copy()
        
    return separated_dataframes

def _sanitize_filename(name):
    # (This function is unchanged)
    name = str(name).replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
    return "".join(c for c in name if c.isalnum() or c in ('_', '.')).rstrip()

def create_global_model(training_df, model_type='RF'):
    """
    Creates and evaluates a SINGLE GLOBAL model for all segments.
    
    --- MODIFIED ---
    - Uses One-Hot Encoding for `segment_id`.
    - Includes speed and heading features.
    
    Args:
        training_df (pd.DataFrame): A combined DataFrame of all
                                      segments to be modeled.
        model_type (str): 'RF' (Random Forest). 'KNN' is removed
                          as it's unsuitable for one-hot data.
    """
    
    # Define all features to be used
    # --- NEW: Added speed, heading, and their lags/deltas
    features = [
        'distance_to_next_stop_t_minus_1', 'distance_to_next_stop_t_minus_2', 
        'distance_to_next_stop_t_minus_3', 'distance_to_next_stop_t_minus_4',
        'distance_to_next_stop',
        'speed_t_minus_1', 'speed_t_minus_2', 'speed_t_minus_3', 'speed_t_minus_4',
        'speed_mph',
        'heading_t_minus_1', 'heading_t_minus_2', 'heading_t_minus_3', 'heading_t_minus_4',
        'heading_degrees',
        'lat_delta_1', 'lon_delta_1', 'speed_delta_1',
        'time_sin', 'time_cos',
        'day_sin', 'day_cos',
        'segment_id'  # We include this for One-Hot Encoding
    ]
    
    # Filter for available columns and drop NaNs
    available_features = [f for f in features if f in training_df.columns]
    df_clean = training_df[available_features + ['ETA_seconds']].dropna()
    
    print(f"Total clean rows for global model: {len(df_clean)}")

    if len(df_clean) < 1000:
        print("Not enough combined data to train a global model.")
        return

    X_raw = df_clean[available_features]
    y = df_clean['ETA_seconds'] 

    # --- NEW: Use One-Hot Encoding for 'segment_id' ---
    # This is the scikit-learn equivalent of an Embedding layer
    print("Applying One-Hot Encoding to 'segment_id'...")
    X_encoded = pd.get_dummies(X_raw, columns=['segment_id'], drop_first=True)
    
    # Store encoded feature names for later
    encoded_feature_names = X_encoded.columns.to_list()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)
    
    if X_train.empty or y_train.empty:
        print("Not enough data to train the model after splitting.")
        return

    # --- Model selection logic ---
    if model_type == 'RF':
        print(f"[Global Model] Training RandomForestRegressor...")
        # n_jobs=-1 uses all available CPU cores
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True, min_samples_leaf=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"OOB Score: {model.oob_score_:.4f}")

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'RF'.")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Global Model RMSE: {rmse:.2f} seconds")
    
    if not X_test.empty:
        print(f"Example Prediction (unscaled features): \n{X_test.iloc[0].to_string()}")
        print(f"Y-test: {y_test.iloc[0]:.0f}, Predicted: {y_pred[0]:.0f}")
        
    # --- PLOT 1: Actual vs. Predicted ---
    plot_title = f"{model_type} - Global Model\nActual vs Predicted (Last 200 samples)"
    plt.figure(figsize=(10, 7))
    plt.plot(y_test.iloc[-200:].values, label='Actual ETA', marker='o', markersize=5, linestyle='None')
    plt.plot(y_pred[-200:], label='Predicted ETA', marker='x', markersize=5, linestyle='None')
    plt.xlabel('Sample Index') 
    plt.ylabel('ETA (seconds)')
    plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    
    filename = _sanitize_filename(f"1_global_model_predictions")
    plt.savefig(PLOTS_DIR / f'{filename}.png')
    plt.close()

    # --- PLOT 2: Feature Importances ---
    if model_type == 'RF':
        importances = model.feature_importances_
        indices = np.argsort(importances)[-25:] # Top 25 features

        plt.figure(figsize=(10, 10))
        plt.title('Top 25 Feature Importances (Global Model)')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [encoded_feature_names[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        
        filename = _sanitize_filename(f"2_global_model_feature_importance")
        plt.savefig(PLOTS_DIR / f'{filename}.png')
        plt.close()

    print(f"Global Model Root Mean Squared Error (RMSE): {rmse:.2f} seconds")

# --- MAIN FUNCTION (MODIFIED) ---
def main():
    
    MODEL_TO_USE = 'RF' 
    
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
    
    del processed_df
    gc.collect() 
    print("Freed memory by deleting full processed DataFrame.")
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- NEW: Get route stop lists for validation ---
    global _ROUTES_DATA_CACHE 
    if _ROUTES_DATA_CACHE is None:
        print("ERROR: Route cache is not populated. Cannot validate segments.")
        route_stop_lists = {}
    else:
        route_stop_lists = {
            route: data["STOPS"] 
            for route, data in _ROUTES_DATA_CACHE.items() 
            if "STOPS" in data
        }
    
    # --- MODIFIED: Combine all valid segments into one DataFrame ---
    print("\n--- Combining Segments for Global Model ---")
    
    all_moving_segments_df = []
    
    for (route_name, segment_id), group_df in separated_dataframes.items():
        
        # --- FILTERS (Unchanged) ---
        if segment_id.startswith('AT_'):
            continue
        
        try:
            parts = segment_id.split('_')
            if parts[0] == 'From' and parts[1] == parts[3]:
                continue
        except IndexError:
            pass 

        if segment_id.startswith('From_') and route_name in route_stop_lists:
            try:
                parts = segment_id.split('_')
                from_stop = parts[1]
                to_stop = parts[3]
                
                stop_list = route_stop_lists[route_name]
                
                if from_stop in stop_list and to_stop in stop_list:
                    from_index = stop_list.index(from_stop)
                    to_index = stop_list.index(to_stop)
                    
                    if (to_index - from_index) > 1:
                        continue
                
            except (IndexError, ValueError):
                pass
        
        # This is a valid segment, add it to the list
        all_moving_segments_df.append(group_df)
    
    if not all_moving_segments_df:
        print("No valid 'moving' segments found to train on. Exiting.")
        return
        
    # Create the master DataFrame
    master_training_df = pd.concat(all_moving_segments_df, ignore_index=True)
    
    print(f"Combined {len(all_moving_segments_df)} segments into one "
          f"master DataFrame with {len(master_training_df)} total rows.")
    
    # --- MODIFIED: Call the single global model function ---
    create_global_model(master_training_df, model_type=MODEL_TO_USE)
    
    print(f"\nAll plots have been saved to '{PLOTS_DIR}' directory.")

if __name__ == '__main__':
    main()