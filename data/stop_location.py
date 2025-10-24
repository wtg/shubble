import pandas as pd
import numpy as np
from data.stops import Stops
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import json
import matplotlib.pyplot as plt
from data.get_distances import calculate_distances_for_route

def create_stop_data(df):
   """
   Applies the stop detection function efficiently and returns the processed DataFrame.
   """
   threshold = 0.050  # 50 meters
   stop_info = df.apply(
      lambda row: Stops.is_at_stop(np.array([[row['latitude'], row['longitude']]]), threshold=threshold), 
      axis=1
   )

   route_info = df.apply(
      lambda row: Stops.get_closest_point(np.array([row['latitude'], row['longitude']])), 
      axis=1
   )
   
   # drop rows where first value of tuple is None
   route_info = route_info[route_info.apply(lambda x: x[0] is not None)]
   
   rows = route_info.iloc[100:110]
   
   print(rows)
   print(rows.iloc[0])
   
   print(rows.apply(lambda x: x[3]))
   # Safely extract distance and route name from the returned tuple/list A tuple with the closest point (latitude, longitude), distance to that point,route name, and polyline index.
   route_dist = route_info.apply(lambda x: x[1])
   route_names = route_info.apply(lambda x: x[2])
   df['route_name'] = route_names
   df['stop_name'] = stop_info.apply(lambda x: x[1] if (isinstance(x, (list, tuple, np.ndarray)) and len(x) > 1) else None)
   
   print(df.shape)
   # Change None stop names values to MOVING
   print(df.shape)
   # df['stop_name'].fillna('MOVING', inplace=True)
   df.dropna(subset=['stop_name'], inplace=True)
   print(df.shape)
   
   df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
   
   # sort by timestamp
   df.sort_values(by='timestamp', inplace=True)
   
   print(df[['timestamp', 'vehicle_id', 'stop_name', 'route_name']])
   return df

def createETA(df):
   # Ignore rows where stop_name is MOVING but apply to df in parameter
   # 1. Sort the DataFrame by vehicle_id and timestamp
   df.sort_values(by=['vehicle_id', 'timestamp'], inplace=True)
   # 2. Create a group identifier for consecutive stops (excluding MOVING)
   
   stop_group_id = (df['stop_name'] != df['stop_name'].shift()).cumsum()

   # 3. Find the starting timestamp for the *next* group of stops
   # First, get the start time of each group
   group_start_times = df.groupby(stop_group_id)['timestamp'].first()
   # Then, shift to get the start time of the *next* group
   next_stop_timestamp_map = group_start_times.shift(-1)

   # 4. Map the next stop's timestamp back to the original DataFrame
   df['next_stop_timestamp'] = stop_group_id.map(next_stop_timestamp_map)

   # 5. Calculate the ETA
   # The result will be a Timedelta object
   df['ETA'] = df['next_stop_timestamp'] - df['timestamp']
   df['ETA'] = df['ETA'].dt.total_seconds()

def create_previous_locations(df):
    df.sort_values(by=['vehicle_id', 'timestamp'], inplace=True)

    df['prev_latitude_1'] = df.groupby('vehicle_id')['latitude'].shift(1)
    df['prev_longitude_1'] = df.groupby('vehicle_id')['longitude'].shift(1)
    df['prev_latitude_2'] = df.groupby('vehicle_id')['latitude'].shift(2)
    df['prev_longitude_2'] = df.groupby('vehicle_id')['longitude'].shift(2)
    df['prev_latitude_3'] = df.groupby('vehicle_id')['latitude'].shift(3)
    df['prev_longitude_3'] = df.groupby('vehicle_id')['longitude'].shift(3)
    df['prev_latitude_4'] = df.groupby('vehicle_id')['latitude'].shift(4)
    df['prev_longitude_4'] = df.groupby('vehicle_id')['longitude'].shift(4)
    
    return df

def create_model(df):
   df_clean = df.dropna()

   # Define features (X) and target (y)
   features = [
      'latitude', 'longitude', 
      'prev_latitude_1', 'prev_longitude_1', 
      'prev_latitude_2', 'prev_longitude_2', 
      'prev_latitude_3', 'prev_longitude_3',
      'prev_latitude_4', 'prev_longitude_4', 'timestamp_day', 'timestamp_time'
   ]
   X = df_clean[features]
   y = df_clean['ETA'] # Make sure you are using the ETA in seconds

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   model = KNeighborsRegressor(n_neighbors=2)

   # Fit the model
   model.fit(X_train, y_train)

   # Make predictions
   y_pred = model.predict(X_test)

   mse = mean_squared_error(y_test, y_pred)

   print(X_test.iloc[0], "Y-test:", y_test.iloc[0], "Predicted:", y_pred[0])
    
   # plot the y_test vs y_pred for the last 100 samples
   
   # plt.figure(figsize=(10, 6))
   # plt.plot(y_test.iloc[-100:].values, label='Actual ETA', marker='o')
   # plt.plot(y_pred[-100:], label='Predicted ETA', marker='x')
   # plt.xlabel('Sample Index') 
   # plt.ylabel('ETA (seconds)')
   # plt.title('Actual vs Predicted ETA for Last 100 Samples')
   # plt.legend()
   # plt.grid()
   # plt.show()
   
   # using the routes.json data, plot points that are at a stop vs those that are not
   with open('data/routes.json', 'r') as f:
       routes = json.load(f)

   # Create a set of all stop names, look at the routes.json structure
   stop_names = set()
   # routes.json is keyed by route id (e.g. "NORTH", "WEST"); each route contains
   # a "STOPS" list of stop keys and a mapping for each stop key that has a "NAME".
   for route_data in routes.values():
       for stop_key in route_data.get('STOPS', []):
           stop_entry = route_data.get(stop_key, {})
           # prefer the human-readable NAME if present, otherwise keep the stop key
           name = stop_entry.get('NAME') if isinstance(stop_entry, dict) else None
           stop_names.add(name if name else stop_key)
           
   # Create a new column in the DataFrame to indicate if the vehicle is at a stop
   df['at_stop'] = df['stop_name'].isin(stop_names)

   # Plot the points
   plt.figure(figsize=(10, 6))
   plt.scatter(df['longitude'], df['latitude'], c=df['at_stop'].astype(int), cmap='coolwarm', alpha=0.5)
   plt.colorbar(label='At Stop')
   plt.xlabel('Longitude')
   plt.ylabel('Latitude')
   plt.title('Vehicle Locations: At Stop vs Not At Stop')
   plt.grid()
   plt.show()

   rmse = np.sqrt(mse)
    
   print(f"Model Root Mean Squared Error (RMSE): {rmse:.2f} seconds")

def time_to_seconds(dt_object):
    """
    Converts a datetime object's time component into total seconds from midnight.

    Args:
        dt_object (datetime): The datetime object to process.

    Returns:
        int: The total number of seconds from midnight (00:00:00) of that day.
    """
    hours = dt_object.hour
    minutes = dt_object.minute
    seconds = dt_object.second
    
    total_seconds = (hours * 3600) + (minutes * 60) + seconds
    return total_seconds // 60
 
def main():
   with open('data/data2.csv', 'r') as f:
      df = pd.read_csv(f)
      
   print(df.columns)
      
   create_stop_data(df)
   createETA(df)
   print(df[['timestamp', 'vehicle_id', 'stop_name', 'route_name', 'ETA']])
   
   # Create p(t), p(t-1), p(t-2) locations
   create_previous_locations(df)
   
   df["timestamp_time"] = df.apply(lambda row: time_to_seconds(row['timestamp']), axis=1)
   df["timestamp_day"] = df["timestamp"].dt.dayofweek
   
   # Remove dates before 08-28-2025
   df = df[df['timestamp'] >= '2025-08-28']
   
   # Remove rows with ETA > 2 hours (7200 seconds)
   df = df[df['ETA'] <= 3600]
   print(df[['timestamp', 'vehicle_id', 'latitude', 'longitude', 'prev_latitude_1', 'prev_longitude_1', 'prev_latitude_2', 'prev_longitude_2', 'ETA']])
   df_clean = df.dropna()
   print(df_clean[['timestamp', 'vehicle_id', 'latitude', 'longitude', 'timestamp_day', 'timestamp_time', 'ETA']])
   create_model(df)
if __name__ == '__main__':
   main()