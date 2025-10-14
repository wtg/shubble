import pandas as pd
import numpy as np
from data.stops import Stops
from datetime import datetime, timedelta

def create_stop_data(df):
   """
   Applies the stop detection function efficiently and returns the processed DataFrame.
   """
   threshold = 0.050  # 50 meters
   stop_info = df.apply(
      lambda row: Stops.is_at_stop(np.array([[row['latitude'], row['longitude']]]), threshold=threshold), 
      axis=1
   )

   df['route_name'] = stop_info.str[0]
   df['stop_name'] = stop_info.str[1]

   # 3. Remove rows where stop_name or route_name is None
   df.dropna(subset=['stop_name', 'route_name'], inplace=True)
   
   df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
   # sort by timestamp
   df.sort_values(by='timestamp', inplace=True)
   
   print(df[['timestamp', 'vehicle_id', 'stop_name', 'route_name']])
   return df

def createETA(df):
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
    
    return df

def main():
   with open('data/data.csv', 'r') as f:
      df = pd.read_csv(f)
      
   print(df.columns)
      
   create_stop_data(df)
   createETA(df)
   print(df[['timestamp', 'vehicle_id', 'stop_name', 'route_name', 'ETA']])
   
   # Create p(t), p(t-1), p(t-2) locations
   create_previous_locations(df)
   
   print(df[['timestamp', 'vehicle_id', 'latitude', 'longitude', 'prev_latitude_1', 'prev_longitude_1', 'prev_latitude_2', 'prev_longitude_2']])

if __name__ == '__main__':
   main()