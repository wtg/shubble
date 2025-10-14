import pandas as pd
import numpy as np
from data.stops import Stops

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
    
    print(df[['timestamp', 'vehicle_id', 'latitude', 'longitude', 'stop_name', 'route_name']])
    
    # sort by timestamp
    df.sort_values(by='timestamp', inplace=True)
    
    return df
 
def main():
   with open('data/data.csv', 'r') as f:
      df = pd.read_csv(f)
      
   print(df.columns)
      
   create_stop_data(df)
   
if __name__ == '__main__':
   main()