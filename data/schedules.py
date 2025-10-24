from datetime import datetime
import pandas as pd
import numpy as np
from stops import Stops

def get_at_stops():
 
    df = pd.read_csv('data/shuttle.csv')
    
    route_names = []
    stop_names = []
    
    for _, row in df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        try:
            route_name, stop_name = Stops.is_at_stop(np.array([[lat, lon]]))
            route_names.append(route_name)
            stop_names.append(stop_name)
        except:
            route_names.append(None)
            stop_names.append(None)
    
    df['route_name'] = route_names
    df['stop_name'] = stop_names
    print ( df['route_name'].size )
    #Filter to only rows at stops
    at_stops = df.dropna(subset=['route_name', 'stop_name'])

    #TODO Calculate 24 hour Time of Day for vehicle

    # Check If Values Correct
    print(at_stops[['timestamp', 'latitude', 'longitude', 'route_name', 'stop_name']].to_string(index=False))
    
    return at_stops[['timestamp', 'latitude', 'longitude', 'route_name', 'stop_name']].to_dict('records')

def compute_error_metric():

    at_stops = get_at_stops()
    active_schedules = Stops.active_routes

    # TODO Calculate 24 hour Time of Day for each schedule time
    #      For each item in at_stops 
    #           Filter out irrelevant schedules
    #           Calculate error metric between each vehicle time, 

if __name__ == "__main__":
    get_at_stops()
