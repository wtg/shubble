import pandas as pd
from data.stop_location import process_vehicle_data 
import matplotlib.pyplot as plt

def main():
   try:
      with open('data/shubble_october.csv', 'r') as f:
         # keep only the first 50000 rows for testing
         df = pd.read_csv(f, nrows=200000)
   except FileNotFoundError:
      print("Error: 'data/shubble_october.csv' not found.")
      return
      
      
   print(f"Original data columns: {df.columns.to_list()}")
     
   # <-- NEW: Optimize dtypes *before* processing
   print("Optimizing data types...")
   df['latitude'] = df['latitude'].astype('float32')
   df['longitude'] = df['longitude'].astype('float32')
   df['speed_mph'] = df['speed_mph'].astype('float32')
   df['heading_degrees'] = df['heading_degrees'].astype('float32')
   df['vehicle_id'] = df['vehicle_id'].astype('category')
   # 'timestamp' will be converted to datetime later
   print("Data types optimized.")
   
   separated_dataframes, processed_df = process_vehicle_data(df)

   # plot etas for each route with histograms with x as eta and y as count
   for route_id, route_df in separated_dataframes.items():
       plt.figure(figsize=(10, 6))
       plt.hist(route_df['ETA_seconds'], bins=30, color='blue', alpha=0.7)
       plt.title(f'ETA to Next Stop Distribution for Route {route_id}')
       plt.xlabel('ETA to Next Stop (seconds)')
       plt.ylabel('Count')
       plt.grid(axis='y', alpha=0.75)
   plt.show()

if __name__ == "__main__":
    main()