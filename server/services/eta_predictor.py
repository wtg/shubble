import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from pathlib import Path
from datetime import timedelta

# Import your existing Stops logic to calculate distances
from data.stops import Stops 

class ETAPredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ETAPredictor, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def initialize(self, model_dir):
        if self.initialized:
            return

        print("Loading ETA Model and Artifacts...")
        try:
            self.model = tf.keras.models.load_model(model_dir / 'global_lstm_eta_model.keras')
            self.seq_scaler = joblib.load(model_dir / 'seq_scaler.pkl')
            self.ctx_scaler = joblib.load(model_dir / 'ctx_scaler.pkl')
            self.target_scaler = joblib.load(model_dir / 'target_scaler.pkl')
            self.segment_encoder = joblib.load(model_dir / 'segment_encoder.pkl')
            self.initialized = True
            print("ETA Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load ETA model: {e}")
            self.model = None

    def predict(self, vehicle_history_df, current_route_name):
        """
        vehicle_history_df: DataFrame containing last 5 rows of VehicleLocation data.
                            Must have cols: [latitude, longitude, speed_mph, heading_degrees, timestamp]
        """
        if not self.initialized or self.model is None:
            return None

        # 1. Need exactly 5 data points for the sequence
        if len(vehicle_history_df) < 5:
            return None

        # 2. Feature Engineering (Must match training logic exactly)
        df = vehicle_history_df.copy()
        
        # Calculate Distance to next stop for each point
        # Note: This is computationally expensive if done for every request. 
        # Optimization: Cache route info or calculate in worker.
        distances = []
        for _, row in df.iterrows():
            dist_info = Stops.get_closest_point((row['latitude'], row['longitude']))
            # dist_info structure depends on your Stops.get_closest_point return
            # Assuming it returns (distance, coords, route_name, polyline_index)
            dist_m = dist_info[0] if dist_info else 0
            distances.append(dist_m)
        
        df['distance_to_next_stop'] = distances

        # Calculate Deltas
        df['lat_delta_1'] = df['latitude'] - df['latitude'].shift(1)
        df['lon_delta_1'] = df['longitude'] - df['longitude'].shift(1)
        df['speed_delta_1'] = df['speed_mph'] - df['speed_mph'].shift(1)
        
        # Time features
        dt = df['timestamp'].dt
        seconds_in_day = 24 * 60 * 60
        time_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        df['time_sin'] = np.sin(2 * np.pi * time_seconds / seconds_in_day)
        df['time_cos'] = np.cos(2 * np.pi * time_seconds / seconds_in_day)
        df['day_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)

        # Fill NaNs created by shifting (use 0 for deltas)
        df = df.fillna(0)

        # 3. Prepare Inputs
        SEQ_FEATURES = [
            'distance_to_next_stop', 'speed_mph', 'heading_degrees',
            'lat_delta_1', 'lon_delta_1', 'speed_delta_1'
        ]
        CTX_FEATURES = ['time_sin', 'time_cos', 'day_sin', 'day_cos']

        # Scale
        X_seq = self.seq_scaler.transform(df[SEQ_FEATURES].values)
        
        # Context is just the last row
        last_row = df.iloc[-1]
        X_ctx = self.ctx_scaler.transform(last_row[CTX_FEATURES].values.reshape(1, -1))

        # Segment Embedding
        # NOTE: You need to construct the segment_id (e.g., "From_A_To_B") 
        # Or pass the raw route_name if that's what you trained on.
        # This example assumes 'current_route_name' maps to the encoder classes.
        try:
            # Handle unknown segments safely
            if current_route_name in self.segment_encoder.classes_:
                seg_idx = self.segment_encoder.transform([current_route_name])
            else:
                seg_idx = np.array([0]) # Fallback
        except:
            seg_idx = np.array([0])

        # Reshape Sequence to (1, 5, 6)
        X_seq = X_seq.reshape(1, 5, len(SEQ_FEATURES))

        # 4. Predict
        inputs = {
            'sequential_input': X_seq,
            'context_input': X_ctx,
            'segment_input': seg_idx
        }
        
        predicted_scaled = self.model.predict(inputs, verbose=0)
        predicted_seconds = self.target_scaler.inverse_transform(predicted_scaled)[0][0]

        return max(0, predicted_seconds) # Ensure no negative time