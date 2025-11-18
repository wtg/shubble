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
            print(f"DEBUG: Not enough history. Has {len(vehicle_history_df)}, need 5.")
            return None

        print(f"\n--- DEBUG PREDICTION START ({current_route_name}) ---")
        print(f"DEBUG: Last timestamp: {vehicle_history_df['timestamp'].iloc[-1]}")

        # 2. Feature Engineering
        df = vehicle_history_df.copy()
        
        # Calculate Distance to next stop for each point
        distances = []
        for _, row in df.iterrows():
            # This logic depends on your specific Stops.get_closest_point implementation
            dist_info = Stops.get_closest_point((row['latitude'], row['longitude']))
            dist_m = dist_info[0] if dist_info else 0
            distances.append(dist_m)
        
        df['distance_to_next_stop'] = distances
        print(f"DEBUG: Distances (m): {distances}")

        # Calculate Deltas (Lat/Lon only, Speed delta removed)
        df['lat_delta_1'] = df['latitude'] - df['latitude'].shift(1)
        df['lon_delta_1'] = df['longitude'] - df['longitude'].shift(1)
        
        # Time features
        dt = df['timestamp'].dt
        seconds_in_day = 24 * 60 * 60
        time_seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        df['time_sin'] = np.sin(2 * np.pi * time_seconds / seconds_in_day)
        df['time_cos'] = np.cos(2 * np.pi * time_seconds / seconds_in_day)
        df['day_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)

        # Fill NaNs created by shifting
        df = df.fillna(0)

        # 3. Prepare Inputs
        # Matches training: No speed features
        SEQ_FEATURES = [
            'distance_to_next_stop', 
            'heading_degrees',
            'lat_delta_1', 
            'lon_delta_1'
        ]
        CTX_FEATURES = ['time_sin', 'time_cos', 'day_sin', 'day_cos']

        print(f"DEBUG: Raw Sequence Data (Tail):\n{df[SEQ_FEATURES].tail(1)}")

        # Pass DataFrame to preserve feature names
        X_seq = self.seq_scaler.transform(df[SEQ_FEATURES])
        
        # Context is the last row
        X_ctx = self.ctx_scaler.transform(df.iloc[[-1]][CTX_FEATURES])

        # Segment Embedding
        try:
            if current_route_name in self.segment_encoder.classes_:
                seg_idx = self.segment_encoder.transform([current_route_name])
                print(f"DEBUG: Segment '{current_route_name}' mapped to Index: {seg_idx[0]}")
            else:
                print(f"DEBUG WARNING: Segment '{current_route_name}' NOT FOUND. Fallback to 0.")
                print(f"DEBUG: Valid classes sample: {self.segment_encoder.classes_[:5]}")
                seg_idx = np.array([0]) 
        except Exception as e:
            print(f"DEBUG ERROR: Encoder failure: {e}")
            seg_idx = np.array([0])

        # Reshape Sequence
        X_seq = X_seq.reshape(1, 5, len(SEQ_FEATURES))

        # 4. Predict
        inputs = {
            'sequential_input': X_seq,
            'context_input': X_ctx,
            'segment_input': seg_idx
        }
        
        predicted_scaled = self.model.predict(inputs, verbose=0)
        print(f"DEBUG: Raw Model Output (Scaled): {predicted_scaled[0][0]}")

        predicted_seconds_numpy = self.target_scaler.inverse_transform(predicted_scaled)[0][0]
        final_val = float(max(0, predicted_seconds_numpy))
        
        print(f"DEBUG: Final ETA: {final_val:.2f} seconds")
        print("--- DEBUG END ---\n")

        return final_val