"""Inference utilities for ML models."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

from ml.deploy.lstm import load_lstm_for_route
from backend.cache_dataframe import get_today_dataframe

# Configure logging
logger = logging.getLogger(__name__)

# Global cache for loaded models
# Key: (route_name, polyline_idx), Value: LSTMModel
_MODEL_CACHE: Dict[Tuple[str, int], Any] = {}


async def generate_eta(
    vehicle_ids: List[str],
    sequence_length: int = 10,
    input_columns: List[str] = ['latitude', 'longitude', 'speed_kmh']
) -> Dict[str, float]:
    """
    Generate ETA predictions for a list of vehicles.

    Fetches the latest data, groups vehicles by their current route segment,
    and runs batched inference using the appropriate LSTM models.

    Args:
        vehicle_ids: List of vehicle IDs to predict for.
        sequence_length: Length of input sequence for LSTM (default: 10).
        input_columns: Feature columns (default: ['latitude', 'longitude', 'speed_kmh']).

    Returns:
        Dictionary mapping vehicle_id to predicted ETA (in seconds).
    """
    # 1. Get cached dataframe (processed)
    try:
        df = await get_today_dataframe()
    except Exception as e:
        logger.error(f"Failed to get dataframe for inference: {e}")
        return {}

    if df.empty:
        return {}

    # 2. Filter for requested vehicles
    # Ensure vehicle_id is string
    df['vehicle_id'] = df['vehicle_id'].astype(str)
    target_ids = [str(vid) for vid in vehicle_ids]
    target_df = df[df['vehicle_id'].isin(target_ids)].copy()

    if target_df.empty:
        return {}

    # 3. Group inputs by model (route, polyline_idx)
    # Map: (route, polyline_idx) -> list of (vehicle_id, feature_matrix)
    batches: Dict[Tuple[str, int], List[Tuple[str, np.ndarray]]] = {}

    for vehicle_id, vehicle_df in target_df.groupby('vehicle_id'):
        # Sort by timestamp to ensure correct sequence
        vehicle_df = vehicle_df.sort_values('timestamp')

        # Check if enough data
        if len(vehicle_df) < sequence_length:
            # logger.debug(f"Insufficient data for vehicle {vehicle_id}: {len(vehicle_df)} < {sequence_length}")
            continue

        # Get the most recent sequence
        sequence_df = vehicle_df.tail(sequence_length)

        # Get the current route and segment from the LAST point
        last_point = sequence_df.iloc[-1]

        # Check if route info is available
        if pd.isna(last_point.get('route')) or pd.isna(last_point.get('polyline_idx')):
            # logger.debug(f"Missing route info for vehicle {vehicle_id}")
            continue

        route_name = last_point['route']
        polyline_idx = int(last_point['polyline_idx'])

        # Prepare input features
        # Ensure all columns exist, fill missing with 0
        for col in input_columns:
            if col not in sequence_df.columns:
                sequence_df[col] = 0.0

        features = sequence_df[input_columns].fillna(0).values.astype(np.float32)

        # Verify shape
        if features.shape != (sequence_length, len(input_columns)):
            logger.warning(f"Invalid feature shape for {vehicle_id}: {features.shape}")
            continue

        # Add to batch
        model_key = (route_name, polyline_idx)
        if model_key not in batches:
            batches[model_key] = []
        batches[model_key].append((vehicle_id, features))

    results = {}

    # 4. Run inference for each batch
    for model_key, batch_data in batches.items():
        route_name, polyline_idx = model_key

        # Separate vehicle IDs and features
        batch_vids = [item[0] for item in batch_data]
        batch_features = np.array([item[1] for item in batch_data]) # Shape: (batch_size, seq_len, input_size)

        # Load model
        model = _MODEL_CACHE.get(model_key)

        if model is None:
            try:
                # Load model
                model = load_lstm_for_route(
                    route_name,
                    polyline_idx,
                    input_columns=input_columns,
                    hidden_size=50, # Default from training pipeline
                    num_layers=2,   # Default from training pipeline
                    dropout=0.1
                )
                _MODEL_CACHE[model_key] = model
            except FileNotFoundError:
                # logger.warning(f"No model found for {route_name} segment {polyline_idx}")
                continue
            except Exception as e:
                logger.error(f"Error loading model for {route_name} segment {polyline_idx}: {e}")
                continue

        # Predict
        try:
            # predict returns (n_samples, output_size)
            predictions = model.predict(batch_features)

            # Map predictions back to vehicle IDs
            for i, vid in enumerate(batch_vids):
                eta_seconds = float(predictions[i][0])
                # Clamp negative predictions
                eta_seconds = max(0.0, eta_seconds)
                results[vid] = eta_seconds

        except Exception as e:
            logger.error(f"Prediction batch failed for {route_name} segment {polyline_idx}: {e}")

    return results
