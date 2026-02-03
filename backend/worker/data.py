"""Data prediction utilities for worker."""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta

from backend.cache_dataframe import get_today_dataframe
from ml.deploy.lstm import load_lstm_for_route
from ml.deploy.arima import load_arima
from ml.training.train import fit_arima
from backend.models import ETA, PredictedLocation
from backend.database import create_async_db_engine, create_session_factory
from backend.config import settings
from backend.cache import cache, soft_clear_namespace
from ml.cache import get_polyline_dir
from shared.stops import Stops
import asyncio

logger = logging.getLogger(__name__)

P = 3
D = 0
Q = 2

# Cache for loaded models: (route_name, polyline_idx) -> LSTMModel
_MODEL_CACHE: Dict[Tuple[str, int], Any] = {}

@cache(soft_ttl=300, hard_ttl=3600, namespace="average_travel_time")
async def load_average_travel_time(route: str, polyline_idx: int) -> Optional[float]:
    """Load average travel time for a polyline from CSV.

    Cached for 5 minutes.

    Args:
        route: Route name
        polyline_idx: Polyline index

    Returns:
        Average travel time in seconds, or None if not found
    """
    polyline_dir = get_polyline_dir(route, polyline_idx)
    csv_path = polyline_dir / "average_travel_time.csv"

    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return None
        return float(df.iloc[0]['avg_travel_time_seconds'])
    except Exception as e:
        logger.warning(f"Failed to load average travel time for {route} polyline {polyline_idx}: {e}")
        return None

async def _get_vehicle_data(vehicle_ids: List[str]) -> pd.DataFrame:
    """Helper to get and filter vehicle data."""
    try:
        df = await get_today_dataframe()
    except Exception as e:
        logger.error(f"Failed to load dataframe for prediction: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    # Filter for vehicles
    df['vehicle_id'] = df['vehicle_id'].astype(str)
    target_ids = [str(vid) for vid in vehicle_ids]
    target_df = df[df['vehicle_id'].isin(target_ids)].copy()

    return target_df

async def predict_eta(vehicle_ids: List[str]) -> Dict[str, datetime]:
    """
    Predict ETA (absolute datetime of arrival at next stop) for a list of vehicle IDs using LSTM.
    """
    target_df = await _get_vehicle_data(vehicle_ids)
    if target_df.empty:
        return {}

    # Group by vehicle and prepare batches
    sequence_length = 10
    input_columns = ['latitude', 'longitude', 'speed_kmh']

    batches: Dict[Tuple[str, int], List[Tuple[str, np.ndarray, datetime]]] = {}

    for vehicle_id, vehicle_df in target_df.groupby('vehicle_id'):
        vehicle_df = vehicle_df.sort_values('timestamp')
        if len(vehicle_df) < sequence_length:
            continue

        sequence_df = vehicle_df.tail(sequence_length)
        last_point = sequence_df.iloc[-1]

        route = last_point.get('route')
        polyline_idx = last_point.get('polyline_idx')

        if pd.isna(route) or pd.isna(polyline_idx):
            continue

        polyline_idx = int(polyline_idx)

        for col in input_columns:
            if col not in sequence_df.columns:
                sequence_df[col] = 0.0

        features = sequence_df[input_columns].fillna(0).values.astype(np.float32)

        # Get last timestamp for this vehicle
        last_ts = pd.to_datetime(last_point['timestamp']).to_pydatetime()
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        key = (route, polyline_idx)
        if key not in batches:
            batches[key] = []
        batches[key].append((vehicle_id, features, last_ts))

    results = {}

    for (route, idx), batch_items in batches.items():
        model = _MODEL_CACHE.get((route, idx))
        if not model:
            try:
                model = load_lstm_for_route(route, idx, input_columns=input_columns)
                _MODEL_CACHE[(route, idx)] = model
            except FileNotFoundError:
                continue
            except Exception as e:
                logger.error(f"Error loading LSTM model for {route} segment {idx}: {e}")
                continue

        vehicle_ids_in_batch = [item[0] for item in batch_items]
        X_batch = np.stack([item[1] for item in batch_items])
        timestamps = [item[2] for item in batch_items]

        try:
            predictions = model.predict(X_batch)
            for i, vid in enumerate(vehicle_ids_in_batch):
                eta_seconds = float(predictions[i][0])
                eta_seconds = max(0.0, eta_seconds)

                # Convert ETA seconds to absolute datetime
                eta_datetime = timestamps[i] + timedelta(seconds=eta_seconds)
                results[vid] = eta_datetime
        except Exception as e:
            logger.error(f"LSTM prediction failed for batch {route}/{idx}: {e}")

    return results

async def get_all_stop_times(vehicle_ids: List[str]) -> Dict[str, List[Tuple[str, datetime]]]:
    """
    Get all stop times for a list of vehicle IDs: historical, predicted, and future.

    For each vehicle, returns:
    1. Historical: Last time the vehicle was at each stop before its current destination
    2. Predicted: LSTM prediction for the stop it's currently going to
    3. Future: Average time predictions for subsequent stops

    Polyline indices map to stop transitions in routes_data['STOPS']:
    - polyline_idx 0: transition from STOPS[0] to STOPS[1]
    - polyline_idx 1: transition from STOPS[1] to STOPS[2]
    - etc.

    Args:
        vehicle_ids: List of vehicle IDs to predict for

    Returns:
        Dictionary mapping vehicle_id to list of (stop_key, datetime) tuples
        for all stops on the route (past, present, and future).
    """

    # Get full vehicle dataframe to find historical stop times and current position
    try:
        full_df = await get_today_dataframe()
    except Exception as e:
        logger.error(f"Failed to load dataframe for get_all_stop_times: {e}")
        return {}

    if full_df.empty:
        return {}

    # Load routes data from Stops class
    routes = Stops.routes_data

    # Get ML prediction for next stop (current transition)
    next_stop_etas = await predict_eta(vehicle_ids)
    if not next_stop_etas:
        return {}

    results: Dict[str, List[Tuple[str, datetime]]] = {}

    for vehicle_id in vehicle_ids:
        # Get all data for this vehicle
        vehicle_df = full_df[full_df['vehicle_id'] == str(vehicle_id)].sort_values('timestamp')
        if vehicle_df.empty:
            continue

        last_point = vehicle_df.iloc[-1]
        route = last_point.get('route')
        current_polyline_idx = last_point.get('polyline_idx')
        now_stop_key = last_point.get('stop_name')

        if pd.isna(route) or pd.isna(current_polyline_idx):
            continue

        current_polyline_idx = int(current_polyline_idx)

        # Check if route exists in routes data
        if route not in routes:
            logger.warning(f"Route {route} not found in routes.json")
            continue

        route_data = routes[route]
        if 'STOPS' not in route_data:
            logger.warning(f"STOPS not found for route {route}")
            continue

        stops = route_data['STOPS']  # List of stop keys like ["STUDENT_UNION", "COLONIE", ...]

        # Polyline index N represents transition from STOPS[N] to STOPS[N+1]
        # So if current_polyline_idx = 0, vehicle is going from STOPS[0] to STOPS[1]

        if not pd.isna(now_stop_key):
            now_stop_idx = stops.index(now_stop_key)
            if now_stop_idx == current_polyline_idx:
                # Vehicle is at the stop it's going to, increment to next
                current_polyline_idx += 1

        # Safety check
        if current_polyline_idx < 0 or current_polyline_idx >= len(stops):
            continue

        # Determine which stop we're heading to (destination of current transition)
        next_stop_idx = current_polyline_idx + 1

        if next_stop_idx >= len(stops):
            # Vehicle is past all stops (end of route)
            continue

        # Build complete list of stop times: historical + predicted + future
        stop_times: List[Tuple[str, datetime]] = []

        # 1. Add historical stop times (stops before the one we're going to)
        for i in range(next_stop_idx):
            stop_key = stops[i]

            # Find last time vehicle was at this stop (where stop_name matches)
            at_stop = vehicle_df[vehicle_df['stop_name'] == stop_key]
            if not at_stop.empty:
                last_at_stop = at_stop.iloc[-1]
                stop_timestamp = pd.to_datetime(last_at_stop['timestamp']).to_pydatetime()
                if stop_timestamp.tzinfo is None:
                    stop_timestamp = stop_timestamp.replace(tzinfo=timezone.utc)
                stop_times.append((stop_key, stop_timestamp))

        # 2. Add ML prediction for the stop we're currently going to
        next_stop_eta = next_stop_etas.get(vehicle_id)
        if not next_stop_eta:
            continue

        next_stop_key = stops[next_stop_idx]
        stop_times.append((next_stop_key, next_stop_eta))

        # 3. Calculate ETAs for subsequent stops using average travel times
        cumulative_eta = next_stop_eta

        for i in range(next_stop_idx + 1, len(stops)):
            curr_stop_key = stops[i]
            # Polyline index for transition TO this stop (from previous stop)
            polyline_idx_for_transition = i - 1

            # Load average travel time for this transition
            avg_time = await load_average_travel_time(route, polyline_idx_for_transition)

            if avg_time is None:
                logger.warning(f"Missing average travel time for {route} polyline {polyline_idx_for_transition}")
                # Skip remaining stops if we don't have complete data
                break

            # Calculate ETA for this stop
            cumulative_eta = cumulative_eta + timedelta(seconds=avg_time)
            stop_times.append((curr_stop_key, cumulative_eta))

        if stop_times:
            results[vehicle_id] = stop_times

    return results

async def predict_next_state(vehicle_ids: List[str]) -> Dict[str, Dict]:
    """
    Predict next state (speed, timestamp) for vehicles using ARIMA.
    """
    target_df = await _get_vehicle_data(vehicle_ids)
    if target_df.empty:
        return {}

    results = {}

    # Load default ARIMA params (p=3, d=0, q=2)
    try:
        # Assuming we trained on 'speed_kmh'
        arima_params = load_arima(p=3, d=0, q=2, value_column='speed_kmh')
    except FileNotFoundError:
        # Silently fail if no model, likely not trained yet
        return {}
    except Exception as e:
        logger.error(f"Error loading ARIMA params: {e}")
        return {}

    for vehicle_id, vehicle_df in target_df.groupby('vehicle_id'):
        vehicle_df = vehicle_df.sort_values('timestamp')
        if len(vehicle_df) < 5:
            continue

        if 'speed_kmh' not in vehicle_df.columns:
            continue

        speeds = vehicle_df['speed_kmh'].fillna(0).values

        try:
            # Fit ARIMA on recent history (max 200 points) using warm start
            recent_speeds = speeds[-200:]

            if len(recent_speeds) < P and len(recent_speeds) < Q + 1:
                # return average speed if not enough data
                predicted_speed = np.array([float(np.mean(recent_speeds))])
            else:
                model = fit_arima(recent_speeds, p=P, d=D, q=Q, start_params=arima_params['params'])

                # Predict next speed
                forecast = model.predict(n_periods=1)
                predicted_speed = float(forecast[0])
                predicted_speed = max(0.0, predicted_speed)

            # Predict next timestamp
            timestamps = vehicle_df['timestamp'].values
            last_ts = pd.to_datetime(timestamps[-1]).to_pydatetime()
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)

            if len(timestamps) >= 2:
                deltas = np.diff(timestamps[-5:]).astype('timedelta64[s]').astype(int)
                avg_delta = np.mean(deltas)
                avg_delta = min(max(avg_delta, 1), 60) # Clamp between 1s and 60s
            else:
                avg_delta = 5

            predicted_ts = last_ts + timedelta(seconds=float(avg_delta))

            results[vehicle_id] = {
                'speed_kmh': predicted_speed,
                'timestamp': predicted_ts
            }

        except Exception as e:
            # logger.warning(f"ARIMA prediction failed for {vehicle_id}: {e}")
            continue

    return results

async def save_predictions(etas: Dict[str, List[Tuple[str, datetime]]], next_states: Dict[str, Dict]):
    """Save predictions to database.

    Args:
        etas: Dictionary mapping vehicle_id to list of (stop_key, eta_datetime) tuples
        next_states: Dictionary mapping vehicle_id to predicted next state (speed, timestamp)
    """
    if not etas and not next_states:
        return

    engine = create_async_db_engine(settings.DATABASE_URL, echo=False)
    session_factory = create_session_factory(engine)

    async with session_factory() as session:
        # Save ETAs (as ISO format strings for JSON storage)
        for vid, stop_etas in etas.items():
            # Convert list of (stop_key, datetime) tuples to dict
            etas_dict = {stop_key: eta_datetime.isoformat() for stop_key, eta_datetime in stop_etas}

            new_eta = ETA(
                vehicle_id=vid,
                etas=etas_dict,
                timestamp=datetime.now(timezone.utc)
            )
            session.add(new_eta)

        # Save PredictedLocations
        for vid, state in next_states.items():
            new_loc = PredictedLocation(
                vehicle_id=vid,
                speed_kmh=state['speed_kmh'],
                timestamp=state['timestamp']
            )
            session.add(new_loc)

        await session.commit()
        await soft_clear_namespace("etas")
        await soft_clear_namespace("velocities")

    await engine.dispose()

async def generate_and_save_predictions(vehicle_ids: List[str]):
    """Generate ETAs and next states, then save to DB."""
    if not vehicle_ids:
        return

    logger.info(f"Generating predictions for {len(vehicle_ids)} vehicles...")

    # Run in parallel
    # Note: get_all_stop_times and predict_next_state both call get_today_dataframe.
    # get_today_dataframe is cached in Redis so overhead is low.
    results = await asyncio.gather(
        get_all_stop_times(vehicle_ids),
        predict_next_state(vehicle_ids)
    )

    etas = results[0]
    next_states = results[1]

    await save_predictions(etas, next_states)

    count_etas = len(etas)
    count_locs = len(next_states)
    if count_etas > 0 or count_locs > 0:
        logger.info(f"Saved {count_etas} ETAs and {count_locs} predicted locations")
