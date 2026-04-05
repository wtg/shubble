"""Data prediction utilities for worker."""
import asyncio
import logging
from contextlib import aclosing
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta

from backend.cache_dataframe import get_today_dataframe
from ml.deploy.lstm import load_lstm_for_route
from ml.deploy.arima import load_arima
from ml.training.train import fit_arima
from backend.models import ETA, PredictedLocation
from backend.database import get_db
from backend.cache import cache, soft_clear_namespace, get_redis
from backend.function_timer import timed
from shared.stops import Stops

logger = logging.getLogger(__name__)

P = 3
D = 0
Q = 2

# Cache for loaded models: (route_name, polyline_idx) -> LSTMModel | None
# None sentinel means "load was attempted and failed — don't retry"
_MODEL_CACHE: Dict[Tuple[str, int], Any] = {}

@timed
async def _get_vehicle_data(vehicle_ids: List[str], df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Helper to get and filter vehicle data."""
    if df is None:
        try:
            df = await get_today_dataframe()
        except Exception as e:
            logger.error(f"Failed to load dataframe for prediction: {e}")
            return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Filter for vehicles
    df['vehicle_id'] = df['vehicle_id'].astype(str)
    target_ids = [str(vid) for vid in vehicle_ids]
    target_df = df[df['vehicle_id'].isin(target_ids)].copy()

    return target_df

@timed
async def predict_eta(vehicle_ids: List[str], df: Optional[pd.DataFrame] = None) -> Dict[str, datetime]:
    """
    Predict ETA (absolute datetime of arrival at next stop) for a list of vehicle IDs using LSTM.
    """
    target_df = await _get_vehicle_data(vehicle_ids, df=df)
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
        key = (route, idx)
        if key in _MODEL_CACHE:
            model = _MODEL_CACHE[key]
            if model is None:
                continue  # Previously failed — skip
        else:
            try:
                model = load_lstm_for_route(route, idx, input_columns=input_columns)
                _MODEL_CACHE[key] = model
            except FileNotFoundError:
                _MODEL_CACHE[key] = None  # Cache the miss
                continue
            except Exception as e:
                logger.error(f"Error loading LSTM model for {route} segment {idx}: {e}")
                _MODEL_CACHE[key] = None  # Cache the failure
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

@timed
async def compute_per_stop_etas(vehicle_ids: List[str], df: Optional[pd.DataFrame] = None) -> Dict[str, Dict]:
    """
    Compute per-stop ETAs: for each stop, the next shuttle to arrive.

    Three-layer approach:
    1. LSTM predicts time to the immediate next stop (fallback: distance / avg speed)
    2. Static OFFSET diffs from routes.json for all subsequent stops
    3. Historical arrivals for stops already passed

    Args:
        vehicle_ids: List of vehicle IDs to predict for
        df: Optional pre-loaded dataframe

    Returns:
        Per-stop dictionary:
        {
            "COLONIE": {
                "eta": "2026-04-03T14:33:00+00:00",
                "vehicle_id": "123",
                "route": "NORTH"
            },
            ...
        }
    """
    from backend.worker.velocity import get_velocity_predictor

    full_df = df
    if full_df is None:
        try:
            full_df = await get_today_dataframe()
        except Exception as e:
            logger.error(f"Failed to load dataframe for compute_per_stop_etas: {e}")
            return {}

    if full_df is None or full_df.empty:
        return {}

    routes = Stops.routes_data
    velocity_predictor = get_velocity_predictor()

    # Get LSTM predictions for next stop (best effort — may be empty)
    next_stop_etas = await predict_eta(vehicle_ids, df=full_df)

    # Per-vehicle: compute next_stop_eta and all subsequent stop ETAs
    # Structure: {vehicle_id: {route, next_stop_eta, stops: [(stop_key, eta)]}}
    vehicle_stop_etas: Dict[str, Dict] = {}

    for vehicle_id in vehicle_ids:
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
        route = str(route)

        if route not in routes:
            continue

        route_data = routes[route]
        if 'STOPS' not in route_data:
            continue

        stops = route_data['STOPS']

        # Polyline index N = transition from STOPS[N] to STOPS[N+1]
        # If vehicle is AT the destination stop, advance to next transition
        if not pd.isna(now_stop_key) and str(now_stop_key) in stops:
            now_stop_idx = stops.index(str(now_stop_key))
            if now_stop_idx == current_polyline_idx + 1:
                current_polyline_idx += 1

        # Next stop is the destination of current transition
        next_stop_idx = current_polyline_idx + 1
        if next_stop_idx >= len(stops):
            continue

        # --- Layer 1: ETA to next stop ---
        next_stop_eta = next_stop_etas.get(vehicle_id)

        if next_stop_eta is None:
            # Fallback: distance / average speed
            last_lat = last_point.get('latitude')
            last_lon = last_point.get('longitude')
            if pd.isna(last_lat) or pd.isna(last_lon):
                continue

            try:
                closest = Stops.get_closest_point(
                    (float(last_lat), float(last_lon)),
                    target_polyline=(route, current_polyline_idx)
                )
                if closest is None:
                    continue
                _, dist_to_end, _ = Stops.get_polyline_distances(
                    (float(last_lat), float(last_lon)),
                    closest_point_result=closest
                )
                if dist_to_end is None or dist_to_end <= 0:
                    dist_to_end = 0.1  # minimum 100m

                speed_kmh = velocity_predictor.predict_speed_kmh(str(vehicle_id), route)
                if speed_kmh <= 0:
                    speed_kmh = 20.0
                eta_seconds = (dist_to_end / speed_kmh) * 3600
            except Exception as e:
                logger.warning(f"Fallback ETA failed for vehicle {vehicle_id}: {e}")
                continue

            last_ts = pd.to_datetime(last_point['timestamp']).to_pydatetime()
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            next_stop_eta = last_ts + timedelta(seconds=eta_seconds)

        next_stop_key = stops[next_stop_idx]
        next_stop_data = route_data.get(next_stop_key)
        if not next_stop_data or 'OFFSET' not in next_stop_data:
            logger.warning(f"Missing OFFSET for stop {next_stop_key} on route {route}")
            continue
        next_stop_offset = next_stop_data['OFFSET']

        # Build this vehicle's complete stop ETA list (preserves ordering)
        vehicle_stops: List[Tuple[str, datetime]] = []
        vehicle_stops.append((next_stop_key, next_stop_eta))

        # --- Layer 2: Subsequent stops via OFFSET diffs ---
        for i in range(next_stop_idx + 1, len(stops)):
            stop_key = stops[i]
            stop_data_entry = route_data.get(stop_key)
            if not stop_data_entry or 'OFFSET' not in stop_data_entry:
                continue
            stop_offset = stop_data_entry['OFFSET']
            offset_diff_seconds = (stop_offset - next_stop_offset) * 60
            stop_eta = next_stop_eta + timedelta(seconds=offset_diff_seconds)
            vehicle_stops.append((stop_key, stop_eta))

        vehicle_stop_etas[str(vehicle_id)] = {
            "route": route,
            "next_stop_eta": next_stop_eta,
            "stops": vehicle_stops,
        }

    # Aggregate per-route: for each route, pick the vehicle arriving soonest
    # at its next stop, then use ALL of that vehicle's ETAs for that route.
    # This preserves stop ordering (no impossible ETAs).
    now_utc = datetime.now(timezone.utc)
    best_per_route: Dict[str, Tuple[str, datetime, List[Tuple[str, datetime]]]] = {}
    # route -> (vehicle_id, next_stop_eta, stops_list)

    for vid, vdata in vehicle_stop_etas.items():
        route = vdata["route"]
        next_eta = vdata["next_stop_eta"]
        if next_eta <= now_utc:
            continue

        if route not in best_per_route or next_eta < best_per_route[route][1]:
            best_per_route[route] = (vid, next_eta, vdata["stops"])

    # Compute last_arrival for each stop from today's data
    last_arrivals: Dict[str, str] = {}
    if full_df is not None and not full_df.empty and 'stop_name' in full_df.columns:
        stops_df = full_df.dropna(subset=['stop_name'])
        if not stops_df.empty:
            for stop_key, group in stops_df.groupby('stop_name'):
                latest = group['timestamp'].max()
                ts = pd.to_datetime(latest).to_pydatetime()
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                last_arrivals[str(stop_key)] = ts.isoformat()

    # Build final per-stop result from the best vehicle per route
    result: Dict[str, Dict] = {}
    for route, (vid, _, stops_list) in best_per_route.items():
        for stop_key, eta_dt in stops_list:
            if eta_dt <= now_utc:
                continue
            entry = {
                "eta": eta_dt.isoformat(),
                "vehicle_id": vid,
                "route": route,
                "last_arrival": last_arrivals.get(stop_key),
            }
            # For stops shared across routes (e.g. STUDENT_UNION), keep earliest
            if stop_key not in result or eta_dt < datetime.fromisoformat(result[stop_key]["eta"]):
                result[stop_key] = entry

    # Add last_arrival for stops that have no future ETA
    for stop_key, last_iso in last_arrivals.items():
        if stop_key not in result:
            result[stop_key] = {
                "eta": None,
                "vehicle_id": None,
                "route": "",
                "last_arrival": last_iso,
            }

    return result

@timed
async def predict_next_state(vehicle_ids: List[str], df: Optional[pd.DataFrame] = None) -> Dict[str, Dict]:
    """
    Predict next state (speed, timestamp) for vehicles using ARIMA.
    """
    target_df = await _get_vehicle_data(vehicle_ids, df=df)
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

@timed
async def save_predictions(per_stop_etas: Dict[str, Dict], next_states: Dict[str, Dict]):
    """Save predictions to database.

    Args:
        per_stop_etas: Per-stop ETA dict from compute_per_stop_etas().
            Each value has "eta", "vehicle_id", "route".
            Saved as one ETA row per vehicle with its contributed stops.
        next_states: Dictionary mapping vehicle_id to predicted next state (speed, timestamp)
    """
    if not per_stop_etas and not next_states:
        return

    async with aclosing(get_db()) as gen:
        session = await anext(gen)

        # Group per-stop ETAs back by vehicle_id for DB storage
        # Skip last_arrival-only entries (vehicle_id is None)
        vehicle_etas: Dict[str, Dict[str, Any]] = {}
        for stop_key, stop_info in per_stop_etas.items():
            vid = stop_info.get("vehicle_id")
            if vid is None:
                continue  # last_arrival-only entry, stored via Redis
            if vid not in vehicle_etas:
                vehicle_etas[vid] = {}
            vehicle_etas[vid][stop_key] = {
                "eta": stop_info["eta"],
                "route": stop_info["route"],
                "last_arrival": stop_info.get("last_arrival"),
            }

        for vid, etas_dict in vehicle_etas.items():
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

@timed
async def generate_and_save_predictions(vehicle_ids: List[str]):
    """Generate ETAs and next states, then save to DB."""
    if not vehicle_ids:
        return

    logger.info(f"Generating predictions for {len(vehicle_ids)} vehicles...")

    # Load dataframe once to avoid repeated overhead
    try:
        df = await get_today_dataframe()
    except Exception as e:
        logger.error(f"Failed to load dataframe for predictions: {e}")
        return

    # Compute per-stop ETAs (uses LSTM for next stop, OFFSET diffs for rest)
    per_stop_etas = await compute_per_stop_etas(vehicle_ids, df=df)

    # Store complete per-stop result in Redis for direct API access
    # (preserves last_arrival-only entries that don't survive the DB round-trip)
    redis = get_redis()
    if redis and per_stop_etas:
        import json as _json
        await redis.set(
            "shubble:per_stop_etas_live",
            _json.dumps(per_stop_etas).encode(),
            ex=120,
        )

    # Predict next state separately (for /api/velocities endpoint)
    next_states = await predict_next_state(vehicle_ids, df=df)

    await save_predictions(per_stop_etas, next_states)

    count_stops = len(per_stop_etas)
    count_locs = len(next_states)
    if count_stops > 0 or count_locs > 0:
        logger.info(f"Saved ETAs for {count_stops} stops and {count_locs} predicted locations")
