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
from backend.time_utils import dev_now
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
    input_columns = ['latitude', 'longitude', 'speed_kmh', 'dist_to_end']

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

                # Clamp to reasonable max (30 min)
                if eta_seconds > 1800:
                    logger.warning(f"LSTM predicted {eta_seconds:.0f}s for vehicle {vid}, skipping (too large)")
                    continue

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

        # If last point has no route (ambiguous location like shared Student Union),
        # look backwards through the vehicle's history to find the last known route
        if pd.isna(route) or pd.isna(current_polyline_idx):
            valid_points = vehicle_df.dropna(subset=['route', 'polyline_idx'])
            if valid_points.empty:
                continue
            fallback_point = valid_points.iloc[-1]
            route = fallback_point.get('route')
            current_polyline_idx = fallback_point.get('polyline_idx')
            if pd.isna(now_stop_key):
                now_stop_key = fallback_point.get('stop_name')
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

        # Use the vehicle's last detected stop to validate/correct polyline_idx.
        # This resolves ambiguity at polyline intersections (e.g., the outbound
        # segment from Student Union crosses the return segment near Houston Field
        # House). The geometric polyline_idx can be wrong at these intersections.
        stop_points = vehicle_df.dropna(subset=['stop_name'])
        if not stop_points.empty:
            last_stop = str(stop_points.iloc[-1].get('stop_name'))
            if last_stop in stops:
                last_stop_idx = stops.index(last_stop)
                half_route = len(stops) // 2
                # Detect loop restart: geometric near start, last stop near end
                # This means the shuttle completed a loop and is starting fresh
                is_loop_restart = (current_polyline_idx < half_route and
                                   last_stop_idx >= half_route)
                if not is_loop_restart:
                    # Only override if geometric is behind the stop
                    # (intersection confusion causing wrong segment)
                    if current_polyline_idx < last_stop_idx:
                        current_polyline_idx = last_stop_idx

        # Map polyline_idx (from POLYLINE_STOPS, which includes ghost stops) to
        # the next STOPS entry. POLYLINE_STOPS[polyline_idx+1] is the destination
        # of the current polyline segment — find that in STOPS.
        polyline_stops = route_data.get('POLYLINE_STOPS', stops)
        poly_dest_idx = current_polyline_idx + 1
        if poly_dest_idx < len(polyline_stops):
            poly_dest_stop = polyline_stops[poly_dest_idx]
            # Find next real stop at or after the polyline destination
            next_stop_idx = None
            for si in range(len(stops)):
                if stops[si] == poly_dest_stop:
                    next_stop_idx = si
                    break
                # If poly_dest is a ghost stop, find the next real stop after it
                if si > 0 and polyline_stops.index(stops[si]) > poly_dest_idx if stops[si] in polyline_stops else False:
                    next_stop_idx = si
                    break
            if next_stop_idx is None:
                # Fallback: find next stop in POLYLINE_STOPS that's also in STOPS
                for pi in range(poly_dest_idx, len(polyline_stops)):
                    if polyline_stops[pi] in stops:
                        next_stop_idx = stops.index(polyline_stops[pi])
                        break
        else:
            next_stop_idx = None

        # If vehicle is AT the next stop, advance to the one after
        if not pd.isna(now_stop_key) and str(now_stop_key) in stops:
            now_stop_idx = stops.index(str(now_stop_key))
            if next_stop_idx is not None and now_stop_idx == next_stop_idx:
                next_stop_idx = now_stop_idx + 1

        if next_stop_idx is None or next_stop_idx >= len(stops):
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
    now_utc = dev_now(timezone.utc)
    best_per_route: Dict[str, Tuple[str, datetime, List[Tuple[str, datetime]]]] = {}
    # route -> (vehicle_id, first_future_eta, stops_list)

    for vid, vdata in vehicle_stop_etas.items():
        route = vdata["route"]
        stops_list = vdata["stops"]

        # Find the first future ETA in this vehicle's stop list.
        # The next_stop_eta may be in the past if the shuttle just passed its
        # next stop and the polyline_idx hasn't advanced yet — but subsequent
        # stops (computed via OFFSET diffs) will still have future ETAs.
        first_future_eta = None
        for _, eta_dt in stops_list:
            if eta_dt > now_utc:
                first_future_eta = eta_dt
                break

        if first_future_eta is None:
            continue

        if route not in best_per_route or first_future_eta < best_per_route[route][1]:
            best_per_route[route] = (vid, first_future_eta, stops_list)

    # Load previous cycle's ETAs from Redis for instant "Last:" updates.
    # When a stop's predicted ETA is now in the past, use it as the implied
    # arrival time (shuttle was predicted to arrive at that time → it did).
    previous_etas: Dict[str, str] = {}
    try:
        redis = get_redis()
        if redis:
            import json as _json
            raw = await redis.get("shubble:per_stop_etas_live")
            if raw:
                prev_data = _json.loads(raw)
                for k, v in prev_data.items():
                    if ':' in k or not isinstance(v, dict):
                        continue
                    eta = v.get('eta')
                    if eta:
                        previous_etas[k] = eta
    except Exception as e:
        logger.debug(f"Could not load previous ETAs: {e}")

    # Compute last_arrival per vehicle, using each vehicle's current loop window.
    # For each vehicle, the current loop started at its most recent detection
    # at the FIRST stop of its route. Only include detections AFTER that time
    # to avoid showing stale timestamps from previous loops.
    last_arrivals: Dict[str, str] = {}

    # First pass: use expired ETAs from previous cycle as implied arrivals.
    # If a stop had a future ETA last cycle but doesn't now (shuttle passed it),
    # the expired ETA IS effectively the arrival time.
    stops_with_future_eta_set: set = set()
    for _route, (_vid, _eta, stops_list) in best_per_route.items():
        for stop_key, eta_dt in stops_list:
            if eta_dt > now_utc:
                stops_with_future_eta_set.add(stop_key)

    for stop_key, prev_eta in previous_etas.items():
        if stop_key in stops_with_future_eta_set:
            continue  # Still ahead in current cycle
        try:
            prev_eta_dt = datetime.fromisoformat(prev_eta)
            # Only use recent past ETAs (within last 2 minutes)
            if (now_utc - prev_eta_dt).total_seconds() > 120:
                continue
            if prev_eta_dt <= now_utc:
                last_arrivals[stop_key] = prev_eta
        except (ValueError, TypeError):
            continue

    if full_df is not None and not full_df.empty and 'stop_name' in full_df.columns:
        tracked_vids = [str(v) for v in vehicle_ids]
        for vid in tracked_vids:
            vdf = full_df[full_df['vehicle_id'] == vid].dropna(subset=['stop_name'])
            if vdf.empty:
                continue

            # Find the vehicle's current route (from its stop_route or route column)
            route_col = vdf['stop_route'] if 'stop_route' in vdf.columns else vdf.get('route')
            vehicle_route = None
            if route_col is not None:
                non_nan = route_col.dropna()
                if not non_nan.empty:
                    vehicle_route = str(non_nan.iloc[-1])

            # Determine loop start: the most recent detection at the route's first stop
            loop_start_ts = None
            if vehicle_route and vehicle_route in routes:
                first_stop = routes[vehicle_route]['STOPS'][0]
                first_stop_detections = vdf[vdf['stop_name'] == first_stop]
                if not first_stop_detections.empty:
                    loop_start = pd.to_datetime(first_stop_detections['timestamp'].max()).to_pydatetime()
                    if loop_start.tzinfo is None:
                        loop_start = loop_start.replace(tzinfo=timezone.utc)
                    loop_start_ts = loop_start

            # For each stop, include only detections from the current loop
            for stop_key, group in vdf.groupby('stop_name'):
                latest = pd.to_datetime(group['timestamp'].max()).to_pydatetime()
                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=timezone.utc)
                # Skip if older than loop start (previous loop data)
                if loop_start_ts and latest < loop_start_ts:
                    continue
                key = str(stop_key)
                # Keep the most recent across vehicles
                if key not in last_arrivals or latest.isoformat() > last_arrivals[key]:
                    last_arrivals[key] = latest.isoformat()

    # Infer last_arrival for stops the shuttle must have passed through.
    # Only interpolate between stops with RECENT timestamps to avoid
    # false inferences from previous loop data.
    for route_name, route_data in routes.items():
        if 'STOPS' not in route_data:
            continue
        route_stops = route_data['STOPS']
        arrival_indices = [i for i, s in enumerate(route_stops) if s in last_arrivals]
        if len(arrival_indices) < 2:
            continue
        for gap_start_pos in range(len(arrival_indices) - 1):
            i_start = arrival_indices[gap_start_pos]
            i_end = arrival_indices[gap_start_pos + 1]
            if i_end - i_start <= 1:
                continue
            ts_start = datetime.fromisoformat(last_arrivals[route_stops[i_start]])
            ts_end = datetime.fromisoformat(last_arrivals[route_stops[i_end]])
            if ts_start >= ts_end:
                continue
            total_gap = i_end - i_start
            for j in range(i_start + 1, i_end):
                stop_key = route_stops[j]
                if stop_key in last_arrivals:
                    continue
                fraction = (j - i_start) / total_gap
                interp_ts = ts_start + (ts_end - ts_start) * fraction
                last_arrivals[stop_key] = interp_ts.isoformat()

    # Build final per-stop result from the best vehicle per route.
    # For stops that appear on multiple routes (e.g. STUDENT_UNION_RETURN),
    # keep each route's ETA separately so the frontend can display the correct
    # one for the selected route. Use "stop_key:route" as a secondary key.
    result: Dict[str, Dict] = {}
    # Track which stops have future ETAs (shuttle hasn't passed them yet)
    stops_with_future_eta: set = set()
    for route, (vid, _, stops_list) in best_per_route.items():
        for stop_key, eta_dt in stops_list:
            if eta_dt <= now_utc:
                continue
            stops_with_future_eta.add(stop_key)
            # Don't include last_arrival for stops with future ETAs —
            # the last_arrival might be from a previous loop
            entry = {
                "eta": eta_dt.isoformat(),
                "vehicle_id": vid,
                "route": route,
                "last_arrival": None,
            }
            # Primary key: bare stop name — keep earliest across routes
            if stop_key not in result or eta_dt < datetime.fromisoformat(result[stop_key]["eta"]):
                result[stop_key] = entry
            # Secondary key: route-qualified — ensures each route's ETA is preserved
            route_key = f"{stop_key}:{route}"
            result[route_key] = entry

    # Add last_arrival only for stops the shuttle has PASSED (no future ETA)
    for stop_key, last_iso in last_arrivals.items():
        if stop_key in stops_with_future_eta:
            continue  # Stop still ahead — don't show stale last_arrival
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
                timestamp=dev_now(timezone.utc)
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
async def compute_trips(vehicle_ids: List[str], df: Optional[pd.DataFrame] = None) -> List[Dict]:
    """Compute per-trip ETAs. Each trip is a (route, departure_time) pair
    assigned to a specific shuttle. Unlike compute_per_stop_etas, this
    does NOT merge shuttles on the same route — each trip is independent.
    """
    from backend.worker.trips import compute_trips_from_vehicle_data
    from backend.config import settings

    full_df = df
    if full_df is None:
        try:
            full_df = await get_today_dataframe()
        except Exception as e:
            logger.error(f"Failed to load dataframe for compute_trips: {e}")
            return []

    if full_df is None or full_df.empty:
        return []

    # Reuse compute_per_stop_etas internals by calling it to populate caches,
    # but we need the per-vehicle data. Call a helper that exposes it.
    vehicle_stop_etas, last_arrivals_by_vehicle = await _compute_vehicle_etas_and_arrivals(vehicle_ids, full_df)

    now_utc = dev_now(timezone.utc)
    trips = compute_trips_from_vehicle_data(
        vehicle_stop_etas=vehicle_stop_etas,
        last_arrivals_by_vehicle=last_arrivals_by_vehicle,
        full_df=full_df,
        routes_data=Stops.routes_data,
        vehicle_ids=vehicle_ids,
        now_utc=now_utc,
        campus_tz=settings.CAMPUS_TZ,
    )
    return trips


async def _compute_vehicle_etas_and_arrivals(
    vehicle_ids: List[str],
    full_df: pd.DataFrame,
) -> Tuple[Dict[str, Dict], Dict[str, Dict[str, str]]]:
    """Extract the per-vehicle ETAs and per-vehicle last_arrivals.

    Returns:
        (vehicle_stop_etas, last_arrivals_by_vehicle)
        where last_arrivals_by_vehicle is {vehicle_id: {stop_name: iso}}
        so concurrent shuttles don't cross-contaminate each other's
        "passed" state.
    """
    # Delegate to compute_per_stop_etas but recover the intermediate data
    # by re-running a lightweight version.
    # Simpler: just call compute_per_stop_etas to prime Redis, then read
    # vehicle stops directly from its result structure.
    # For now, we re-implement the per-vehicle loop inline.
    from backend.worker.velocity import get_velocity_predictor
    routes = Stops.routes_data
    velocity_predictor = get_velocity_predictor()
    next_stop_etas = await predict_eta(vehicle_ids, df=full_df)

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
            valid_points = vehicle_df.dropna(subset=['route', 'polyline_idx'])
            if valid_points.empty:
                continue
            fallback_point = valid_points.iloc[-1]
            route = fallback_point.get('route')
            current_polyline_idx = fallback_point.get('polyline_idx')
            if pd.isna(now_stop_key):
                now_stop_key = fallback_point.get('stop_name')
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

        # Stop-based polyline validation
        stop_points = vehicle_df.dropna(subset=['stop_name'])
        last_stop_idx = None
        polyline_was_reset = False
        if not stop_points.empty:
            last_stop = str(stop_points.iloc[-1].get('stop_name'))
            if last_stop in stops:
                last_stop_idx = stops.index(last_stop)
                half_route = len(stops) // 2
                # Loop-boundary reset: the polyline wraps near Student Union,
                # so the GPS closest-point match can return a polyline index
                # near the END of the route even when the shuttle is physically
                # at the START. If the last detected stop is the FIRST stop,
                # trust that over the polyline_idx and reset to 0.
                if last_stop_idx == 0:
                    if current_polyline_idx != 0:
                        polyline_was_reset = True
                    current_polyline_idx = 0
                else:
                    is_loop_restart = (current_polyline_idx < half_route and last_stop_idx >= half_route)
                    if not is_loop_restart and current_polyline_idx < last_stop_idx:
                        current_polyline_idx = last_stop_idx

        # POLYLINE_STOPS → STOPS mapping
        polyline_stops = route_data.get('POLYLINE_STOPS', stops)
        poly_dest_idx = current_polyline_idx + 1
        next_stop_idx = None
        if poly_dest_idx < len(polyline_stops):
            poly_dest_stop = polyline_stops[poly_dest_idx]
            if poly_dest_stop in stops:
                next_stop_idx = stops.index(poly_dest_stop)
            else:
                for pi in range(poly_dest_idx, len(polyline_stops)):
                    if polyline_stops[pi] in stops:
                        next_stop_idx = stops.index(polyline_stops[pi])
                        break

        # Stop-detection override: if the latest row has a detected stop,
        # trust that over the polyline-based next_stop. This catches cases
        # where polyline matching finds a geometrically-close but loop-stale
        # position (e.g. polyline intersection at loop boundary).
        if not pd.isna(now_stop_key) and str(now_stop_key) in stops:
            now_stop_idx = stops.index(str(now_stop_key))
            # Always treat the shuttle as "at now_stop", so the next stop is
            # the one immediately after. Guards against polyline matching
            # returning a position far ahead of where the shuttle actually is.
            next_stop_idx = now_stop_idx + 1
        elif last_stop_idx is not None and next_stop_idx is not None:
            # now_stop_key is NaN (between-stop GPS ping), but we have a last
            # detected stop. The polyline-based next_stop should be close to
            # last_stop_idx + 1. If it's too far ahead (> 2 stops forward),
            # clamp it down — polyline matching is likely wrong.
            if next_stop_idx > last_stop_idx + 2:
                next_stop_idx = last_stop_idx + 1

        if next_stop_idx is None or next_stop_idx >= len(stops):
            continue

        # Layer 1: next stop ETA from LSTM or fallback.
        # If we reset the polyline_idx (shuttle at Student Union starting
        # a new loop), skip the LSTM prediction — it was computed for
        # the OLD polyline_idx (near end of route) and would produce a
        # wrong ETA for the first stop of the new loop.
        next_stop_eta = None if polyline_was_reset else next_stop_etas.get(vehicle_id)
        if next_stop_eta is None:
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
                    dist_to_end = 0.1
                speed_kmh = velocity_predictor.predict_speed_kmh(str(vehicle_id), route)
                if speed_kmh <= 0:
                    speed_kmh = 20.0
                eta_seconds = (dist_to_end / speed_kmh) * 3600
            except Exception:
                continue
            last_ts = pd.to_datetime(last_point['timestamp']).to_pydatetime()
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            next_stop_eta = last_ts + timedelta(seconds=eta_seconds)

        next_stop_key = stops[next_stop_idx]
        next_stop_data = route_data.get(next_stop_key)
        if not next_stop_data or 'OFFSET' not in next_stop_data:
            continue
        next_stop_offset = next_stop_data['OFFSET']

        vehicle_stops: List[Tuple[str, datetime]] = [(next_stop_key, next_stop_eta)]
        for i in range(next_stop_idx + 1, len(stops)):
            stop_key = stops[i]
            sde = route_data.get(stop_key)
            if not sde or 'OFFSET' not in sde:
                continue
            offset_diff_sec = (sde['OFFSET'] - next_stop_offset) * 60
            vehicle_stops.append((stop_key, next_stop_eta + timedelta(seconds=offset_diff_sec)))

        vehicle_stop_etas[str(vehicle_id)] = {
            "route": route,
            "stops": vehicle_stops,
        }

    # Compute per-vehicle last_arrivals with loop-start cutoff. Keyed by
    # vehicle_id so trip A doesn't inherit trip B's "passed" stops.
    last_arrivals_by_vehicle: Dict[str, Dict[str, str]] = {}
    if 'stop_name' in full_df.columns:
        tracked_vids = [str(v) for v in vehicle_ids]
        for vid in tracked_vids:
            vdf = full_df[full_df['vehicle_id'] == vid].dropna(subset=['stop_name'])
            if vdf.empty:
                continue
            route_col = vdf['stop_route'] if 'stop_route' in vdf.columns else vdf.get('route')
            vehicle_route = None
            if route_col is not None:
                non_nan = route_col.dropna()
                if not non_nan.empty:
                    vehicle_route = str(non_nan.iloc[-1])
            loop_start_ts = None
            if vehicle_route and vehicle_route in routes:
                first_stop = routes[vehicle_route]['STOPS'][0]
                first_stop_detections = vdf[vdf['stop_name'] == first_stop]
                if not first_stop_detections.empty:
                    loop_start = pd.to_datetime(first_stop_detections['timestamp'].max()).to_pydatetime()
                    if loop_start.tzinfo is None:
                        loop_start = loop_start.replace(tzinfo=timezone.utc)
                    loop_start_ts = loop_start
            vehicle_las: Dict[str, str] = {}
            for stop_key, group in vdf.groupby('stop_name'):
                latest = pd.to_datetime(group['timestamp'].max()).to_pydatetime()
                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=timezone.utc)
                if loop_start_ts and latest < loop_start_ts:
                    continue
                vehicle_las[str(stop_key)] = latest.isoformat()
            if vehicle_las:
                last_arrivals_by_vehicle[vid] = vehicle_las

    return vehicle_stop_etas, last_arrivals_by_vehicle


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

    # Compute per-stop ETAs (legacy API) and per-trip ETAs (new API)
    per_stop_etas = await compute_per_stop_etas(vehicle_ids, df=df)
    trips = await compute_trips(vehicle_ids, df=df)

    # Store both in Redis
    redis = get_redis()
    if redis:
        import json as _json
        if per_stop_etas:
            await redis.set(
                "shubble:per_stop_etas_live",
                _json.dumps(per_stop_etas).encode(),
                ex=120,
            )
        if trips:
            await redis.set(
                "shubble:trips_live",
                _json.dumps(trips).encode(),
                ex=120,
            )

    # Predict next state separately (for /api/velocities endpoint)
    next_states = await predict_next_state(vehicle_ids, df=df)

    await save_predictions(per_stop_etas, next_states)

    count_stops = len(per_stop_etas)
    count_locs = len(next_states)
    if count_stops > 0 or count_locs > 0:
        logger.info(f"Saved ETAs for {count_stops} stops and {count_locs} predicted locations")
