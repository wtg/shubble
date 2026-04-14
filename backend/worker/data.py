"""Data prediction utilities for worker."""
import asyncio
import json
import logging
from contextlib import aclosing
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone, timedelta

from backend.cache_dataframe import get_today_dataframe
from backend.config import settings
from ml.deploy.lstm import load_lstm_for_route, list_cached_models
from ml.deploy.arima import load_arima
from ml.training.train import fit_arima
from backend.models import ETA, PredictedLocation
from backend.database import get_db
from backend.cache import soft_clear_namespace, get_redis
from backend.function_timer import timed
from backend.time_utils import dev_now
from backend.worker.velocity import get_velocity_predictor
from shared.stops import Stops, haversine

logger = logging.getLogger(__name__)

P = 3
D = 0
Q = 2

# Input columns used by predict_eta. Must match the signature the LSTM models
# on disk were trained with — preloading uses this exact list.
_LSTM_INPUT_COLUMNS = ['latitude', 'longitude', 'speed_kmh', 'dist_to_end']

# Cache for loaded models: (route_name, polyline_idx) -> LSTMModel | None
# None sentinel means "load was attempted and failed — don't retry"
_MODEL_CACHE: Dict[Tuple[str, int], Any] = {}


def preload_lstm_models() -> int:
    """
    Eagerly load every LSTM model on disk into _MODEL_CACHE.

    Called once at worker startup so the first prediction cycle doesn't pay
    the ~100-300ms cold-load cost per (route, polyline_idx) pair. Failures
    are tolerated and cached as None so predict_eta() skips them later.

    Returns:
        Number of models successfully loaded.
    """
    loaded = 0
    skipped = 0
    for meta in list_cached_models():
        route = meta['route_name']
        idx = meta['polyline_idx']
        key = (route, idx)
        if key in _MODEL_CACHE:
            continue
        try:
            _MODEL_CACHE[key] = load_lstm_for_route(
                route, idx, input_columns=_LSTM_INPUT_COLUMNS
            )
            loaded += 1
        except FileNotFoundError:
            _MODEL_CACHE[key] = None
            skipped += 1
        except Exception as e:
            logger.error(f"Error preloading LSTM model for {route} segment {idx}: {e}")
            _MODEL_CACHE[key] = None
            skipped += 1
    logger.info(f"LSTM preload complete: {loaded} loaded, {skipped} skipped")
    return loaded

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

    # PERF: no longer mutating the shared df. The vehicle_id column is
    # produced as str by the Samsara ingest + ML pipeline, so we can just
    # compare as-is. If a caller ever passes a df with non-str ids, isin()
    # against a str set will miss — which is fine, it just returns empty.
    target_ids = {str(vid) for vid in vehicle_ids}
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
    input_columns = _LSTM_INPUT_COLUMNS

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

    # PERF: vectorize the per-vehicle slicing. The prior per-iteration
    # `full_df[full_df['vehicle_id'] == str(vid)].sort_values('timestamp')`
    # did N full-df scans; a single sort + groupby gives us O(1) lookup per
    # vehicle and the groups are already timestamp-sorted thanks to the
    # stable sort upstream.
    target_ids = {str(v) for v in vehicle_ids}
    work_df = full_df[full_df['vehicle_id'].astype(str).isin(target_ids)]
    if work_df.empty:
        return {}
    work_df = work_df.sort_values('timestamp', kind='mergesort')
    vehicle_groups = {
        str(vid): grp for vid, grp in work_df.groupby('vehicle_id', sort=False)
    }

    for vehicle_id in vehicle_ids:
        vehicle_df = vehicle_groups.get(str(vehicle_id))
        if vehicle_df is None or vehicle_df.empty:
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

        # If vehicle is AT or PAST the next stop, advance to the one after.
        # `>=` (not ==) handles the case where polyline_idx lags two or more
        # stops behind the true position: without the broader check, the
        # advance was skipped and the predictor kept picking the stale
        # next_stop for multiple ticks.
        if not pd.isna(now_stop_key) and str(now_stop_key) in stops:
            now_stop_idx = stops.index(str(now_stop_key))
            if next_stop_idx is not None and now_stop_idx >= next_stop_idx:
                next_stop_idx = now_stop_idx + 1

        # Dwelling-at-Union: the vehicle's latest stop_name is the LAST stop
        # in STOPS (e.g. STUDENT_UNION_RETURN) AND that stop's coordinates
        # coincide with the FIRST stop's coordinates (loop-boundary stop
        # like STUDENT_UNION ≡ STUDENT_UNION_RETURN on NORTH/WEST). In that
        # case the shuttle is physically at the boundary between loops —
        # emit ETAs starting from stop index 1 so the dwell-promote logic
        # in compute_trips_from_vehicle_data can match it to the next
        # scheduled slot instead of dropping the vehicle entirely.
        if (
            not pd.isna(now_stop_key)
            and str(now_stop_key) in stops
            and stops.index(str(now_stop_key)) == len(stops) - 1
            and len(stops) >= 2
        ):
            first_stop_data = route_data.get(stops[0])
            last_stop_data = route_data.get(stops[-1])
            if (
                isinstance(first_stop_data, dict)
                and isinstance(last_stop_data, dict)
                and first_stop_data.get("COORDINATES") is not None
                and last_stop_data.get("COORDINATES") is not None
                and tuple(first_stop_data["COORDINATES"]) == tuple(last_stop_data["COORDINATES"])
            ):
                next_stop_idx = 1

        if next_stop_idx is None or next_stop_idx >= len(stops):
            continue

        # --- Layer 1: ETA to next stop ---
        # Always compute the physics-based fallback first so we can
        # sanity-check the LSTM prediction against it. The LSTM has
        # been observed to return predictions ~10x the actual travel
        # time (e.g. 13 min for a 587m distance), usually when the
        # input features contain noisy stationary-shuttle speed
        # values or cross-loop confusion. A simple distance/speed
        # estimate is more trustworthy in those cases.
        last_lat = last_point.get('latitude')
        last_lon = last_point.get('longitude')
        fallback_seconds: Optional[float] = None
        if not (pd.isna(last_lat) or pd.isna(last_lon)):
            try:
                closest = Stops.get_closest_point(
                    (float(last_lat), float(last_lon)),
                    target_polyline=(route, current_polyline_idx)
                )
                if closest is not None:
                    _, dist_to_end, _ = Stops.get_polyline_distances(
                        (float(last_lat), float(last_lon)),
                        closest_point_result=closest
                    )
                    if dist_to_end is not None and dist_to_end > 0:
                        speed_kmh = velocity_predictor.predict_speed_kmh(
                            str(vehicle_id), route
                        )
                        if speed_kmh <= 0:
                            speed_kmh = 20.0
                        fallback_seconds = (dist_to_end / speed_kmh) * 3600
            except Exception as e:
                logger.debug(f"Fallback ETA estimate failed for {vehicle_id}: {e}")

        next_stop_eta = next_stop_etas.get(vehicle_id)

        # Sanity check: if the LSTM prediction is more than 3x the
        # physics estimate AND the fallback is a reasonable length,
        # the LSTM is probably confused. Prefer the fallback.
        if next_stop_eta is not None and fallback_seconds is not None:
            last_ts_for_check = pd.to_datetime(last_point['timestamp']).to_pydatetime()
            if last_ts_for_check.tzinfo is None:
                last_ts_for_check = last_ts_for_check.replace(tzinfo=timezone.utc)
            lstm_seconds = (next_stop_eta - last_ts_for_check).total_seconds()
            if (
                lstm_seconds > 60  # only sanity-check non-trivial ETAs
                and fallback_seconds > 0
                and lstm_seconds > fallback_seconds * 3
            ):
                logger.debug(
                    f"Vehicle {vehicle_id}: LSTM predicted {lstm_seconds:.0f}s vs "
                    f"fallback {fallback_seconds:.0f}s — using fallback"
                )
                next_stop_eta = None  # force fallback path below

        if next_stop_eta is None:
            if fallback_seconds is None:
                continue
            last_ts = pd.to_datetime(last_point['timestamp']).to_pydatetime()
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            next_stop_eta = last_ts + timedelta(seconds=fallback_seconds)

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
            raw = await redis.get("shubble:per_stop_etas_live")
            if raw:
                prev_data = json.loads(raw)
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
        # PERF: single filter + groupby instead of N full-df boolean masks.
        tracked_vids = {str(v) for v in vehicle_ids}
        stopped_df = full_df[
            full_df['vehicle_id'].astype(str).isin(tracked_vids)
        ].dropna(subset=['stop_name'])
        vdf_by_vid = {
            str(vid): grp for vid, grp in stopped_df.groupby('vehicle_id', sort=False)
        }
        for vid in tracked_vids:
            vdf = vdf_by_vid.get(vid)
            if vdf is None or vdf.empty:
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

            # PERF: vectorized max-per-stop instead of per-group python loop.
            # For each stop, include only detections from the current loop.
            stop_maxes = vdf.groupby('stop_name', sort=False)['timestamp'].max()
            for stop_key, latest_raw in stop_maxes.items():
                if pd.isna(latest_raw):
                    continue
                latest = (
                    latest_raw.to_pydatetime()
                    if hasattr(latest_raw, 'to_pydatetime')
                    else pd.Timestamp(latest_raw).to_pydatetime()
                )
                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=timezone.utc)
                # Skip if older than loop start (previous loop data)
                if loop_start_ts and latest < loop_start_ts:
                    continue
                key = str(stop_key)
                # Keep the most recent across vehicles
                latest_iso = latest.isoformat()
                if key not in last_arrivals or latest_iso > last_arrivals[key]:
                    last_arrivals[key] = latest_iso

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
    from backend.worker.trips import compute_trips_from_vehicle_data  # local import avoids a cycle at module load

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
    routes = Stops.routes_data
    velocity_predictor = get_velocity_predictor()
    next_stop_etas = await predict_eta(vehicle_ids, df=full_df)

    vehicle_stop_etas: Dict[str, Dict] = {}

    # PERF: single sort + groupby vs N boolean masks.
    target_ids = {str(v) for v in vehicle_ids}
    work_df = full_df[full_df['vehicle_id'].astype(str).isin(target_ids)]
    if work_df.empty:
        return ({}, {})
    work_df = work_df.sort_values('timestamp', kind='mergesort')
    vehicle_groups = {
        str(vid): grp for vid, grp in work_df.groupby('vehicle_id', sort=False)
    }

    for vehicle_id in vehicle_ids:
        vehicle_df = vehicle_groups.get(str(vehicle_id))
        if vehicle_df is None or vehicle_df.empty:
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
                    # POLYLINE-INTERSECTION GUARD: the same
                    # (current_polyline_idx < half_route,
                    #  last_stop_idx >= half_route)
                    # pattern is produced by TWO different situations:
                    #
                    #  (a) TRUE loop restart. Shuttle physically reached
                    #      Union, new loop starting. polyline_idx correctly
                    #      points at the start of the new loop.
                    #
                    #  (b) INTERSECTION CONFUSION. The return-leg polyline
                    #      (HFH → ... → Union) geographically crosses the
                    #      first-outbound polyline (Union → COLONIE)
                    #      somewhere near HFH. When the shuttle is physically
                    #      at HFH on its return leg, closest-point matching
                    #      can pick a point on the first-outbound polyline
                    #      instead of the return polyline — giving a spurious
                    #      current_polyline_idx = 0 even though the shuttle
                    #      never touched Union.
                    #
                    # Without this guard, case (b) was misclassified as a
                    # loop restart: the override below was skipped, the
                    # predictor built vehicle_stops starting from COLONIE,
                    # and HFH landed in eta_lookup with a future eta. The
                    # spurious-detection scrub in build_trip_etas then saw
                    # "HFH has a future eta 5 min out AND a real detection"
                    # and DROPPED the real detection as noise — flipping the
                    # UI from "Last: HH:MM" to "ETA: HH:MM LIVE" on HFH
                    # despite the shuttle having just passed it.
                    #
                    # Distinguishing signal: a TRUE loop restart requires
                    # the shuttle's physical GPS to be near the first stop's
                    # coordinate. Intersection confusion has the GPS
                    # hundreds of meters away (still near HFH). 100 m is a
                    # generous threshold — well beyond GPS jitter, tight
                    # enough to reject the HFH-area intersection.
                    is_physically_at_first_stop = False
                    first_stop_data = route_data.get(stops[0], {})
                    first_stop_coord = first_stop_data.get("COORDINATES")
                    last_lat = last_point.get('latitude')
                    last_lon = last_point.get('longitude')
                    if (
                        first_stop_coord is not None
                        and not pd.isna(last_lat)
                        and not pd.isna(last_lon)
                    ):
                        try:
                            d_km = haversine(
                                (float(last_lat), float(last_lon)),
                                (float(first_stop_coord[0]), float(first_stop_coord[1])),
                            )
                            is_physically_at_first_stop = d_km < 0.100  # 100 m
                        except (TypeError, ValueError):
                            pass

                    is_loop_restart = (
                        current_polyline_idx < half_route
                        and last_stop_idx >= half_route
                        and is_physically_at_first_stop
                    )
                    if not is_loop_restart and current_polyline_idx < last_stop_idx:
                        # INDEX-SPACE CORRECTION: `last_stop_idx` is an
                        # index into STOPS, which excludes ghost stops.
                        # `current_polyline_idx` indexes into the ROUTES
                        # polyline list, which does include them
                        # (NORTH has GHOST_STOP_1 between COLONIE and
                        # GEORGIAN, GHOST_STOP_2 between HFH and SUR).
                        # Assigning last_stop_idx directly produces an
                        # off-by-ghost-count polyline index for any
                        # last_stop past the first ghost. Map through
                        # POLYLINE_STOPS to get the correct index.
                        polyline_stops_list = route_data.get('POLYLINE_STOPS', stops)
                        if last_stop in polyline_stops_list:
                            current_polyline_idx = polyline_stops_list.index(last_stop)
                        else:
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

        # Dwelling-at-Union: the vehicle's latest stop_name is the LAST stop
        # in STOPS (e.g. STUDENT_UNION_RETURN) AND that stop's coordinates
        # coincide with the FIRST stop's coordinates. In that case the shuttle
        # is physically at the boundary between loops — emit ETAs starting
        # from stop index 1 so the dwell-promote logic in
        # compute_trips_from_vehicle_data can match it to the next scheduled
        # slot instead of dropping the vehicle entirely (it would otherwise
        # fall through the `next_stop_idx >= len(stops)` guard below).
        if (
            not pd.isna(now_stop_key)
            and str(now_stop_key) in stops
            and stops.index(str(now_stop_key)) == len(stops) - 1
            and len(stops) >= 2
        ):
            first_stop_data = route_data.get(stops[0])
            last_stop_data = route_data.get(stops[-1])
            if (
                isinstance(first_stop_data, dict)
                and isinstance(last_stop_data, dict)
                and first_stop_data.get("COORDINATES") is not None
                and last_stop_data.get("COORDINATES") is not None
                and tuple(first_stop_data["COORDINATES"]) == tuple(last_stop_data["COORDINATES"])
            ):
                next_stop_idx = 1

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

    # Compute per-vehicle last_arrivals. Keyed by vehicle_id so trip A
    # doesn't inherit trip B's "passed" stops.
    #
    # PER-TRIP SCOPING CONTRACT: this function returns the raw latest
    # detection per (vehicle, stop) without any time-based filter.
    # `build_trip_etas` is responsible for filtering by the current
    # trip's `loop_cutoff` (= actual_departure for active trips,
    # prior_departure for completed trips). That's the only place
    # with enough context to know which detections belong to which
    # loop — a time-based window here leaked prior-loop detections
    # into the current trip's display (see commit history: the
    # LOOP_FRESHNESS_SEC sliding window and the earlier
    # `loop_start_ts = max(first_stop_detection)` approaches both
    # had loop-boundary bugs for different reasons. Scoping in
    # `build_trip_etas` sidesteps both, because by that point we
    # already know per-trip departure times).
    last_arrivals_by_vehicle: Dict[str, Dict[str, str]] = {}

    # Build per-vehicle last_arrivals from the raw stop_name column, but
    # use the ACTUAL CLOSEST-APPROACH timestamp — not the max timestamp
    # over every ping tagged with a given stop_name.
    #
    # Why: the ML pipeline's `stop_name` column is over-permissive. A
    # single stop crossing gets tagged on MANY consecutive pings along
    # the approach, the drive-by, and the departure — not just the one
    # closest-approach ping. (See add_stops_from_segments, clean_stops,
    # _assign_multi_jump_stops in ml/data/stops.py. They backfill
    # stop_name from polyline-index transitions and segment
    # perpendiculars, both of which can assign a stop to a row whose
    # GPS is hundreds of meters away from the stop's real coordinate.)
    # Live example seen while debugging: rows at 16:57:54 (533 m from
    # COLONIE), 16:58:04 (419 m), 16:58:09 (343 m), 16:58:14 (248 m),
    # 16:58:19 (156 m) were all tagged stop_name=COLONIE before the
    # shuttle was actually anywhere near the stop.
    #
    # Using max(timestamp) per (vehicle, stop) treats the last tagged
    # ping as "the arrival", but that's often a post-drive-by or
    # lingering approach sample. Users then see "Last: HH:MM" on stops
    # the shuttle hasn't really arrived at yet.
    #
    # Fix (two parts):
    #  1. Filter to pings that are PHYSICALLY within CLOSE_APPROACH_M
    #     of the stop's coordinate. 60 m is ~1.5x typical inter-tick
    #     travel at the 20 mph test-shuttle speed (~44 m per 5 s tick),
    #     so a single GPS ping mid-drive-by still lands inside the
    #     radius. (The previous 40 m — 2x the 20 m pipeline threshold —
    #     was tight enough that a shuttle moving at test speed could
    #     straddle the radius between ticks and skip the detection
    #     entirely, e.g. at GEORGIAN.) 60 m still leaves plenty of
    #     headroom for GPS jitter while staying well below the ~80–120 m
    #     inter-stop spacing on campus, so it doesn't cause
    #     adjacent-stop false positives.
    #  2. Among the surviving pings, pick the one with the SMALLEST
    #     distance (the true closest approach) and use ITS timestamp
    #     as the last_arrival. Not max, not min — closest-approach.
    CLOSE_APPROACH_M = 60.0
    if 'stop_name' in full_df.columns:
        tracked_vids_set = {str(v) for v in vehicle_ids}
        stops_df = full_df.dropna(subset=['stop_name', 'latitude', 'longitude'])
        if not stops_df.empty:
            routes = Stops.routes_data
            # Build a stop_name -> (lat, lon) lookup on the union of
            # every known route. Same key may exist under multiple
            # routes (STUDENT_UNION_RETURN on NORTH+WEST) with the same
            # coordinate — first occurrence wins, that's fine.
            stop_coord_lookup: Dict[str, Tuple[float, float]] = {}
            for route_info in routes.values():
                for stop_name, stop_data in route_info.items():
                    if not isinstance(stop_data, dict):
                        continue
                    coord = stop_data.get("COORDINATES")
                    if coord and stop_name not in stop_coord_lookup:
                        stop_coord_lookup[stop_name] = (float(coord[0]), float(coord[1]))

            # Vectorized haversine: compute distance from each (lat, lon)
            # to the stop coordinate its row was tagged with.
            sn = stops_df['stop_name'].astype(str).values
            lats = stops_df['latitude'].astype(float).values
            lons = stops_df['longitude'].astype(float).values
            stop_lats = np.empty(len(sn), dtype=float)
            stop_lons = np.empty(len(sn), dtype=float)
            for i, name in enumerate(sn):
                coord = stop_coord_lookup.get(name)
                if coord is None:
                    stop_lats[i] = np.nan
                    stop_lons[i] = np.nan
                else:
                    stop_lats[i], stop_lons[i] = coord

            # Haversine in meters (vectorized). R = 6,371,000 m.
            R = 6371000.0
            phi1 = np.radians(lats)
            phi2 = np.radians(stop_lats)
            dphi = np.radians(stop_lats - lats)
            dlam = np.radians(stop_lons - lons)
            a = (
                np.sin(dphi / 2.0) ** 2
                + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
            )
            dists_m = 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

            close_mask = dists_m <= CLOSE_APPROACH_M
            close_rows = stops_df[close_mask].copy()
            close_rows['_dist_m'] = dists_m[close_mask]

            if not close_rows.empty:
                # Sort by distance ASCENDING within each (vehicle, stop)
                # group, then drop_duplicates keeping the first row —
                # that's the closest-approach ping. Its timestamp is the
                # real last_arrival.
                close_rows = close_rows.sort_values('_dist_m', kind='mergesort')
                closest_approach = close_rows.drop_duplicates(
                    subset=['vehicle_id', 'stop_name'], keep='first'
                )
                ts_series = pd.to_datetime(closest_approach['timestamp'], utc=True)
                for (vid, stop_key, ts) in zip(
                    closest_approach['vehicle_id'].values,
                    closest_approach['stop_name'].astype(str).values,
                    ts_series,
                ):
                    vid_str = str(vid)
                    if vid_str not in tracked_vids_set:
                        continue
                    if pd.isna(ts):
                        continue
                    ts_dt = ts.to_pydatetime()
                    if ts_dt.tzinfo is None:
                        ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                    vehicle_las = last_arrivals_by_vehicle.setdefault(vid_str, {})
                    vehicle_las[str(stop_key)] = ts_dt.isoformat()

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

    # Run per-trip ETAs and next-state prediction in parallel. They share
    # the same dataframe and don't depend on each other. `compute_per_stop_etas`
    # was removed from the cycle — /api/trips is now the single source of
    # truth and the frontend derives per-stop aggregates client-side via
    # `deriveStopEtasFromTrips` in useTrips.ts. The function is still exercised
    # by unit tests for its per-vehicle ETA math, but no longer runs per cycle.
    trips, next_states = await asyncio.gather(
        compute_trips(vehicle_ids, df=df),
        predict_next_state(vehicle_ids, df=df),
    )

    # Write trips to Redis for /api/trips and /api/trips/stream consumers.
    redis = get_redis()
    if redis and trips:
        await redis.set(
            "shubble:trips_live",
            json.dumps(trips).encode(),
            ex=120,
        )
        # PUSH: notify /api/trips/stream SSE subscribers. Fire-and-forget;
        # if nobody's listening the publish is a ~0ms no-op.
        try:
            await redis.publish("shubble:trips_updated", b"1")
        except Exception as e:
            logger.warning(f"Failed to publish shubble:trips_updated: {e}")

    # save_predictions still writes PredictedLocation rows for /api/velocities;
    # passing an empty per_stop_etas dict makes it a no-op for the ETA table.
    await save_predictions({}, next_states)

    count_trips = len(trips)
    count_locs = len(next_states)
    if count_trips > 0 or count_locs > 0:
        logger.info(f"Saved {count_trips} trips and {count_locs} predicted locations")
