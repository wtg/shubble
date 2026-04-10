"""Stop-related functions for vehicle location data."""
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()


# Cached duplicate-stop remap, computed once from routes.json at
# module load. Used by `resolve_duplicate_stops` — pure config, so
# rebuilding it on every call (multiple cycles per minute × 2
# routes) was pure waste. See P2 in session-handoff-2026-04-10.md.
_ROUTE_REMAP_CACHE: dict = {}  # {route_name: [(first_name, last_name, threshold_idx), ...]}


def _build_route_remap_cache() -> None:
    """Populate `_ROUTE_REMAP_CACHE` once from Stops.routes_data."""
    global _ROUTE_REMAP_CACHE
    if _ROUTE_REMAP_CACHE:
        return
    from shared.stops import Stops

    remap: dict = {}
    for route_name, route_data in Stops.routes_data.items():
        stops_list = route_data.get('STOPS', [])
        if not stops_list:
            continue
        coord_to_names: dict = {}
        for stop_name in stops_list:
            if stop_name not in route_data:
                continue
            coords = route_data[stop_name].get('COORDINATES')
            if not coords:
                continue
            key = (round(float(coords[0]), 6), round(float(coords[1]), 6))
            coord_to_names.setdefault(key, []).append(stop_name)

        duplicates = [(k, names) for k, names in coord_to_names.items() if len(names) > 1]
        if not duplicates:
            continue

        n_polys = len(route_data.get('ROUTES', []))
        if n_polys < 2:
            continue
        threshold_idx = n_polys // 2

        remap_entries: list = []
        for _, names in duplicates:
            first_name = names[0]
            last_name = names[-1]
            if first_name != last_name:
                remap_entries.append((first_name, last_name, threshold_idx))
        if remap_entries:
            remap[route_name] = remap_entries

    _ROUTE_REMAP_CACHE = remap


def resolve_duplicate_stops(
    df: pd.DataFrame,
    route_column: str,
    polyline_index_column: str,
    stop_column: str,
) -> None:
    """Disambiguate stops that share coordinates on the same route.

    Some routes have multiple stops at the same physical point — e.g.
    STUDENT_UNION (route start) and STUDENT_UNION_RETURN (route end)
    both sit at (42.730711, -73.676737). The per-point ``is_at_stop``
    check returns the geometrically-closest stop, which by argmin
    always resolves to the same one (STUDENT_UNION). The "return"
    variant never registers detections, breaking trip-completion
    logic that looks for the final stop.

    This function re-maps ambiguous stop assignments using
    ``polyline_idx`` context. For each duplicate pair on a route, we
    know which polyline index sits near the route end; a row whose
    current polyline_idx is in that range gets the "return" stop
    name instead of the start one.

    Only remaps within-row assignments — it doesn't invent detections
    where none exist.
    """
    if df.empty or stop_column not in df.columns:
        return

    _build_route_remap_cache()
    if not _ROUTE_REMAP_CACHE:
        return

    remapped = 0
    for route_name, entries in _ROUTE_REMAP_CACHE.items():
        for first_name, last_name, threshold_idx in entries:
            mask = (
                (df[route_column] == route_name)
                & (df[stop_column] == first_name)
                & (df[polyline_index_column].notna())
                & (df[polyline_index_column] >= threshold_idx)
            )
            count = int(mask.sum())
            if count > 0:
                df.loc[mask, stop_column] = last_name
                remapped += count

    if remapped > 0:
        print(f"   Remapped {remapped} duplicate-coordinate stop assignments")


def add_stops(
    df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    output_columns: dict[str, str],
    threshold: float = 0.020
) -> None:
    """
    Add stop information by checking if coordinates are near any stop.

    For each GPS coordinate, uses the Stops.is_at_stop() function to determine
    if the location is close enough to a stop (within threshold distance).

    The is_at_stop function returns:
    - route_name: Name of the route if at a stop, otherwise None
    - stop_name: Name of the stop if at a stop, otherwise None

    Modifies the dataframe in place by adding new columns based on output_columns mapping.

    Args:
        df: Pandas DataFrame to modify
        lat_column: Name of the latitude column
        lon_column: Name of the longitude column
        output_columns: Dictionary mapping return value names to output column names.
                       Valid keys: 'route_name', 'stop_name'
                       Only specified keys will be added as columns.
        threshold: Distance threshold in km to consider as "at stop" (default: 0.020)

    Raises:
        KeyError: If lat_column or lon_column doesn't exist in the dataframe

    Example:
        >>> df = pd.DataFrame({
        ...     'latitude': [42.7284, 42.7295],
        ...     'longitude': [-73.6788, -73.6799]
        ... })
        >>> add_stops(df, 'latitude', 'longitude', {
        ...     'route_name': 'stop_route',
        ...     'stop_name': 'stop'
        ... }, threshold=0.020)
        >>> # df now has columns: stop_route, stop
    """
    # Import here to avoid circular imports
    from shared.stops import Stops

    # Validation
    if lat_column not in df.columns:
        raise KeyError(f"Column '{lat_column}' not found in dataframe")
    if lon_column not in df.columns:
        raise KeyError(f"Column '{lon_column}' not found in dataframe")

    if not output_columns:
        return

    # PERF: drop progress_apply — it allocates a pd.Series per row.
    # Collect into plain lists and bulk-assign.
    out_cols: dict[str, list] = {output_col: [] for output_col in output_columns.values()}
    subset = df[[lat_column, lon_column]]
    for lat, lon in subset.itertuples(index=False, name=None):
        if pd.isna(lat) or pd.isna(lon):
            for output_col in output_columns.values():
                out_cols[output_col].append(None)
            continue
        route_name, stop_name = Stops.is_at_stop((lat, lon), threshold=threshold)
        value_map = {'route_name': route_name, 'stop_name': stop_name}
        for key, output_col in output_columns.items():
            out_cols[output_col].append(value_map[key])
    for output_col, values in out_cols.items():
        df[output_col] = values


def add_polyline_distances(
    df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    output_columns: dict[str, str],
    distance_column: str = None,
    closest_point_lat_column: str = None,
    closest_point_lon_column: str = None,
    route_column: str = None,
    polyline_index_column: str = None,
    segment_index_column: str = None
) -> None:
    """
    Add polyline distance information using Stops.get_polyline_distances.

    For each GPS coordinate, calculates the distance from the start of the polyline,
    distance to the end of the polyline, and total polyline length.

    If the closest point columns are provided, they will be used to construct the
    closest_point_result tuple and avoid redundant calculations. Otherwise,
    get_closest_point will be called for each row.

    Modifies the dataframe in place by adding new columns based on output_columns mapping.

    Args:
        df: Pandas DataFrame to modify
        lat_column: Name of the latitude column
        lon_column: Name of the longitude column
        output_columns: Dictionary mapping return value names to output column names.
                       Valid keys: 'distance_from_start', 'distance_to_end', 'total_length'
                       Only specified keys will be added as columns.
        distance_column: Optional name of column with distance to closest point (from get_closest_point)
        closest_point_lat_column: Optional name of column with closest point latitude
        closest_point_lon_column: Optional name of column with closest point longitude
        route_column: Optional name of column with route name (from get_closest_point)
        polyline_index_column: Optional name of column with polyline index (from get_closest_point)
        segment_index_column: Optional name of column with segment index (from get_closest_point)

    Raises:
        KeyError: If required columns don't exist in the dataframe

    Example:
        >>> # Using existing closest point data
        >>> df = pd.DataFrame({
        ...     'latitude': [42.7284, 42.7295],
        ...     'longitude': [-73.6788, -73.6799],
        ...     'route': ['North Route', 'North Route'],
        ...     'polyline_idx': [0, 0],
        ...     'segment_idx': [5, 6],
        ...     'dist_to_route': [0.005, 0.007],
        ...     'closest_lat': [42.7285, 42.7296],
        ...     'closest_lon': [-73.6789, -73.6800]
        ... })
        >>> add_polyline_distances(
        ...     df, 'latitude', 'longitude',
        ...     {
        ...         'distance_from_start': 'dist_from_start',
        ...         'distance_to_end': 'dist_to_end',
        ...         'total_length': 'total_km'
        ...     },
        ...     distance_column='dist_to_route',
        ...     closest_point_lat_column='closest_lat',
        ...     closest_point_lon_column='closest_lon',
        ...     route_column='route',
        ...     polyline_index_column='polyline_idx',
        ...     segment_index_column='segment_idx'
        ... )
        >>> # df now has columns: dist_from_start, dist_to_end, total_km

        >>> # Without existing closest point data (will call get_closest_point)
        >>> df = pd.DataFrame({
        ...     'latitude': [42.7284, 42.7295],
        ...     'longitude': [-73.6788, -73.6799]
        ... })
        >>> add_polyline_distances(
        ...     df, 'latitude', 'longitude',
        ...     {
        ...         'distance_from_start': 'dist_from_start',
        ...         'distance_to_end': 'dist_to_end'
        ...     }
        ... )
    """
    # Import here to avoid circular imports
    from shared.stops import Stops

    # Validation
    if lat_column not in df.columns:
        raise KeyError(f"Column '{lat_column}' not found in dataframe")
    if lon_column not in df.columns:
        raise KeyError(f"Column '{lon_column}' not found in dataframe")

    # Check if we have all the columns needed to construct closest_point_result
    has_closest_point_data = all([
        distance_column and distance_column in df.columns,
        closest_point_lat_column and closest_point_lat_column in df.columns,
        closest_point_lon_column and closest_point_lon_column in df.columns,
        route_column and route_column in df.columns,
        polyline_index_column and polyline_index_column in df.columns,
        segment_index_column and segment_index_column in df.columns
    ])

    if not output_columns:
        return

    # PERF: columnar collection + bulk assign. Previously progress_apply ran
    # process_row per row, allocating a pd.Series each time — that's a lot of
    # overhead when the pipeline runs on every worker cycle.
    out_cols: dict[str, list] = {output_col: [] for output_col in output_columns.values()}
    if has_closest_point_data:
        subset = df[[
            lat_column, lon_column,
            distance_column, closest_point_lat_column, closest_point_lon_column,
            route_column, polyline_index_column, segment_index_column,
        ]]
        for (lat, lon, distance, closest_lat, closest_lon,
             route_name, polyline_idx, segment_idx) in subset.itertuples(index=False, name=None):
            if pd.isna(lat) or pd.isna(lon):
                for output_col in output_columns.values():
                    out_cols[output_col].append(np.nan)
                continue
            closest_point_result = None
            if not (pd.isna(distance) or pd.isna(closest_lat) or pd.isna(closest_lon)
                    or pd.isna(route_name) or pd.isna(polyline_idx) or pd.isna(segment_idx)):
                closest_point_result = (
                    distance, [closest_lat, closest_lon], route_name, polyline_idx, segment_idx
                )
            distance_from_start, distance_to_end, total_length = Stops.get_polyline_distances(
                (lat, lon),
                closest_point_result=closest_point_result,
            )
            # Normalize raw None returns from get_polyline_distances to np.nan
            # so pandas doesn't reject None when assigning to a float64 column.
            value_map = {
                'distance_from_start': distance_from_start if distance_from_start is not None else np.nan,
                'distance_to_end': distance_to_end if distance_to_end is not None else np.nan,
                'total_length': total_length if total_length is not None else np.nan,
            }
            for key, output_col in output_columns.items():
                out_cols[output_col].append(value_map[key])
    else:
        subset = df[[lat_column, lon_column]]
        for lat, lon in subset.itertuples(index=False, name=None):
            if pd.isna(lat) or pd.isna(lon):
                for output_col in output_columns.values():
                    out_cols[output_col].append(np.nan)
                continue
            distance_from_start, distance_to_end, total_length = Stops.get_polyline_distances(
                (lat, lon), closest_point_result=None
            )
            value_map = {
                'distance_from_start': distance_from_start if distance_from_start is not None else np.nan,
                'distance_to_end': distance_to_end if distance_to_end is not None else np.nan,
                'total_length': total_length if total_length is not None else np.nan,
            }
            for key, output_col in output_columns.items():
                out_cols[output_col].append(value_map[key])

    # All three output columns are numeric — cast to float so pandas infers
    # the right dtype and accepts np.nan for missing values.
    for output_col, values in out_cols.items():
        df[output_col] = np.asarray(values, dtype=float)


def add_stops_from_segments(
    df: pd.DataFrame,
    vehicle_id_column: str,
    timestamp_column: str,
    lat_column: str,
    lon_column: str,
    stop_column: str,
    route_column: str,
    max_time_gap_sec: float = 30.0,
    max_segment_km: float = 0.200,
    threshold: float = 0.020,
) -> None:
    """
    Backfill missed stop detections by checking the line segment between
    consecutive GPS polls, not just the sample points.

    At ~30 km/h a shuttle moves ~42m per 5s poll, which can exceed the
    20m per-point threshold used by ``add_stops``. Two consecutive pings can
    straddle a stop without either being within 20m of it, so the drive-by
    is never recorded. This function fills the gap by computing the
    perpendicular distance from each route stop to the line segment
    ``(prev_point, curr_point)`` and, when it falls below ``threshold``,
    assigning the stop to whichever endpoint is closer to the stop
    location.

    Safe to call multiple times — rows that already have a stop_name
    assigned are left untouched.

    Modifies the dataframe in place by populating ``stop_column`` where
    segment-based detection finds a drive-by that ``add_stops`` missed.

    Args:
        df: DataFrame to modify (must contain the columns below)
        vehicle_id_column: Column grouping rows into per-vehicle sequences
        timestamp_column: datetime64 column used to measure poll-to-poll gap
        lat_column: Latitude column
        lon_column: Longitude column
        stop_column: Stop-name column (populated by add_stops). Will be
            updated in place for rows where a drive-by is detected.
        route_column: Route column (from add_polyline_distances). Used to
            narrow the stop candidate list to the pair's active route.
        max_time_gap_sec: Pairs with a larger time gap are skipped — the
            line-segment interpolation is only meaningful for consecutive
            polls. Default: 30 seconds.
        max_segment_km: Pairs whose endpoints are farther apart than this
            are skipped. A straight-line interpolation through a long
            segment can cross a stop the shuttle never actually visited
            (e.g. two polls 150m apart with a diagonal between them).
            Default: 0.100 km (100m), i.e. shuttles at up to ~70 km/h
            between 5s polls.
        threshold: Distance threshold in km to consider "at stop". Must
            match the per-point threshold used by ``add_stops``.
    """
    # Import here to avoid circular imports
    from shared.stops import Stops, haversine_vectorized

    if df.empty or stop_column not in df.columns:
        return

    required_cols = [vehicle_id_column, timestamp_column, lat_column,
                     lon_column, stop_column, route_column]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataframe")

    # Sort a lightweight view by (vehicle_id, timestamp) so .shift(1) yields
    # the previous GPS sample for the same vehicle. We preserve the
    # original index so assignments write back to the right rows.
    order = df[[vehicle_id_column, timestamp_column]].sort_values(
        [vehicle_id_column, timestamp_column], kind='stable'
    ).index

    ordered_vids = df[vehicle_id_column].reindex(order).values
    ordered_lats = df[lat_column].reindex(order).values
    ordered_lons = df[lon_column].reindex(order).values
    ordered_ts = df[timestamp_column].reindex(order).values
    ordered_stops = df[stop_column].reindex(order).values
    ordered_routes = df[route_column].reindex(order).values
    ordered_idx = order.to_numpy()

    # Walk consecutive pairs, collect candidate assignments, apply at end.
    # Building the mask vectorized, then iterating only the survivors,
    # keeps the Python loop bounded by actual drive-by candidates
    # (typically a handful per worker cycle).
    n = len(ordered_vids)
    if n < 2:
        return

    prev_vids = ordered_vids[:-1]
    curr_vids = ordered_vids[1:]
    same_vehicle = prev_vids == curr_vids

    prev_lats = ordered_lats[:-1]
    curr_lats = ordered_lats[1:]
    prev_lons = ordered_lons[:-1]
    curr_lons = ordered_lons[1:]
    valid_coords = (
        ~pd.isna(prev_lats) & ~pd.isna(curr_lats)
        & ~pd.isna(prev_lons) & ~pd.isna(curr_lons)
    )

    prev_ts = ordered_ts[:-1]
    curr_ts = ordered_ts[1:]
    # Use pandas to coerce to timedelta safely regardless of input dtype
    time_gap_sec = pd.to_timedelta(
        pd.Series(curr_ts) - pd.Series(prev_ts)
    ).dt.total_seconds().values
    within_window = (time_gap_sec > 0) & (time_gap_sec <= max_time_gap_sec)

    prev_stops = ordered_stops[:-1]
    curr_stops = ordered_stops[1:]
    # Fire on pairs where AT LEAST ONE endpoint is empty. Previously
    # we required both empty, which meant add_stops (a per-point
    # check that picks the closest stop globally) could block this
    # pass: if the shuttle was near ECAV at T1 and slightly past it
    # heading to HOUSTON_FIELD_HOUSE at T2, add_stops picks ECAV for
    # T1 but T2's closer stop is none → T2 ends up with stop_name=None
    # while add_stops_from_segments would have filled HFH. With the
    # stricter rule, T1 having ECAV blocked the whole pair and we
    # lost the HFH detection.
    at_least_one_empty = pd.isna(prev_stops) | pd.isna(curr_stops)

    candidate_mask = same_vehicle & valid_coords & within_window & at_least_one_empty
    candidate_positions = np.nonzero(candidate_mask)[0]

    if len(candidate_positions) == 0:
        return

    # Cache stop arrays per route so we don't re-fetch on every candidate
    route_stops_cache: dict = {}

    def _route_stop_coords(route_name):
        cached = route_stops_cache.get(route_name)
        if cached is not None:
            return cached
        route_info = Stops.routes_data.get(route_name)
        if not route_info:
            route_stops_cache[route_name] = None
            return None
        stops_list = route_info.get('STOPS', [])
        if not stops_list:
            route_stops_cache[route_name] = None
            return None
        coords = np.array([
            route_info[stop_name]['COORDINATES'] for stop_name in stops_list
        ], dtype=float)
        entry = (stops_list, coords)
        route_stops_cache[route_name] = entry
        return entry

    # Pending assignments: {original_df_index: stop_name}
    assignments: dict = {}
    assigned_count = 0

    prev_routes = ordered_routes[:-1]
    curr_routes = ordered_routes[1:]

    for pos in candidate_positions:
        # Pick whichever row has a route (prefer current — it's the freshest)
        route = curr_routes[pos]
        if pd.isna(route):
            route = prev_routes[pos]
        if pd.isna(route):
            continue

        entry = _route_stop_coords(route)
        if entry is None:
            continue
        stops_list, stop_coords = entry

        p1 = np.array([prev_lats[pos], prev_lons[pos]], dtype=float)
        p2 = np.array([curr_lats[pos], curr_lons[pos]], dtype=float)
        seg_vec = p2 - p1
        length_sq = float(np.dot(seg_vec, seg_vec))

        if length_sq < 1e-14:
            # Zero-length segment — add_stops would already have fired on
            # either endpoint, so this is effectively just a per-point
            # check. Skip to keep the logic focused on true drive-bys.
            continue

        # Reject pairs that are too far apart: a straight-line
        # interpolation through a very long gap would fabricate stop
        # crossings the shuttle never actually made. This is the GPS
        # analogue of the "time gap too big" skip above — physical
        # distance is a cleaner signal than time when polls are sparse.
        seg_len_km = float(haversine_vectorized(p1[np.newaxis, :],
                                                p2[np.newaxis, :])[0])
        if seg_len_km > max_segment_km:
            continue

        # Project every route stop onto the segment in degree space
        diffs = stop_coords - p1  # (N, 2)
        t = diffs @ seg_vec / length_sq  # (N,)
        t = np.clip(t, 0.0, 1.0)
        closest = p1 + t[:, np.newaxis] * seg_vec  # (N, 2)

        # Haversine distance from each stop to its closest point on segment
        dists = haversine_vectorized(stop_coords, closest)

        within = dists < threshold
        if not np.any(within):
            continue

        # If multiple stops qualify (unusual on a single ~42m segment),
        # pick the closest so we don't pollute the row with a distant stop
        best_local_idx = int(np.argmin(np.where(within, dists, np.inf)))
        best_stop_name = stops_list[best_local_idx]
        best_t = float(t[best_local_idx])

        # Choose target endpoint. Preference order:
        # 1. Endpoint closer to the stop (t<0.5 -> prev, else curr)
        # 2. Fall back to the OTHER endpoint if the preferred one
        #    already has a stop_name (either pre-existing or set by
        #    an earlier candidate in this pass). Preserves any correct
        #    per-point assignment while still recording the drive-by.
        prev_df_idx = int(ordered_idx[pos])
        curr_df_idx = int(ordered_idx[pos + 1])
        prev_is_empty = pd.isna(prev_stops[pos]) and prev_df_idx not in assignments
        curr_is_empty = pd.isna(curr_stops[pos]) and curr_df_idx not in assignments

        if best_t < 0.5:
            preferred_idx, preferred_empty = prev_df_idx, prev_is_empty
            fallback_idx, fallback_empty = curr_df_idx, curr_is_empty
        else:
            preferred_idx, preferred_empty = curr_df_idx, curr_is_empty
            fallback_idx, fallback_empty = prev_df_idx, prev_is_empty

        if preferred_empty:
            assignments[preferred_idx] = best_stop_name
            assigned_count += 1
        elif fallback_empty:
            assignments[fallback_idx] = best_stop_name
            assigned_count += 1
        # Else: both endpoints already occupied, nothing to do

    if not assignments:
        return

    # Apply all assignments in a single vectorized write
    idx_array = np.fromiter(assignments.keys(), dtype=np.int64,
                            count=len(assignments))
    val_array = np.array(list(assignments.values()), dtype=object)
    df.loc[idx_array, stop_column] = val_array

    print(f"   Segment pass assigned {assigned_count} drive-by stops")


def clean_stops(
    df: pd.DataFrame,
    route_column: str,
    polyline_index_column: str,
    stop_column: str,
    lat_column: str,
    lon_column: str,
    distance_column: str,
) -> None:
    """
    Rectify unrecorded stops by identifying jumps in polyline indices without stop records.

    When a shuttle passes a stop but the event is not recorded in the data, there will be
    a jump in polyline indices between consecutive rows without any stop being logged.
    This function detects these gaps and assigns the missing stop to either the previous
    or current data point based on which GPS position is closer to the actual stop location.

    The function examines consecutive rows within the same route and looks for cases where:
    1. The polyline index increases between rows
    2. Neither row has a stop recorded
    3. A stop should exist between the two polyline indices

    For each detected gap, it determines which GPS point (previous or current) is closer
    to the unrecorded stop and assigns that stop accordingly.

    Modifies the dataframe in place by populating the stop_column where stops were missing.

    Args:
        df: Pandas DataFrame to modify
        route_column: Name of the column containing route names
        polyline_index_column: Name of the column containing polyline indices
        stop_column: Name of the column containing stop names (will be populated)
        lat_column: Name of the latitude column
        lon_column: Name of the longitude column
        distance_column: Name of the column containing distance to closest point on route

    Raises:
        None

    Example:
        >>> # Before clean_stops: polyline_idx jumps from 0 to 1 without any stops for either point
        >>> df = pd.DataFrame({
        ...     'route': ['North Route', 'North Route', 'North Route'],
        ...     'polyline_idx': [0, 1, 2],
        ...     'stop': [None, None, 'Georgian'],
        ...     'latitude': [42.7284, 42.7295, 42.7300],
        ...     'longitude': [-73.6788, -73.6799, -73.6805],
        ...     'distance_to_route': [0.015, 0.003, 0.001]
        ... })
        >>> clean_stops(
        ...     df, 'route', 'polyline_idx', 'stop',
        ...     'latitude', 'longitude', 'distance_to_route'
        ... )
        >>> # After clean_stops: the middle point (index 1) is closer to the stop 
        >>> # location than the previous point (index 0), so the stop is assigned to index 1
        >>> df = pd.DataFrame({
        ...     'route': ['North Route', 'North Route', 'North Route'],
        ...     'polyline_idx': [0, 1, 2],
        ...     'stop': [None, 'Colonie', 'Georgian'],
        ...     'latitude': [42.7284, 42.7295, 42.7300],
        ...     'longitude': [-73.6788, -73.6799, -73.6805],
        ...     'distance_to_route': [0.015, 0.003, 0.001]
        ... })
        >>> clean_stops(
        ...     df, 'route', 'polyline_idx', 'stop',
        ...     'latitude', 'longitude', 'distance_to_route'
        ... )
    """
    # Import here to avoid circular imports

    df['prev_route'] = df[route_column].shift(1)
    df['prev_polyline_index'] = df[polyline_index_column].shift(1)
    df['prev_stop'] = df[stop_column].shift(1)
    df['prev_lat'] = df[lat_column].shift(1)
    df['prev_lon'] = df[lon_column].shift(1)
    df['prev_distance'] = df[distance_column].shift(1)

    # Identify any jumps
    jumps_mask = (
        (df[route_column] == df['prev_route']) & # Same route?
        (df[polyline_index_column].notna()) & # Current index valid?
        (df['prev_polyline_index'].notna()) & # Previous index valid?
        (df[polyline_index_column] == df['prev_polyline_index'] + 1) # Polyline index increased?
    )

    # Filter for only unidentified stops with jumps
    unrecorded_mask = (
        jumps_mask &
        (df[stop_column].isna()) & # No current stop
        (df['prev_stop'].isna()) # No previous stop
    )

    unrecorded_jumps = df[unrecorded_mask]

    # Clean data frame if no unrecorded single-jumps found — but still
    # run the multi-jump pass below, it fires on jumps of 2+ which are
    # an independent population of gaps.
    if len(unrecorded_jumps) == 0:
        print("   No unrecorded stop jumps found")
        df.drop(columns=['prev_route', 'prev_polyline_index', 'prev_stop', 'prev_lat', 'prev_lon', 'prev_distance'], inplace=True)
        _assign_multi_jump_stops(
            df,
            route_column=route_column,
            polyline_index_column=polyline_index_column,
            stop_column=stop_column,
        )
        return

    print(f"   Found {len(unrecorded_jumps)} unrecorded stop jumps")

    # PERF: actually-vectorized per-route lookup. The prior `.apply(find_stop,
    # axis=1)` was mis-labeled "Vectorized" — it ran a Python function per
    # row. Splitting by route lets us use numpy array indexing once per route.
    from shared.stops import Stops as _Stops
    subset = df.loc[unrecorded_mask, [route_column, polyline_index_column]]
    matched_values = pd.Series(None, index=subset.index, dtype=object)
    for route_name, grp in subset.groupby(route_column, sort=False):
        polyline_stops = _Stops.routes_data.get(route_name, {}).get('POLYLINE_STOPS', [])
        if not polyline_stops:
            continue
        max_idx = len(polyline_stops) - 1
        idx_series = grp[polyline_index_column]
        in_range = idx_series.notna() & (idx_series >= 0) & (idx_series <= max_idx)
        if not in_range.any():
            continue
        valid = idx_series[in_range].astype(int).to_numpy()
        stops_arr = np.asarray(polyline_stops, dtype=object)
        matched_values.loc[idx_series[in_range].index] = stops_arr[valid]
    df.loc[unrecorded_mask, 'matched_stop'] = matched_values

    valid_comparison = (
        unrecorded_mask & 
        df['matched_stop'].notna() &
        df['prev_distance'].notna() &
        df[distance_column].notna()
    )
    
    prev_closer_mask = valid_comparison & (df['prev_distance'] < df[distance_column])
    next_closer_mask = valid_comparison & (df['prev_distance'] >= df[distance_column])

    # Assign stops using vectorized operations
    # For rows where next (current) is closer: directly assign to current row
    df.loc[next_closer_mask, stop_column] = df.loc[next_closer_mask, 'matched_stop']
    
    # For rows where prev is closer: shift assignment to previous row
    df['stop_to_assign_prev'] = None
    df.loc[prev_closer_mask, 'stop_to_assign_prev'] = df.loc[prev_closer_mask, 'matched_stop']
    
    # This effectively assigns the stop to the previous row
    df['stop_assignment'] = df['stop_to_assign_prev'].shift(-1)
    
    # Apply the assignment where we have values
    assignment_mask = df['stop_assignment'].notna()
    df.loc[assignment_mask, stop_column] = df.loc[assignment_mask, 'stop_assignment']
    
    stops_assigned = (prev_closer_mask | next_closer_mask).sum()

    print(f" Assigned {stops_assigned} unrecorded stops")
    temp_cols = ['prev_route', 'prev_polyline_index', 'prev_stop', 'prev_distance',
                 'matched_stop', 'stop_to_assign_prev', 'stop_assignment']
    df.drop(columns=temp_cols, inplace=True, errors='ignore')

    # Second pass: polyline_idx jumps larger than 1.
    #
    # The single-jump logic above only fires when polyline_idx advances by
    # exactly 1. When a poll interval is too sparse to trust segment
    # interpolation (add_stops_from_segments skips it) but polyline_idx
    # still advanced by 2+, we know the shuttle traversed one or more full
    # polylines between samples. Each polyline has a destination stop
    # recorded in POLYLINE_STOPS, so POLYLINE_STOPS[prev_idx+1 : curr_idx+1]
    # is the ordered list of stops the shuttle must have passed.
    #
    # With only two rows available (prev/curr) we can at most backfill two
    # stop_name cells, so we assign the FIRST missed stop to the prev row
    # (closer in time to the start of the gap) and the LAST missed stop to
    # the curr row. For jumps of 3+ the middle stops are dropped — their
    # exact timestamps are unrecoverable without synthetic rows, and their
    # last_arrivals get restored on the next lap anyway.
    _assign_multi_jump_stops(
        df,
        route_column=route_column,
        polyline_index_column=polyline_index_column,
        stop_column=stop_column,
    )


def _assign_multi_jump_stops(
    df: pd.DataFrame,
    route_column: str,
    polyline_index_column: str,
    stop_column: str,
) -> None:
    """Backfill stops for polyline_idx jumps larger than 1.

    See ``clean_stops`` for the rationale. This helper is split out to
    keep its temporary columns isolated from the single-jump pass.
    """
    from shared.stops import Stops

    if df.empty or stop_column not in df.columns:
        return

    shift_cols = {
        '_mj_prev_route': df[route_column].shift(1),
        '_mj_prev_polyline': df[polyline_index_column].shift(1),
        '_mj_prev_stop': df[stop_column].shift(1),
    }
    for name, series in shift_cols.items():
        df[name] = series

    multi_jump_mask = (
        (df[route_column] == df['_mj_prev_route'])
        & df[polyline_index_column].notna()
        & df['_mj_prev_polyline'].notna()
        & (df[polyline_index_column] > df['_mj_prev_polyline'] + 1)
        & df[stop_column].isna()
        & df['_mj_prev_stop'].isna()
    )

    if not multi_jump_mask.any():
        df.drop(columns=list(shift_cols.keys()), inplace=True, errors='ignore')
        return

    candidate_positions = np.nonzero(multi_jump_mask.values)[0]
    df_index = df.index.to_numpy()
    route_values = df[route_column].values
    prev_poly_values = df['_mj_prev_polyline'].values
    curr_poly_values = df[polyline_index_column].values

    # Pending writes: {df_index: stop_name}. Dict guarantees one assignment
    # per row — later multi-jump pairs won't clobber earlier ones.
    assignments: dict = {}
    assigned_count = 0
    jumps_processed = 0

    for pos in candidate_positions:
        route = route_values[pos]
        try:
            prev_idx = int(prev_poly_values[pos])
            curr_idx = int(curr_poly_values[pos])
        except (TypeError, ValueError):
            continue

        route_info = Stops.routes_data.get(route)
        if not route_info:
            continue
        polyline_stops = route_info.get('POLYLINE_STOPS') or []
        if not polyline_stops:
            continue

        # Slice of stops the shuttle passed between samples (inclusive
        # of the curr row's polyline stop — matches the single-jump
        # convention of assigning POLYLINE_STOPS[curr_idx]).
        start = prev_idx + 1
        end = min(curr_idx + 1, len(polyline_stops))
        if start >= end:
            continue
        missed = [polyline_stops[i] for i in range(start, end) if polyline_stops[i]]
        if not missed:
            continue

        jumps_processed += 1

        prev_row_idx = int(df_index[pos - 1]) if pos > 0 else None
        curr_row_idx = int(df_index[pos])

        # First missed stop → prev row (closer in time to gap start)
        if prev_row_idx is not None and prev_row_idx not in assignments:
            assignments[prev_row_idx] = missed[0]
            assigned_count += 1

        # Last missed stop → curr row (closer in time to gap end). Skip
        # if it equals the first one we already assigned to prev, to
        # avoid a single-stop jump duplicating itself.
        if len(missed) > 1 and curr_row_idx not in assignments:
            assignments[curr_row_idx] = missed[-1]
            assigned_count += 1
        elif len(missed) == 1 and curr_row_idx not in assignments and prev_row_idx is None:
            # Edge case: multi-jump on the very first row (no prev available)
            assignments[curr_row_idx] = missed[0]
            assigned_count += 1

    df.drop(columns=list(shift_cols.keys()), inplace=True, errors='ignore')

    if not assignments:
        return

    idx_array = np.fromiter(assignments.keys(), dtype=np.int64,
                            count=len(assignments))
    val_array = np.array(list(assignments.values()), dtype=object)
    df.loc[idx_array, stop_column] = val_array

    print(f"   Multi-jump pass assigned {assigned_count} stops "
          f"across {jumps_processed} jumps")