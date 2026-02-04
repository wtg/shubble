"""Segmentation functions for vehicle location data."""
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()


def segment_by_consecutive(
    df: pd.DataFrame,
    max_timedelta: float,
    segment_column: str,
    distance_column: str = None,
    max_distance_to_route: float = None
) -> pd.DataFrame:
    """
    Add segment_id column to dataframe based on vehicle_id, time gaps, and route distance.

    Assigns a segment_id to each row where consecutive data points from the same
    vehicle with time gaps less than the threshold share the same segment_id.

    A new segment starts when:
    - The vehicle_id changes, OR
    - The time gap between consecutive points exceeds max_timedelta, OR
    - The distance to route exceeds max_distance_to_route (if specified)

    Args:
        df: DataFrame with 'vehicle_id' and 'epoch_seconds' columns
        max_timedelta: Maximum time gap (in seconds) to consider points consecutive
        segment_column: Name of the column to create for segment IDs
        distance_column: Optional column name containing distance to route (in km)
        max_distance_to_route: Optional maximum distance from route (in km).
                               If specified, creates new segment when exceeded.

    Returns:
        DataFrame with added segment_column (sorted by vehicle_id, epoch_seconds)

    Raises:
        KeyError: If 'vehicle_id', 'epoch_seconds', or distance_column are missing
        ValueError: If max_distance_to_route specified without distance_column

    Example:
        >>> df = pd.DataFrame({
        ...     'vehicle_id': [1, 1, 1, 2, 2],
        ...     'epoch_seconds': [0, 5, 100, 0, 5],
        ...     'value': ['a', 'b', 'c', 'd', 'e']
        ... })
        >>> result = segment_by_consecutive(df, max_timedelta=30, segment_column='segment_id')
        >>> result[['vehicle_id', 'epoch_seconds', 'segment_id']]
           vehicle_id  epoch_seconds  segment_id
        0           1              0           1
        1           1              5           1
        2           1            100           2
        3           2              0           3
        4           2              5           3
    """
    # Validation
    if 'vehicle_id' not in df.columns:
        raise KeyError("Column 'vehicle_id' not found in dataframe")
    if 'epoch_seconds' not in df.columns:
        raise KeyError("Column 'epoch_seconds' not found in dataframe")

    if max_distance_to_route is not None and distance_column is None:
        raise ValueError("distance_column must be specified when max_distance_to_route is provided")

    if distance_column is not None and distance_column not in df.columns:
        raise KeyError(f"Column '{distance_column}' not found in dataframe")

    # Work with a copy to avoid modifying original
    df_work = df.sort_values(['vehicle_id', 'epoch_seconds']).reset_index(drop=True).copy()

    # Calculate time deltas within each vehicle group
    df_work['_time_delta'] = df_work.groupby('vehicle_id')['epoch_seconds'].diff()

    # Mark where vehicle changes
    df_work['_vehicle_change'] = df_work['vehicle_id'] != df_work['vehicle_id'].shift(1)

    # Mark segment boundaries (where vehicle changes or time gap is too large)
    # First row of each vehicle is also a boundary (NaN time_delta)
    df_work['_new_segment'] = (
        df_work['_vehicle_change'] |
        (df_work['_time_delta'] > max_timedelta) |
        df_work['_time_delta'].isna()
    )

    # Add distance-based segmentation if specified
    if max_distance_to_route is not None and distance_column is not None:
        # Mark points that are too far from route
        df_work['_off_route'] = (df_work[distance_column] > max_distance_to_route)
        df_work['_prev_off_route'] = (
            df_work.groupby('vehicle_id')['_off_route']
                    .shift(1, fill_value=False)
        )

        # Create boundary at:
        # 1. Any off-route point
        # 2. First on-route point after being off-route
        df_work['_distance_boundary'] = (
            df_work['_off_route'] |  # Any off-route point
            (~df_work['_off_route'] & df_work['_prev_off_route'])  # First on-route after off-route
        )

        df_work['_new_segment'] = df_work['_new_segment'] | df_work['_distance_boundary']
        df_work = df_work.drop(columns=['_off_route', '_prev_off_route', '_distance_boundary'])

    # Create segment IDs by cumulative sum of boundaries
    df_work[segment_column] = df_work['_new_segment'].cumsum()

    # Drop temporary columns
    df_work = df_work.drop(columns=['_time_delta', '_vehicle_change', '_new_segment'])

    return df_work


def filter_segments_by_length(
    df: pd.DataFrame,
    segment_column: str,
    min_length: int
) -> pd.DataFrame:
    """
    Filter out segments that have fewer than min_length points.

    Args:
        df: DataFrame with segment data
        segment_column: Name of the column containing segment IDs
        min_length: Minimum number of points required to keep a segment

    Returns:
        DataFrame with short segments removed

    Raises:
        KeyError: If segment_column doesn't exist in the dataframe
        ValueError: If min_length is less than 1

    Example:
        >>> df = pd.DataFrame({
        ...     'segment_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        ...     'value': [10, 20, 30, 40, 50, 60, 70, 80, 90]
        ... })
        >>> result = filter_segments_by_length(df, 'segment_id', min_length=3)
        >>> # Segment 2 is removed (only 2 points)
        >>> result[['segment_id', 'value']]
           segment_id  value
        0           1     10
        1           1     20
        2           1     30
        5           3     60
        6           3     70
        7           3     80
        8           3     90
    """
    # Validation
    if segment_column not in df.columns:
        raise KeyError(f"Column '{segment_column}' not found in dataframe")
    if min_length < 1:
        raise ValueError(f"min_length must be at least 1, got {min_length}")

    # Count points in each segment
    segment_counts = df.groupby(segment_column).size()

    # Find segments that meet the minimum length requirement
    valid_segments = segment_counts[segment_counts >= min_length].index

    # Filter dataframe to keep only valid segments
    filtered_df = df[df[segment_column].isin(valid_segments)].copy()

    return filtered_df


def clean_closest_route(
    df: pd.DataFrame,
    route_column: str = 'route',
    polyline_idx_column: str = 'polyline_idx',
    segment_column: str = 'segment_id',
    window_size: int = 5,
    require_majority_valid: bool = False
) -> pd.DataFrame:
    """
    Fill NaN values in route and polyline_idx columns using majority vote from surrounding window.

    For each NaN entry within a segment, looks at surrounding rows within the window and fills
    with the most common (route, polyline_idx) pair if there's a clear majority. This helps
    clean up data where route assignment is ambiguous or failed for individual points but the
    surrounding context makes the route clear.

    Windows are constrained to stay within segment boundaries to avoid crossing segments
    where routes may legitimately change.

    This function uses vectorized operations for performance.

    Args:
        df: DataFrame with route, polyline_idx, and segment_id columns
        route_column: Name of column containing route names
        polyline_idx_column: Name of column containing polyline indices
        segment_column: Name of column containing segment IDs (default: 'segment_id')
        window_size: Number of rows to look before and after each NaN (default: 5)
        require_majority_valid: If True, does NOT require strict majority (>50%) of valid neighbors.
            This allows filling NaNs at segment endpoints (default: False).
            If False, requires strict majority, preventing endpoint fills but being more conservative.

    Returns:
        DataFrame with NaN route values filled based on majority vote from surrounding window

    Raises:
        KeyError: If route_column, polyline_idx_column, or segment_column doesn't exist
    """
    from scipy import stats

    # Validation
    if route_column not in df.columns:
        raise KeyError(f"Column '{route_column}' not found in dataframe")
    if polyline_idx_column not in df.columns:
        raise KeyError(f"Column '{polyline_idx_column}' not found in dataframe")
    if segment_column not in df.columns:
        raise KeyError(f"Column '{segment_column}' not found in dataframe")

    # Work on a copy
    df_clean = df.copy()

    # Find indices where route is NaN
    nan_mask = df_clean[route_column].isna()
    # Get positional indices of NaNs
    nan_indices = np.where(nan_mask)[0]
    total_nans = len(nan_indices)

    if total_nans == 0:
        return df_clean

    print(f"Cleaning {total_nans} NaN route values using window size {window_size} (Vectorized)...")

    # 1. Map (route, polyline_idx) to integer IDs to enable vectorized mode calculation
    # Filter for valid pairs to build the mapping
    valid_rows = df_clean[df_clean[route_column].notna()]

    if len(valid_rows) == 0:
        return df_clean

    # Get unique pairs
    unique_pairs = valid_rows[[route_column, polyline_idx_column]].drop_duplicates()
    unique_pairs = unique_pairs.reset_index(drop=True)
    # Assign a unique integer ID to each pair
    unique_pairs['pair_id'] = unique_pairs.index

    # Store lookup arrays for mapping back later
    pair_routes = unique_pairs[route_column].values
    pair_polys = unique_pairs[polyline_idx_column].values

    # Map original dataframe rows to these IDs
    # Use merge to map, but preserve order via a temporary index
    df_clean['_temp_sort_idx'] = np.arange(len(df_clean))

    merged = df_clean.merge(
        unique_pairs,
        on=[route_column, polyline_idx_column],
        how='left'
    )
    merged = merged.sort_values('_temp_sort_idx')

    # Extract numpy arrays for processing
    segment_ids = merged[segment_column].values
    pair_ids = merged['pair_id'].values  # float array (contains NaNs)

    # Clean up temp column
    df_clean = df_clean.drop(columns=['_temp_sort_idx'])

    # 2. Vectorized Window Construction
    # Create relative offsets: [-w, ..., -1, 1, ..., w]
    offsets = np.concatenate([np.arange(-window_size, 0), np.arange(1, window_size + 1)])

    # Broadcast to create matrix of neighbor indices for each NaN row
    # Shape: (N_nans, 2*w)
    neighbor_indices = nan_indices[:, None] + offsets[None, :]

    # 3. Handle Boundary Conditions and Validity
    num_rows = len(df_clean)

    # Clip indices to valid range [0, len-1] to safely access arrays
    # We will filter out the invalid logical indices using bounds_mask
    clamped_indices = np.clip(neighbor_indices, 0, num_rows - 1)

    # Retrieve data for neighbors
    neighbor_segments = segment_ids[clamped_indices]
    neighbor_pairs = pair_ids[clamped_indices]

    # Retrieve data for target rows (to check segment consistency)
    target_segments = segment_ids[nan_indices]

    # Create Validity Mask
    # 1. Index was physically within bounds?
    bounds_mask = (neighbor_indices >= 0) & (neighbor_indices < num_rows)
    # 2. Segment matches target row's segment?
    segment_mask = (neighbor_segments == target_segments[:, None])
    # 3. Neighbor has a valid pair ID (not NaN itself)?
    value_mask = ~np.isnan(neighbor_pairs)

    # Combined valid mask
    valid_mask = bounds_mask & segment_mask & value_mask

    # Apply mask: Set invalid entries to NaN so mode calculation ignores them
    neighbor_pairs_cleaned = np.where(valid_mask, neighbor_pairs, np.nan)

    # 4. Compute Majority Vote
    # mode returns mode and count. nan_policy='omit' ignores NaNs.
    # axis=1 performs it row-wise.
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='After omitting NaNs')
        mode_result = stats.mode(neighbor_pairs_cleaned, axis=1, nan_policy='omit')

    modes = mode_result.mode
    counts = mode_result.count

    # Handle different scipy versions output shapes
    if modes.ndim > 1:
        modes = modes.flatten()
        counts = counts.flatten()

    # Calculate number of valid voters per row
    valid_counts = np.sum(valid_mask, axis=1)

    # 5. Apply Threshold (> 50% of valid window)
    # We require a strict majority of the *valid* neighbors
    update_mask = (valid_counts > 0) & (counts > valid_counts / 2)

    # 6. Apply Updates
    if np.any(update_mask):
        # Get indices that met the criteria
        indices_to_update = nan_indices[update_mask]

        # Get the new IDs
        new_pair_ids = modes[update_mask].astype(int)

        # Map IDs back to route and polyline values
        new_routes = pair_routes[new_pair_ids]
        new_polys = pair_polys[new_pair_ids]

        # Update DataFrame using labels (handles non-RangeIndex safely)
        labels_to_update = df_clean.index[indices_to_update]
        df_clean.loc[labels_to_update, route_column] = new_routes
        df_clean.loc[labels_to_update, polyline_idx_column] = new_polys

        filled_count = len(indices_to_update)
        print(f"  ✓ Filled {filled_count}/{total_nans} NaN values ({filled_count/total_nans*100:.1f}%)")
    else:
        print(f"  ✓ Filled 0/{total_nans} NaN values (0.0%)")

    return df_clean


def add_closest_points_educated(
    df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    route_column: str,
    polyline_idx_column: str,
    output_columns: dict[str, str]
) -> None:
    """
    Add closest point details for rows that have a route/polyline but missing segment info.

    This is meant to be run after clean_closest_route. It finds rows where the route
    and polyline index are known (possibly filled by clean_closest_route) but the
    geometric details (segment index, exact closest point) are missing.

    For these rows, it calls Stops.get_closest_point with the specific target polyline,
    forcing a match to that route segment.

    Args:
        df: Pandas DataFrame to modify
        lat_column: Name of the latitude column
        lon_column: Name of the longitude column
        route_column: Name of the column containing route names
        polyline_idx_column: Name of the column containing polyline indices
        output_columns: Dictionary mapping return value names to output column names.
                       Must contain 'segment_index'.
                       Valid keys: 'distance', 'closest_point_lat', 'closest_point_lon',
                                  'segment_index', 'route_name', 'polyline_index'

    Raises:
        KeyError: If required columns don't exist
    """
    from shared.stops import Stops

    # Validation
    if lat_column not in df.columns:
        raise KeyError(f"Column '{lat_column}' not found in dataframe")
    if lon_column not in df.columns:
        raise KeyError(f"Column '{lon_column}' not found in dataframe")
    if route_column not in df.columns:
        raise KeyError(f"Column '{route_column}' not found in dataframe")
    if polyline_idx_column not in df.columns:
        raise KeyError(f"Column '{polyline_idx_column}' not found in dataframe")

    if 'segment_index' not in output_columns:
        raise KeyError("output_columns must contain 'segment_index' to identify missing data")

    segment_col = output_columns['segment_index']
    if segment_col not in df.columns:
        # If the column doesn't exist yet, we can't filter by it being NaN
        raise KeyError(f"Column '{segment_col}' not found in dataframe")

    # Filter rows: Route & Polyline present, Segment Index missing
    mask = (
        df[route_column].notna() &
        df[polyline_idx_column].notna() &
        df[segment_col].isna()
    )

    rows_to_process_indices = df.index[mask]

    if len(rows_to_process_indices) == 0:
        return

    print(f"Refining {len(rows_to_process_indices)} rows with educated route guesses...")

    def process_row(row):
        lat = row[lat_column]
        lon = row[lon_column]
        route = row[route_column]
        poly_idx = row[polyline_idx_column]

        # Stops.get_closest_point expects poly_idx as int
        try:
            poly_idx = int(poly_idx)
        except (ValueError, TypeError):
            result = {output_col: np.nan for output_col in output_columns.values()}
            return pd.Series(result)

        # Get closest point forced to the specific route/polyline
        distance, closest_point, _, _, segment_index = Stops.get_closest_point(
            (lat, lon),
            target_polyline=(route, poly_idx)
        )

        value_map = {
            'distance': distance,
            'closest_point_lat': closest_point[0] if closest_point is not None else np.nan,
            'closest_point_lon': closest_point[1] if closest_point is not None else np.nan,
            'route_name': route,
            'polyline_index': poly_idx,
            'segment_index': segment_index
        }

        result = {output_col: value_map.get(key, np.nan) for key, output_col in output_columns.items()}
        return pd.Series(result)

    # Apply to filtered rows
    results = df.loc[mask].progress_apply(process_row, axis=1)

    # Assign back to original dataframe
    for key, output_col in output_columns.items():
        if output_col in results.columns:
            df.loc[mask, output_col] = results[output_col]
