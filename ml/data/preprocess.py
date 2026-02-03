"""Preprocessing functions for vehicle location data."""
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()

# Epoch offset: 2025-01-01 00:00:00 UTC in Unix seconds
EPOCH_2025_OFFSET = int(datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc).timestamp())


def to_epoch_seconds(
    df: pd.DataFrame,
    input_column: str,
    output_column: str
) -> None:
    """
    Convert a datetime column to epoch seconds since 2025-01-01 00:00:00 UTC.

    Modifies the dataframe in place by adding a new column with epoch timestamps
    relative to the start of 2025. This produces smaller numbers that are easier
    to work with in machine learning models.

    Args:
        df: Pandas DataFrame to modify
        input_column: Name of the datetime column to convert
        output_column: Name of the new column to create with epoch seconds

    Raises:
        KeyError: If input_column doesn't exist in the dataframe
        TypeError: If input_column is not a datetime type

    Example:
        >>> df = pd.DataFrame({'timestamp': pd.to_datetime(['2025-01-01', '2025-01-02'])})
        >>> to_epoch_seconds(df, 'timestamp', 'epoch')
        >>> print(df)
                  timestamp     epoch
        0 2025-01-01            0.0
        1 2025-01-02        86400.0
    """
    if input_column not in df.columns:
        raise KeyError(f"Column '{input_column}' not found in dataframe")

    if not pd.api.types.is_datetime64_any_dtype(df[input_column]):
        raise TypeError(
            f"Column '{input_column}' must be datetime type, "
            f"got {df[input_column].dtype}"
        )

    # Convert datetime to Unix epoch seconds, then subtract 2025 offset
    df[output_column] = (df[input_column].astype('int64') / 1e9) - EPOCH_2025_OFFSET


def add_closest_points(
    df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    output_columns: dict[str, str],
    additive: bool = False
) -> None:
    """
    Add route information by finding the closest point on route polylines.

    For each GPS coordinate, uses the Stops.get_closest_point() function to find
    the nearest route polyline and adds the requested information as new columns.

    The get_closest_point function returns:
    - distance: Distance to the closest point on any route (km)
    - closest_point: The closest point coordinates [lat, lon]
    - route_name: Name of the closest route
    - polyline_index: Index of the polyline within that route
    - segment_index: Index of the segment within the polyline

    Modifies the dataframe in place by adding new columns based on output_columns mapping.

    Args:
        df: Pandas DataFrame to modify
        lat_column: Name of the latitude column
        lon_column: Name of the longitude column
        output_columns: Dictionary mapping return value names to output column names.
                       Valid keys: 'distance', 'closest_point_lat', 'closest_point_lon',
                                  'route_name', 'polyline_index'
                       Only specified keys will be added as columns.
        additive: If True, only process rows where output columns have NaN values.
                 Useful for incremental updates where some rows already have closest points.
                 (default: False)

    Raises:
        KeyError: If lat_column or lon_column doesn't exist in the dataframe

    Example:
        >>> df = pd.DataFrame({
        ...     'latitude': [42.7284, 42.7295],
        ...     'longitude': [-73.6788, -73.6799]
        ... })
        >>> add_closest_points(df, 'latitude', 'longitude', {
        ...     'distance': 'dist_to_route',
        ...     'route_name': 'route',
        ...     'closest_point_lat': 'closest_lat'
        ... })
        >>> # df now has columns: dist_to_route, route, closest_lat
    """
    # Import here to avoid circular imports
    from shared.stops import Stops

    # Validation
    if lat_column not in df.columns:
        raise KeyError(f"Column '{lat_column}' not found in dataframe")
    if lon_column not in df.columns:
        raise KeyError(f"Column '{lon_column}' not found in dataframe")

    if not output_columns:
        return  # Nothing to do

    # If additive mode, initialize columns if they don't exist and determine which rows need processing
    if additive:
        # Initialize output columns with NaN if they don't exist
        for output_col in output_columns.values():
            if output_col not in df.columns:
                df[output_col] = np.nan

        # Determine which rows need processing (any row with NaN in any output column)
        route_col = output_columns['route_name']
        rows_to_process_mask = df[route_col].isna()
        rows_to_process = df[rows_to_process_mask].copy()

        if len(rows_to_process) == 0:
            print(f'Additive mode: All {len(df)} rows already have closest points. Skipping.')
            return

        print(f'Additive mode: Processing {len(rows_to_process)}/{len(df)} rows with missing closest points')
    else:
        rows_to_process = df
        rows_to_process_mask = pd.Series([True] * len(df), index=df.index)

    def process_row(row):
        """Process a single row and return closest point information."""
        lat = row[lat_column]
        lon = row[lon_column]

        # Return NaN/None for all outputs if coordinates are invalid
        if pd.isna(lat) or pd.isna(lon):
            result = {}
            for key, output_col in output_columns.items():
                if key == 'route_name':
                    result[output_col] = None
                else:
                    result[output_col] = np.nan
            return pd.Series(result)

        # Get closest point information
        distance, closest_point, route_name, polyline_index, segment_index = Stops.get_closest_point((lat, lon))

        # Build result dictionary with requested values
        value_map = {
            'distance': distance,
            'closest_point_lat': closest_point[0] if closest_point is not None else np.nan,
            'closest_point_lon': closest_point[1] if closest_point is not None else np.nan,
            'route_name': route_name,
            'polyline_index': polyline_index,
            'segment_index': segment_index
        }
        result = {output_col: value_map[key] for key, output_col in output_columns.items()}

        return pd.Series(result)

    # Apply the function to rows that need processing
    result_df = rows_to_process.progress_apply(process_row, axis=1)
    print('Done! Adding columns to dataframe.')

    # Assign results back to the original dataframe for the processed rows
    for output_col in output_columns.values():
        df.loc[rows_to_process_mask, output_col] = result_df[output_col].values


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

    def process_row(row):
        """Process a single row and return stop information."""
        lat = row[lat_column]
        lon = row[lon_column]

        # Return None for all outputs if coordinates are invalid
        if pd.isna(lat) or pd.isna(lon):
            result = {output_col: None for output_col in output_columns.values()}
            return pd.Series(result)

        # Get stop information
        route_name, stop_name = Stops.is_at_stop((lat, lon), threshold=threshold)

        # Build result dictionary with requested values
        value_map = {
            'route_name': route_name,
            'stop_name': stop_name
        }
        result = {output_col: value_map[key] for key, output_col in output_columns.items()}

        return pd.Series(result)

    # Apply the function to each row and assign results
    if output_columns:  # Only process if there are columns to add
        result_df = df.progress_apply(process_row, axis=1)
        for output_col in output_columns.values():
            df[output_col] = result_df[output_col]

def clean_stops(
    df: pd.DataFrame,
    route_column: str,
    polyline_index_column: str,
    stop_column: str,
    distance_column: str
) -> None:
    """
    Rectifies unrecorded stops by identifying jumps in polyline indices without stop records.

    Essentially, if a shuttle were to pass a stop, but this instance was not recorded in the data,
    the data would only show the before and after positions, with a jump in polyline index. Consider
    which point is closer to the actual stop, and use that data point to record that the shuttle has
    visisted said stop.
    """
    # Import here to avoid circular imports
    from shared.stops import stops

    df['prev_route'] = df[route_column].shift(1)
    df['prev_polyline_index'] = df[polyline_index_column].shift(1)
    df['prev_stop'] = df[stop_column].shift(1)
    df['prev_distance'] = df[distance_column].shift(1)

    # Identify any jumps
    jumps_mask = (
        (df[route_column] == df['prev_route']) & # Same route?
        (df[polyline_index_column].notna()) & # Current index valid?
        (df['prev_polyline_index'].notna()) & # Previous index valid?
        (df[polyline_index_column] > df['prev_polyline_index']) # Polyline index increased?
    )

    # Filter for only unidentified stops with jumps
    unrecorded_mask = (
        jumps_mask &
        (df[stop_column].isna()) & # No current stop
        (df['prev_stop'].isna()) # No previous stop
    )

    unrecorded_jumps = df[unrecorded_mask]

    # Clean data frame
    if len(unrecorded_jumps) == 0:
        print("   No unrecorded stop jumps found")
        df.drop(columns=['prev_route', 'prev_polyline_index', 'prev_stop', 'prev_distance'], inplace=True)
        return
    
    print(f"   Found {len(unrecorded_jumps)} unrecorded stop jumps")


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

    def process_row(row):
        """Process a single row and return polyline distance information."""
        lat = row[lat_column]
        lon = row[lon_column]

        # Return NaN for all outputs if coordinates are invalid
        if pd.isna(lat) or pd.isna(lon):
            result = {output_col: np.nan for output_col in output_columns.values()}
            return pd.Series(result)

        # Construct closest_point_result if we have the data
        closest_point_result = None
        if has_closest_point_data:
            distance = row[distance_column]
            closest_lat = row[closest_point_lat_column]
            closest_lon = row[closest_point_lon_column]
            route_name = row[route_column]
            polyline_idx = row[polyline_index_column]
            segment_idx = row[segment_index_column]

            # Only use if all values are valid
            if not (pd.isna(distance) or pd.isna(closest_lat) or pd.isna(closest_lon) or
                    pd.isna(route_name) or pd.isna(polyline_idx) or pd.isna(segment_idx)):
                closest_coords = [closest_lat, closest_lon]
                closest_point_result = (distance, closest_coords, route_name, polyline_idx, segment_idx)

        # Get polyline distance information
        distance_from_start, distance_to_end, total_length = Stops.get_polyline_distances(
            (lat, lon),
            closest_point_result=closest_point_result
        )

        # Build result dictionary with requested values
        value_map = {
            'distance_from_start': distance_from_start,
            'distance_to_end': distance_to_end,
            'total_length': total_length
        }
        result = {output_col: value_map[key] for key, output_col in output_columns.items()}

        return pd.Series(result)

    # Apply the function to each row and assign results
    if output_columns:  # Only process if there are columns to add
        result_df = df.progress_apply(process_row, axis=1)
        for output_col in output_columns.values():
            df[output_col] = result_df[output_col]


def distance_delta(
    df: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    output_column: str
) -> pd.DataFrame:
    """
    Compute the distance traveled between consecutive GPS points.

    For each point, computes the great-circle distance from the previous point
    using the Haversine formula. The first row will have NaN since there's no
    previous point.

    Modifies the dataframe in place by adding a new column with distances in kilometers.

    Args:
        df: Pandas DataFrame to modify
        lat_column: Name of the latitude column
        lon_column: Name of the longitude column
        output_column: Name of the new column to create (default: 'distance_delta')

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # Import here to avoid circular imports
    from shared.stops import haversine_vectorized

    # Validation
    if lat_column not in df.columns:
        raise KeyError(f"Column '{lat_column}' not found in dataframe")
    if lon_column not in df.columns:
        raise KeyError(f"Column '{lon_column}' not found in dataframe")

    if len(df) == 0:
        df[output_column] = []
        return df

    # Create coordinate arrays: [lat, lon]
    coords = df[[lat_column, lon_column]].values

    # Shift coordinates to get previous position
    coords_prev = np.roll(coords, 1, axis=0)

    # Compute distances between consecutive points
    distances = haversine_vectorized(coords_prev, coords)

    # Set first distance to NaN (no previous point)
    distances[0] = np.nan

    # Add to dataframe
    df[output_column] = distances
    return df


def speed(
    df: pd.DataFrame,
    distance_column: str,
    time_column: str,
    output_column: str,
) -> pd.DataFrame:
    """
    Compute speed from distance and time deltas, with outlier filtering.

    Calculates speed by dividing distance by the time difference between consecutive
    points.

    Speed = distance / time_delta

    Modifies the dataframe in place by adding a new column with speed values.

    Args:
        df: Pandas DataFrame to modify
        distance_column: Name of the column containing distances (e.g., from distance_delta)
        time_column: Name of the column containing time values in seconds
        output_column: Name of the new column to create with speed values

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # Validation
    if distance_column not in df.columns:
        raise KeyError(f"Column '{distance_column}' not found in dataframe")
    if time_column not in df.columns:
        raise KeyError(f"Column '{time_column}' not found in dataframe")

    if len(df) == 0:
        df[output_column] = []
        return df

    # Get time values and compute deltas
    time_values = df[time_column].values
    time_deltas = np.diff(time_values, prepend=np.nan)

    # Get distance values
    distances = df[distance_column].values

    # Compute speed: distance / time_delta
    # Handle division by zero by setting to NaN where time_delta is 0
    with np.errstate(divide='ignore', invalid='ignore'):
        speeds = distances / time_deltas
        # Convert km/s to km/h by multiplying by 3600
        speeds = speeds * 3600

    # Set speed to NaN where time_delta is zero or negative
    speeds = np.where((time_deltas <= 0) | np.isnan(time_deltas), np.nan, speeds)

    # Add to dataframe
    df[output_column] = speeds
    return df


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


def filter_rows_after_stop(
    df: pd.DataFrame,
    segment_column: str,
    stop_column: str
) -> pd.DataFrame:
    """
    Filter out rows that occur after and including the last stop in each segment.

    Rows after and including the last stop cannot have accurate ETAs because there is no
    "next stop" to calculate the ETA to. This function removes those rows
    to ensure all remaining rows have valid ETA targets.

    Note: This function also implicitly removes segments that have no stops at all,
    since all rows in such segments are considered "after the last stop" (which
    doesn't exist).

    Args:
        df: DataFrame with segment and stop data
        segment_column: Name of the column containing segment IDs
        stop_column: Name of the column containing stop names (None/NaN when not at stop)

    Returns:
        DataFrame with:
        - Rows after the last stop in each segment removed
        - Segments without any stops completely removed

    Raises:
        KeyError: If segment_column or stop_column doesn't exist in the dataframe

    Example:
        >>> df = pd.DataFrame({
        ...     'segment_id': [1, 1, 1, 1, 1],
        ...     'stop': [None, 'Stop A', None, 'Stop B', None],
        ...     'value': [10, 20, 30, 40, 50]
        ... })
        >>> result = filter_rows_after_stop(df, 'segment_id', 'stop')
        >>> # Last row is removed (after Stop B)
        >>> result[['segment_id', 'stop', 'value']]
           segment_id     stop  value
        0           1     None     10
        1           1  Stop A     20
        2           1     None     30
        3           1  Stop B     40
    """
    # Validation
    if segment_column not in df.columns:
        raise KeyError(f"Column '{segment_column}' not found in dataframe")
    if stop_column not in df.columns:
        raise KeyError(f"Column '{stop_column}' not found in dataframe")

    # 1. Identify indices of rows where a stop occurs
    # We use the dataframe's index directly
    stops_mask = df[stop_column].notna()

    # 2. Find the max index (last stop) for each segment
    # Group the indices of stops by segment and find the max
    # We use df.index.to_series() to treat the index as values we can aggregate
    last_stop_indices = df.index.to_series()[stops_mask].groupby(df.loc[stops_mask, segment_column]).max()

    # 3. Map the last stop index back to the original dataframe
    # This creates a Series aligned with df, containing the cut-off index for each row's segment
    # Segments with no stops will have NaN, which correctly results in False during comparison
    limit_indices = df[segment_column].map(last_stop_indices)
    # add limit_indices to df for debugging
    df['limit_indices'] = limit_indices

    # 4. Filter: Keep rows where the current index is < the last stop index for that segment
    # This vectorizes the "rows after last stop" check
    keep_mask = df.index <= limit_indices

    # Filter dataframe
    filtered_df = df[keep_mask].copy()

    return filtered_df


def split_by_route_polyline_index(
    df: pd.DataFrame,
    route_column: str = 'route',
    polyline_index_column: str = 'polyline_idx'
) -> dict[tuple[str, int], pd.DataFrame]:
    """
    Split a dataframe into multiple dataframes, one for each unique (route, polyline_index).

    Args:
        df: DataFrame with route and polyline index data
        route_column: Name of the column containing route names
        polyline_index_column: Name of the column containing polyline indices

    Returns:
        Dictionary mapping (route, polyline_index) tuples to their corresponding DataFrames

    Raises:
        KeyError: If required columns don't exist in the dataframe
    """
    # Validation
    if route_column not in df.columns:
        raise KeyError(f"Column '{route_column}' not found in dataframe")
    if polyline_index_column not in df.columns:
        raise KeyError(f"Column '{polyline_index_column}' not found in dataframe")

    # Group by route and polyline index
    grouped = df.groupby([route_column, polyline_index_column])

    # Create a dictionary of dataframes
    split_dfs = {name: group.copy() for name, group in grouped}

    return split_dfs


def add_eta(
    df: pd.DataFrame,
    stop_column: str,
    time_column: str,
    output_column: str
) -> pd.DataFrame:
    """
    Add ETA (estimated time of arrival) column showing time until next stop arrival.

    For each data point, calculates the time (in seconds) until the vehicle arrives
    at the next stop. Properly handles multiple vehicles by grouping by vehicle_id.

    Uses efficient pandas built-in functions for vectorized operations.

    Args:
        df: DataFrame with vehicle location data
        stop_column: Name of the column containing stop names (None/NaN when not at stop)
        time_column: Name of the column containing time values (in seconds, e.g., epoch_seconds)
        output_column: Name of the new column to create with ETA values

    Returns:
        DataFrame with added ETA column (NaN if no next stop exists)

    Raises:
        KeyError: If required columns don't exist in the dataframe

    Example:
        >>> df = pd.DataFrame({
        ...     'vehicle_id': [1, 1, 1, 1, 2, 2, 2],
        ...     'epoch_seconds': [0, 10, 20, 30, 0, 15, 30],
        ...     'stop': [None, None, 'Stop A', None, None, 'Stop B', None]
        ... })
        >>> result = add_eta(df, 'stop', 'epoch_seconds', 'eta_seconds')
        >>> # Row 0 (t=0) has eta=20 (arrives at Stop A at t=20)
        >>> # Row 1 (t=10) has eta=10 (arrives at Stop A at t=20)
        >>> # Row 2 (t=20) is at stop, eta=20 (next stop at t=40)
        >>> # Row 3 (t=30) has eta=10 (arrives at next stop at t=40)
    """
    # Validation
    if 'vehicle_id' not in df.columns:
        raise KeyError("Column 'vehicle_id' not found in dataframe")
    if stop_column not in df.columns:
        raise KeyError(f"Column '{stop_column}' not found in dataframe")
    if time_column not in df.columns:
        raise KeyError(f"Column '{time_column}' not found in dataframe")

    # Work with a copy
    df_work = df.copy()

    # Initialize ETA column with NaN
    df_work[output_column] = np.nan

    # Create a helper column to identify stops
    df_work['_is_at_stop'] = df_work[stop_column].notna()

    def calculate_eta_for_vehicle(group):
        """Calculate ETA for a single vehicle using vectorized operations."""
        # Get indices where vehicle is at stops
        stop_mask = group['_is_at_stop']

        if not stop_mask.any():
            # No stops, return group with NaN ETAs
            return group

        # Get stop times as a sorted array
        stop_times = group.loc[stop_mask, time_column].values
        stop_times_sorted = np.sort(stop_times)

        # For each row, find the next stop time using searchsorted
        current_times = group[time_column].values

        # searchsorted with side='right' gives us the index of the first element > current_time
        next_stop_indices = np.searchsorted(stop_times_sorted, current_times, side='right')

        # Get the next stop times (NaN if no next stop exists)
        next_stop_time_values = np.full(len(group), np.nan)
        valid_mask = next_stop_indices < len(stop_times_sorted)
        next_stop_time_values[valid_mask] = stop_times_sorted[next_stop_indices[valid_mask]]

        # Calculate ETA: next_stop_time - current_time
        group[output_column] = next_stop_time_values - current_times

        return group

    # Apply to each vehicle, segment group
    df_work = df_work.groupby(['vehicle_id', 'segment_id'], group_keys=False).apply(
        calculate_eta_for_vehicle
    )

    # Clean up helper column
    df_work = df_work.drop(columns=['_is_at_stop'])

    return df_work


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


if __name__ == "__main__":
    # Run individual preprocessing functions demo
    from ml.data.load import load_vehicle_locations

    print("Running preprocessing functions demo...")
    print("For complete pipeline, use: python -m ml.pipelines")
    print()

    # Load data
    print("Loading vehicle locations...")
    df = load_vehicle_locations()
    print(f"Loaded {len(df)} records")
    print()

    # Demo: Apply individual preprocessing functions
    print("Demo: Applying individual preprocessing functions...")
    print("-"*60)

    # Take a small sample for demo
    sample_df = df.head(1000).copy()

    print("1. Converting timestamps to epoch seconds...")
    to_epoch_seconds(sample_df, 'timestamp', 'epoch_seconds')
    print(f"   Added 'epoch_seconds' column")

    print("2. Adding closest route points...")
    add_closest_points(sample_df, 'latitude', 'longitude', {
        'distance': 'dist_to_route',
        'route_name': 'route',
        'polyline_index': 'polyline_idx'
    })
    print(f"   Added 'dist_to_route', 'route', and 'polyline_idx' columns")

    print("3. Computing distance deltas...")
    distance_delta(sample_df, 'latitude', 'longitude', 'distance_km')
    print(f"   Added 'distance_km' column")

    print("4. Computing speed...")
    speed(sample_df, 'distance_km', 'epoch_seconds', 'speed_kmh')
    print(f"   Added 'speed_kmh' column")

    print("5. Cleaning closest route (Vectorized)...")
    # Setup for cleaning: ensure segment_id exists
    sample_df['segment_id'] = 1

    # Assign back (function returns copy)
    sample_df = clean_closest_route(sample_df, 'route', 'polyline_idx', 'segment_id')
    print(f"   Cleaned NaNs")

    print()
    print("="*60)
    print("SAMPLE PREPROCESSED DATA")
    print("="*60)
    pd.options.display.float_format = '{:.3f}'.format
    cols = ['vehicle_id', 'route', 'distance_km', 'speed_kmh', 'dist_to_route']
    print(sample_df[cols].head(10))

    print()
    print("="*60)
    print("For complete pipeline with caching, run:")
    print("  python -m ml.pipelines")
    print("="*60)
