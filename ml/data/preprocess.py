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
    output_columns: dict[str, str]
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

    Modifies the dataframe in place by adding new columns based on output_columns mapping.

    Args:
        df: Pandas DataFrame to modify
        lat_column: Name of the latitude column
        lon_column: Name of the longitude column
        output_columns: Dictionary mapping return value names to output column names.
                       Valid keys: 'distance', 'closest_point_lat', 'closest_point_lon',
                                  'route_name', 'polyline_index'
                       Only specified keys will be added as columns.

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
        distance, closest_point, route_name, polyline_index = Stops.get_closest_point((lat, lon), ambiguous=True)

        # Build result dictionary with requested values
        value_map = {
            'distance': distance,
            'closest_point_lat': closest_point[0] if closest_point is not None else np.nan,
            'closest_point_lon': closest_point[1] if closest_point is not None else np.nan,
            'route_name': route_name,
            'polyline_index': polyline_index
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


if __name__ == "__main__":
    # Run individual preprocessing functions demo
    from ml.data.load import load_vehicle_locations

    print("Running preprocessing functions demo...")
    print("For complete pipeline, use: python -m ml.pipelines")
    print()

    # Load data
    print("Loading vehicle locations...")
    df = load_vehicle_locations(force_reload=False)
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
        'route_name': 'route'
    })
    print(f"   Added 'dist_to_route' and 'route' columns")

    print("3. Computing distance deltas...")
    distance_delta(sample_df, 'latitude', 'longitude', 'distance_km')
    print(f"   Added 'distance_km' column")

    print("4. Computing speed...")
    speed(sample_df, 'distance_km', 'epoch_seconds', 'speed_kmh')
    print(f"   Added 'speed_kmh' column")

    print()
    print("="*60)
    print("SAMPLE PREPROCESSED DATA")
    print("="*60)
    pd.options.display.float_format = '{:.3f}'.format
    print(sample_df[['vehicle_id', 'route', 'distance_km', 'speed_kmh', 'dist_to_route']].head(10))

    print()
    print("="*60)
    print("For complete pipeline with caching, run:")
    print("  python -m ml.pipelines")
    print("="*60)
