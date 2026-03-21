"""Stop-related functions for vehicle location data."""
import pandas as pd
import numpy as np
from tqdm import tqdm

tqdm.pandas()


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
        result_df = df.apply(process_row, axis=1)
        for output_col in output_columns.values():
            df[output_col] = result_df[output_col]


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
        result_df = df.apply(process_row, axis=1)
        for output_col in output_columns.values():
            df[output_col] = result_df[output_col]


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
    from shared.stops import Stops

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

    # Clean data frame if no unrecorded jumps found
    if len(unrecorded_jumps) == 0:
        print("   No unrecorded stop jumps found")
        df.drop(columns=['prev_route', 'prev_polyline_index', 'prev_stop', 'prev_lat', 'prev_lon', 'prev_distance'], inplace=True)
        return
    
    print(f"   Found {len(unrecorded_jumps)} unrecorded stop jumps")

    # Vectorized function to find matching stops
    def find_stop(row):
        """
        Find the stop name at the current polyline index.
        Uses the route's POLYLINE_STOPS list, where each index maps directly
        to a stop name. Returns the stop at idx (the current polyline
        index for this row).
        """
        route_name = row[route_column]
        idx = int(row[polyline_index_column])
        
        if route_name not in Stops.routes_data:
            return None
        
        route_data = Stops.routes_data[route_name]
        polyline_stops = route_data.get('POLYLINE_STOPS', [])
        
        # Find stops that fall between before and after indices
        if idx < len(polyline_stops):
            return polyline_stops[idx]
        
        return None
    
    df.loc[unrecorded_mask, 'matched_stop'] = df[unrecorded_mask].apply(
        find_stop, axis=1
    )

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

    print(f"   ✓ Assigned {stops_assigned} unrecorded stops")
    temp_cols = ['prev_route', 'prev_polyline_index', 'prev_stop', 'prev_distance',
                 'matched_stop', 'stop_to_assign_prev', 'stop_assignment']
    df.drop(columns=temp_cols, inplace=True, errors='ignore')