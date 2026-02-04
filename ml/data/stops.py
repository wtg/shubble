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
        result_df = df.progress_apply(process_row, axis=1)
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
        result_df = df.progress_apply(process_row, axis=1)
        for output_col in output_columns.values():
            df[output_col] = result_df[output_col]
