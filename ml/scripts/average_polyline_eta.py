#!/usr/bin/env python3
"""
Calculate average time to traverse each polyline based on stop-to-stop segments.

This script:
1. Loads data from the stops pipeline
2. Finds segments where the shuttle stops at more than one stop
3. Calculates the time difference between consecutive stops within each segment
4. Aggregates the average travel time for each polyline
5. Saves results to lstm/<route>_<idx>/ directories

The resulting data shows how long it typically takes to travel each polyline segment
between stops, which can be used for ETA prediction.
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml.pipelines import stops_pipeline
from ml.cache import get_polyline_dir

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_polyline_travel_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average travel time for each polyline based on stop-to-stop segments.

    Args:
        df: DataFrame from stops pipeline with columns:
            - segment_id: Segment identifier
            - timestamp: Timestamp of each point
            - stop_name: Name of stop (NaN if not at stop)
            - route: Route name
            - polyline_idx: Polyline index

    Returns:
        DataFrame with columns:
            - route: Route name
            - polyline_idx: Polyline index
            - avg_travel_time_seconds: Average time to traverse this polyline
            - std_travel_time_seconds: Standard deviation of travel times
            - sample_count: Number of observations
            - min_time: Minimum observed time
            - max_time: Maximum observed time
    """
    # Filter to only rows at stops
    stops_only = df[df['stop_name'].notna()].copy()

    logger.info(f"Found {len(stops_only)} points at stops across {stops_only['segment_id'].nunique()} segments")

    # Group by segment and count stops per segment
    stops_per_segment = stops_only.groupby('segment_id').size()
    segments_with_multiple_stops = stops_per_segment[stops_per_segment > 1].index

    logger.info(f"Found {len(segments_with_multiple_stops)} segments with multiple stops")

    # Filter to segments with multiple stops
    multi_stop_segments = stops_only[stops_only['segment_id'].isin(segments_with_multiple_stops)].copy()

    # Sort by segment and timestamp to get chronological order within each segment
    multi_stop_segments = multi_stop_segments.sort_values(['segment_id', 'timestamp'])

    # Calculate time differences between consecutive stops within each segment
    travel_times = []

    for segment_id, segment_data in multi_stop_segments.groupby('segment_id'):
        segment_data = segment_data.reset_index(drop=True)

        # Iterate through consecutive stop pairs
        for i in range(len(segment_data) - 1):
            start_stop = segment_data.iloc[i]
            end_stop = segment_data.iloc[i + 1]

            # Calculate time difference in seconds
            time_delta = (end_stop['timestamp'] - start_stop['timestamp']).total_seconds()

            # Only include positive time deltas (shouldn't happen, but just in case)
            if time_delta > 0:
                travel_times.append({
                    'segment_id': segment_id,
                    'route': start_stop['route'],
                    'polyline_idx': start_stop['polyline_idx'],
                    'start_stop': start_stop['stop_name'],
                    'end_stop': end_stop['stop_name'],
                    'travel_time_seconds': time_delta
                })

    if not travel_times:
        logger.warning("No valid travel times found!")
        return pd.DataFrame()

    travel_times_df = pd.DataFrame(travel_times)
    logger.info(f"Calculated {len(travel_times_df)} stop-to-stop travel times")

    # Group by route and polyline_idx to calculate statistics
    polyline_stats = travel_times_df.groupby(['route', 'polyline_idx']).agg(
        avg_travel_time_seconds=('travel_time_seconds', 'mean'),
        std_travel_time_seconds=('travel_time_seconds', 'std'),
        sample_count=('travel_time_seconds', 'count'),
        min_time=('travel_time_seconds', 'min'),
        max_time=('travel_time_seconds', 'max')
    ).reset_index()

    # Fill NaN std (happens when sample_count = 1) with 0
    polyline_stats['std_travel_time_seconds'] = polyline_stats['std_travel_time_seconds'].fillna(0)

    logger.info(f"Calculated statistics for {len(polyline_stats)} unique polylines")

    return polyline_stats


def save_polyline_stats(stats_df: pd.DataFrame):
    """
    Save polyline travel time statistics to appropriate directories.

    Args:
        stats_df: DataFrame with polyline statistics (output of calculate_polyline_travel_times)
    """
    saved_count = 0

    for _, row in stats_df.iterrows():
        route = row['route']
        polyline_idx = int(row['polyline_idx'])

        # Get the directory for this polyline
        polyline_dir = get_polyline_dir(route, polyline_idx)
        polyline_dir.mkdir(parents=True, exist_ok=True)

        # Create a single-row dataframe with this polyline's stats
        polyline_stats = pd.DataFrame([row])

        # Save to CSV
        output_path = polyline_dir / 'average_travel_time.csv'
        polyline_stats.to_csv(output_path, index=False)
        saved_count += 1

    logger.info(f"Saved statistics to {saved_count} polyline directories")


def main():
    """Main execution function."""
    logger.info("="*70)
    logger.info("AVERAGE POLYLINE TRAVEL TIME CALCULATION")
    logger.info("="*70)

    # Load data from stops pipeline
    logger.info("Loading data from stops pipeline...")
    df = stops_pipeline()
    logger.info(f"Loaded {len(df)} records")

    # Calculate polyline travel times
    logger.info("Calculating polyline travel times...")
    stats_df = calculate_polyline_travel_times(df)

    if len(stats_df) == 0:
        logger.error("No polyline statistics calculated. Exiting.")
        return 1

    # Display summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total unique polylines: {len(stats_df)}")
    logger.info(f"  Routes: {sorted(stats_df['route'].unique().tolist())}")
    logger.info(f"  Average travel time (all polylines): {stats_df['avg_travel_time_seconds'].mean():.1f} seconds")
    logger.info(f"  Total observations: {stats_df['sample_count'].sum()}")

    # Show top 10 polylines by sample count
    logger.info("\nTop 10 polylines by sample count:")
    top_polylines = stats_df.nlargest(10, 'sample_count')[
        ['route', 'polyline_idx', 'avg_travel_time_seconds', 'sample_count']
    ]
    for _, row in top_polylines.iterrows():
        logger.info(f"  {row['route']:5} polyline {int(row['polyline_idx']):2d}: "
                   f"{row['avg_travel_time_seconds']:6.1f}s (n={int(row['sample_count'])})")

    # Save results
    logger.info("\nSaving results to polyline directories...")
    save_polyline_stats(stats_df)

    logger.info("="*70)
    logger.info("COMPLETE")
    logger.info("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
