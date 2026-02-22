#!/usr/bin/env python3

"""
Calculate average time a vehicle is stopped at a stop.

This script:
1. Loads data from the stops pipeline
2. Identifies stop visits using segment structure and stop_name
3. Calculates average dwell time per stop
4. Saves results to lstm/<route>_<polyline_idx>/ directories

The resulting data can be combined with polyline travel times to compute ETAs.
"""

import logging
import sys
from pathlib import Path
import pandas as pd

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


def calculate_stop_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average stop time for each stop.

    Args:
        df: DataFrame from stops_pipeline with columns:
            - segment_id
            - timestamp
            - stop_name
            - route
            - polyline_idx

    Returns:
        DataFrame with columns:
            - route
            - polyline_idx
            - stop_name
            - avg_stop_time_seconds
            - std_stop_time_seconds
            - sample_count
            - min_time
            - max_time
    """

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Only consider rows where the shuttle is at a stop
    stops_df = df[df['stop_name'].notna()].copy()
    if stops_df.empty:
        logger.warning("No stop data found!")
        return pd.DataFrame()

    # Sort by segment and timestamp
    stops_df = stops_df.sort_values(['segment_id', 'timestamp'])

    # Define new stop visit whenever segment changes or stop_name changes
    stops_df['new_visit'] = (
        (stops_df['segment_id'] != stops_df['segment_id'].shift()) |
        (stops_df['stop_name'] != stops_df['stop_name'].shift())
    )
    stops_df['stop_visit_id'] = stops_df['new_visit'].cumsum()

    # Compute duration per visit
    visit_durations = (
        stops_df
        .groupby(['route', 'polyline_idx', 'stop_name', 'stop_visit_id'])['timestamp']
        .agg(['min', 'max'])
        .reset_index()
    )
    visit_durations['duration'] = (visit_durations['max'] - visit_durations['min']).dt.total_seconds()

    # Remove zero-duration visits
    visit_durations = visit_durations[visit_durations['duration'] > 0]

    if visit_durations.empty:
        logger.warning("No valid stop durations found!")
        return pd.DataFrame()

    # Aggregate statistics per stop
    stop_stats = (
        visit_durations
        .groupby(['route', 'polyline_idx', 'stop_name'])['duration']
        .agg(
            avg_stop_time_seconds=lambda x: x.replace(0, 5).mean(),
            std_stop_time_seconds='std',
            sample_count='count',
            min_time=lambda x: x.replace(0, 5).min(),
            max_time='max'
        )
        .reset_index()
    )
    # Fill NaN std (occurs when sample_count = 1)
    stop_stats['std_stop_time_seconds'] = stop_stats['std_stop_time_seconds'].fillna(0)

    logger.info(f"Calculated stop times for {len(stop_stats)} unique stops")
    return stop_stats


def save_stop_stats(stats_df: pd.DataFrame):
    """
    Save stop time statistics to polyline directories.

    Args:
        stats_df: DataFrame with stop statistics
    """
    if stats_df.empty:
        logger.warning("No stop statistics to save")
        return

    saved_count = 0
    for _, row in stats_df.iterrows():
        route = row['route']
        polyline_idx = int(row['polyline_idx'])

        polyline_dir = get_polyline_dir(route, polyline_idx)
        polyline_dir.mkdir(parents=True, exist_ok=True)

        stop_stats = pd.DataFrame([row])
        output_path = polyline_dir / 'average_stop_time.csv'
        stop_stats.to_csv(output_path, index=False)
        saved_count += 1

    logger.info(f"Saved statistics to {saved_count} polyline directories")


def main():
    logger.info("="*70)
    logger.info("AVERAGE STOP TIME CALCULATION")
    logger.info("="*70)

    # Load data from stops pipeline
    logger.info("Loading data from stops pipeline...")
    df = stops_pipeline()
    logger.info(f"Loaded {len(df)} records")

    # Calculate stop times
    logger.info("Calculating stop times...")
    stats_df = calculate_stop_times(df)

    if stats_df.empty:
        logger.error("No stop statistics calculated. Exiting.")
        return 1

    # Display summary
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total unique stops: {len(stats_df)}")
    logger.info(f"  Routes: {sorted(stats_df['route'].unique().tolist())}")
    logger.info(f"  Total observations: {stats_df['sample_count'].sum()}")

    # Top 10 stops by sample count
    logger.info("\nTop 10 stops by sample count:")
    top_stops = stats_df.nlargest(10, 'sample_count')[
        ['route', 'polyline_idx', 'stop_name', 'avg_stop_time_seconds', 'sample_count']
    ]
    for _, row in top_stops.iterrows():
        logger.info(f"  {row['route']:5} polyline {int(row['polyline_idx']):2d} stop {row['stop_name']}: "
                    f"{row['avg_stop_time_seconds']:.1f}s (n={int(row['sample_count'])})")

    # Save results
    logger.info("\nSaving results to polyline directories...")
    save_stop_stats(stats_df)

    logger.info("="*70)
    logger.info("COMPLETE")
    logger.info("="*70)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())