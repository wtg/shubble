"""Extract optimal break-detection training data from the production database.

Pulls vehicle_locations for weekday service hours across the last N days,
computes per-shuttle Union visit cadence, identifies breaks and stay-points,
and writes a calibration report + CSV for offline analysis.

The output calibrates three constants in break_detection.py:
  - CUSUM_MIN_FIRE_MIN  (fleet-wide P95 of normal inter-Union intervals)
  - STAY_POINT_MIN_DWELL_SEC  (min dwell to avoid FPs from traffic stops)
  - CUSUM_SLACK_MIN / CUSUM_THRESHOLD_MIN  (CUSUM sensitivity)

Run:
  # Against production DB (requires DATABASE_URL in .env):
  python -m ml.extract_break_data

  # Against the historical CSV (no DB needed):
  python -m ml.extract_break_data --csv
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CAMPUS_TZ = "America/New_York"
UNION_LAT = 42.730711
UNION_LON = -73.676737
UNION_RADIUS_M = 75.0
DWELL_GAP_SEC = 120

# How many days of history to pull. More data = better P95 estimates.
# 60 days covers ~8 weeks of weekday patterns.
LOOKBACK_DAYS = 60

# Service hours in campus local time (inclusive).
SERVICE_START_HOUR = 7
SERVICE_END_HOUR = 22

# Break detection: gap between consecutive Union visits exceeding this
# (in minutes) during 10:00-14:00 local is a confirmed break.
BREAK_GAP_MIN = 30
BREAK_WINDOW_START_HOUR = 10
BREAK_WINDOW_END_HOUR = 14


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * math.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _compress_dwells(times: List[datetime]) -> List[datetime]:
    if not times:
        return []
    times = sorted(times)
    out = [times[0]]
    for t in times[1:]:
        if (t - out[-1]).total_seconds() > DWELL_GAP_SEC:
            out.append(t)
    return out


# --─ DB extraction --------------------------------------------------─


async def _fetch_from_db(days: int) -> pd.DataFrame:
    """Pull vehicle_locations from the production database.

    Filters:
      - Last `days` days
      - Weekdays only (Mon-Fri, computed server-side via EXTRACT(dow))
      - Service hours only (7 AM - 10 PM campus local)

    Returns a DataFrame with columns: vehicle_id, latitude, longitude, timestamp (UTC).
    """
    from backend.database import create_async_db_engine, create_session_factory
    from sqlalchemy import text

    engine = create_async_db_engine()
    session_factory = create_session_factory(engine)

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    query = text("""
        SELECT vehicle_id, latitude, longitude, timestamp
        FROM vehicle_locations
        WHERE timestamp >= :cutoff
          AND EXTRACT(dow FROM timestamp AT TIME ZONE 'America/New_York')
              BETWEEN 1 AND 5  -- Mon=1 .. Fri=5
          AND EXTRACT(hour FROM timestamp AT TIME ZONE 'America/New_York')
              BETWEEN :start_hour AND :end_hour
        ORDER BY vehicle_id, timestamp
    """)

    async with session_factory() as session:
        result = await session.execute(query, {
            "cutoff": cutoff,
            "start_hour": SERVICE_START_HOUR,
            "end_hour": SERVICE_END_HOUR,
        })
        rows = result.all()

    await engine.dispose()

    df = pd.DataFrame(rows, columns=["vehicle_id", "latitude", "longitude", "timestamp"])
    df["vehicle_id"] = df["vehicle_id"].astype(str)
    return df


def _load_from_csv() -> pd.DataFrame:
    csv_path = Path(__file__).parent / "cache" / "shared" / "locations_raw.csv"
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df["vehicle_id"] = df["vehicle_id"].astype(str)

    df["ts_utc"] = df["timestamp"].dt.tz_localize("UTC")
    df["ts_local"] = df["ts_utc"].dt.tz_convert(CAMPUS_TZ)

    # Filter to weekdays + service hours to match DB query
    mask = (
        (df["ts_local"].dt.weekday < 5)
        & (df["ts_local"].dt.hour >= SERVICE_START_HOUR)
        & (df["ts_local"].dt.hour <= SERVICE_END_HOUR)
    )
    return df.loc[mask, ["vehicle_id", "latitude", "longitude", "timestamp"]].copy()


# --─ Analysis --------------------------------------------------------


def analyze(df: pd.DataFrame) -> None:
    """Run the full calibration analysis and print recommendations."""
    logger.info(f"Analyzing {len(df):,} pings, {df['vehicle_id'].nunique()} vehicles")

    if "ts_utc" not in df.columns:
        df["ts_utc"] = df["timestamp"].dt.tz_localize("UTC") if df["timestamp"].dt.tz is None else df["timestamp"]
        df["ts_local"] = df["ts_utc"].dt.tz_convert(CAMPUS_TZ)
    df["local_date"] = df["ts_local"].dt.date
    df["dow"] = df["ts_local"].dt.weekday

    # Filter to Union pings
    dists = _haversine_m(
        df["latitude"].to_numpy(), df["longitude"].to_numpy(),
        UNION_LAT, UNION_LON,
    )
    union = df.loc[dists <= UNION_RADIUS_M].copy()
    logger.info(f"Union pings: {len(union):,}")

    # -- 1. Inter-Union-visit intervals --

    all_intervals: List[float] = []
    break_intervals: List[float] = []
    normal_intervals: List[float] = []

    groups = union.groupby(["vehicle_id", "local_date"], sort=False)
    for (vid, d), g in groups:
        visits = sorted(g["ts_local"].tolist())
        compressed = _compress_dwells(visits)
        if len(compressed) < 2:
            continue
        for i in range(1, len(compressed)):
            iv_min = (compressed[i] - compressed[i - 1]).total_seconds() / 60.0
            hour = compressed[i - 1].hour
            all_intervals.append(iv_min)
            if iv_min >= BREAK_GAP_MIN and BREAK_WINDOW_START_HOUR <= hour < BREAK_WINDOW_END_HOUR:
                break_intervals.append(iv_min)
            elif iv_min < BREAK_GAP_MIN:
                normal_intervals.append(iv_min)

    iv = np.array(normal_intervals)

    print("\n" + "=" * 60)
    print("BREAK-DETECTION CALIBRATION REPORT")
    print("=" * 60)

    print(f"\nData: {len(df):,} pings, {df['local_date'].nunique()} days, "
          f"{df['vehicle_id'].nunique()} vehicles")
    print(f"Union visits (after dwell compression): "
          f"{len(all_intervals):,} intervals")
    print(f"  Normal (<{BREAK_GAP_MIN}min): {len(normal_intervals):,}")
    print(f"  Break (>={BREAK_GAP_MIN}min, {BREAK_WINDOW_START_HOUR}-{BREAK_WINDOW_END_HOUR} local): "
          f"{len(break_intervals)}")

    # -- 2. Normal cadence statistics -> CUSUM_MIN_FIRE_MIN --

    print("\n-- Normal service cadence --")
    for label, pct in [("P50", 50), ("P75", 75), ("P90", 90),
                       ("P95", 95), ("P99", 99), ("Max", 100)]:
        val = np.percentile(iv, pct) if pct < 100 else iv.max()
        print(f"  {label}: {val:.1f} min")

    recommended_floor = round(np.percentile(iv, 97) + 2, 0)
    print(f"\n  -> Recommended CUSUM_MIN_FIRE_MIN: {recommended_floor:.0f} min")
    print(f"    (P97 + 2min margin = {np.percentile(iv, 97):.1f} + 2)")

    # -- 3. Break gap precision at various thresholds --

    print("\n-- Break detection precision by gap threshold --")
    print(f"  (within {BREAK_WINDOW_START_HOUR}:00-{BREAK_WINDOW_END_HOUR}:00 local)")
    print(f"  {'threshold':>10} {'gaps':>6} {'breaks':>7} {'precision':>10}")

    n_breaks = len(break_intervals)
    for thresh in [15, 20, 25, 30, 35, 38, 40, 45]:
        n_above = sum(1 for iv_min in all_intervals
                      if iv_min >= thresh)
        prec = n_breaks / n_above if n_above > 0 else 0
        print(f"  {thresh:>8}m {n_above:>6} {n_breaks:>7} {prec:>9.1%}")

    # -- 4. Stay-point dwell distribution (non-Union stationary events) --

    print("\n-- Non-Union stay-points (for STAY_POINT_MIN_DWELL_SEC tuning) --")

    # Find stationary clusters NOT at Union
    stay_durations_break: List[float] = []
    stay_durations_nonbreak: List[float] = []

    for (vid, d), g in df.groupby(["vehicle_id", "local_date"], sort=False):
        pings = g.sort_values("ts_local")
        if len(pings) < 5:
            continue

        # Simple stationary detector: consecutive pings within 75m
        lats = pings["latitude"].to_numpy()
        lons = pings["longitude"].to_numpy()
        times = pings["ts_local"].tolist()

        i = 0
        while i < len(pings) - 1:
            anchor_lat, anchor_lon = lats[i], lons[i]
            # Skip if at Union
            if _haversine_m(
                np.array([anchor_lat]), np.array([anchor_lon]),
                UNION_LAT, UNION_LON,
            )[0] <= UNION_RADIUS_M:
                i += 1
                continue

            # Walk forward while within 75m
            j = i + 1
            while j < len(pings):
                d_m = _haversine_m(
                    np.array([lats[j]]), np.array([anchor_lon]),
                    anchor_lat, anchor_lon,
                )
                # Use scalar haversine properly
                dx = 6371000.0 * 2 * math.asin(math.sqrt(
                    math.sin(math.radians(lats[j] - anchor_lat) / 2) ** 2
                    + math.cos(math.radians(anchor_lat))
                    * math.cos(math.radians(lats[j]))
                    * math.sin(math.radians(lons[j] - anchor_lon) / 2) ** 2
                ))
                if dx > 75.0:
                    break
                j += 1

            duration_min = (times[j - 1] - times[i]).total_seconds() / 60.0
            if duration_min >= 3.0:
                hour = times[i].hour
                is_break_window = BREAK_WINDOW_START_HOUR <= hour < BREAK_WINDOW_END_HOUR

                # Check if this is during an actual Union-gap break
                union_pings_day = union[
                    (union["vehicle_id"] == vid)
                    & (union["local_date"] == d)
                ]
                union_visits = sorted(union_pings_day["ts_local"].tolist())
                compressed = _compress_dwells(union_visits)
                is_during_break = False
                for k in range(len(compressed) - 1):
                    gap = (compressed[k + 1] - compressed[k]).total_seconds() / 60.0
                    if gap >= BREAK_GAP_MIN and compressed[k] <= times[i] <= compressed[k + 1]:
                        is_during_break = True
                        break

                if is_during_break:
                    stay_durations_break.append(duration_min)
                elif is_break_window:
                    stay_durations_nonbreak.append(duration_min)

            i = j

    if stay_durations_break:
        sb = np.array(stay_durations_break)
        print(f"  During confirmed breaks: {len(sb)} stay-points")
        print(f"    Min: {sb.min():.1f}  P25: {np.percentile(sb, 25):.1f}  "
              f"Median: {np.median(sb):.1f}  P75: {np.percentile(sb, 75):.1f}")

    if stay_durations_nonbreak:
        sn = np.array(stay_durations_nonbreak)
        print(f"  During non-break periods ({BREAK_WINDOW_START_HOUR}-{BREAK_WINDOW_END_HOUR}): "
              f"{len(sn)} stay-points (potential FPs)")
        print(f"    Min: {sn.min():.1f}  P25: {np.percentile(sn, 25):.1f}  "
              f"Median: {np.median(sn):.1f}  P75: {np.percentile(sn, 75):.1f}")

        # Optimal dwell threshold: minimize FPs while keeping most true breaks
        if stay_durations_break:
            print(f"\n  Dwell threshold sweep:")
            print(f"  {'thresh':>8} {'break_recall':>13} {'fp_count':>9} {'fp_rate':>8}")
            for t in [3, 5, 7, 10, 15]:
                recall = sum(1 for d in stay_durations_break if d >= t) / len(stay_durations_break)
                fp = sum(1 for d in stay_durations_nonbreak if d >= t)
                print(f"  {t:>6}min {recall:>12.0%} {fp:>9} {fp/max(1,len(sn)):>7.0%}")

    # -- 5. Summary recommendations --

    print("\n-- RECOMMENDATIONS --")
    print(f"  CUSUM_MIN_FIRE_MIN = {recommended_floor:.0f}  "
          f"(P97 of normal intervals + 2min)")

    if stay_durations_nonbreak and stay_durations_break:
        # Pick threshold where FP rate drops below 20%
        best_thresh = 5
        for t in [5, 7, 10]:
            fp_rate = sum(1 for d in stay_durations_nonbreak if d >= t) / len(stay_durations_nonbreak)
            recall = sum(1 for d in stay_durations_break if d >= t) / len(stay_durations_break)
            if fp_rate < 0.20 and recall > 0.80:
                best_thresh = t
                break
        print(f"  STAY_POINT_MIN_DWELL_SEC = {best_thresh * 60}  "
              f"({best_thresh}min — best recall/FP balance)")

    print(f"\n  Copy to backend/fastapi/break_detection.py and redeploy.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Extract break-detection calibration data")
    parser.add_argument("--csv", action="store_true",
                        help="Use historical CSV instead of DB")
    parser.add_argument("--days", type=int, default=LOOKBACK_DAYS,
                        help=f"Days of history to pull (default: {LOOKBACK_DAYS})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.csv:
        df = _load_from_csv()
    else:
        df = asyncio.run(_fetch_from_db(args.days))

    analyze(df)


if __name__ == "__main__":
    main()
