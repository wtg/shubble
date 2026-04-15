"""Offline archetype extractor for Phase 3 break detection.

Reads historical GPS pings (ml/cache/shared/locations_raw.csv), identifies
per-shuttle Union-visit rhythm, detects break windows (>=40-min gap
between consecutive Union visits), and clusters per day-of-week into
archetypes. Writes shared/archetypes/<dow>.json.

Each archetype captures:
  break_start     — cluster-center minute-of-day (campus tz)
  median_gap_min  — median observed break length for cluster members
  signature       — mean of member morning Union-departure times (minutes
                    since 07:00), padded/truncated to SIGNATURE_LEN.

Production matcher loads these and performs Hungarian assignment between
today's active shuttles and archetypes to predict each shuttle's break
time.

Run:  python -m ml.build_archetypes
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

logger = logging.getLogger(__name__)


_CSV_PATH = Path(__file__).parent / "cache" / "shared" / "locations_raw.csv"
_OUT_DIR = Path(__file__).parent.parent / "shared" / "archetypes"

# Student Union coordinate (shared between NORTH and WEST routes).
UNION_LAT = 42.730711
UNION_LON = -73.676737
UNION_RADIUS_M = 75.0  # 75m catches all dwell pings; the on-route polyline
                       # itself sits ~20m from the stop marker on some loops.

# Campus timezone — CSV timestamps are UTC; converting here lets us bucket
# breaks by local day-of-week (what students actually experience).
CAMPUS_TZ = "America/New_York"

# Break window (campus local). Handoff's 14:00-18:00 cluster was UTC;
# converted to America/New_York (UTC-4 in summer, UTC-5 in winter) that's
# 10:00-14:00 local. Add slack to cover DST drift either way.
BREAK_WINDOW_START_MIN = 10 * 60
BREAK_WINDOW_END_MIN = 14 * 60

# Morning window (campus local): pre-break signature window. 07:00-11:00
# sits fully before the earliest possible break start at 10:00 local.
MORNING_WINDOW_START_MIN = 7 * 60
MORNING_WINDOW_END_MIN = 11 * 60

# Break threshold. Matches decision #1 from the break-detection handoff:
# data shows breaks 42-50m, P95 normal 15-20m, so 40m is the natural cutoff.
BREAK_GAP_MIN = 40

# Fixed-length signature vector — first N morning Union departures relative
# to 07:00. Shorter shuttle-days get padded with NaN (excluded from mean).
SIGNATURE_LEN = 6

# How many archetypes to extract per day-of-week. Matches the handoff's
# observed pattern (4 weekday, 2 Sunday, 1 Saturday).
# Keyed by weekday() (0=Mon..6=Sun).
K_PER_DOW: Dict[int, int] = {0: 4, 1: 4, 2: 4, 3: 4, 4: 4, 5: 1, 6: 2}

# Minimum cluster members to accept an archetype (avoid single-outlier
# clusters masquerading as archetypes).
MIN_CLUSTER_MEMBERS = 2


def _haversine_m(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: float, lon2: float) -> np.ndarray:
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * math.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _compress_dwells(visit_times: List[pd.Timestamp], gap_sec: int = 120) -> List[pd.Timestamp]:
    """Collapse a cluster of consecutive pings at Union into a single
    visit timestamp (the first ping). Pings > gap_sec apart start a new
    visit. Keeps a shuttle idling at Union from flooding the signature.
    """
    if not visit_times:
        return []
    compressed = [visit_times[0]]
    for t in visit_times[1:]:
        if (t - compressed[-1]).total_seconds() > gap_sec:
            compressed.append(t)
    return compressed


def _find_break_start(visits_local: List[pd.Timestamp]) -> pd.Timestamp | None:
    """Return the first visit whose follow-on gap >= BREAK_GAP_MIN and
    whose start time is inside the break window. None if no break found."""
    for i in range(len(visits_local) - 1):
        cur = visits_local[i]
        nxt = visits_local[i + 1]
        cur_min = cur.hour * 60 + cur.minute
        if cur_min < BREAK_WINDOW_START_MIN or cur_min > BREAK_WINDOW_END_MIN:
            continue
        if (nxt - cur).total_seconds() / 60.0 >= BREAK_GAP_MIN:
            return cur
    return None


def _morning_signature(visits_local: List[pd.Timestamp]) -> List[float]:
    """Return first SIGNATURE_LEN morning visit times as minutes since 07:00.
    Shorter arrays are padded with NaN; longer arrays are truncated.
    """
    morning: List[float] = []
    for v in visits_local:
        m = v.hour * 60 + v.minute
        if MORNING_WINDOW_START_MIN <= m < MORNING_WINDOW_END_MIN:
            morning.append(float(m - MORNING_WINDOW_START_MIN))
        if len(morning) >= SIGNATURE_LEN:
            break
    while len(morning) < SIGNATURE_LEN:
        morning.append(float("nan"))
    return morning


def _cluster_center_with_signature(
    candidates: List[dict], k: int,
) -> List[dict]:
    """Cluster candidates by break_start_min; return archetype dicts.

    Each candidate: {vehicle_id, date, break_start_min, signature}
    """
    if not candidates:
        return []
    if k < 1:
        return []

    # Effective k: don't ask for more clusters than candidates support at
    # MIN_CLUSTER_MEMBERS per cluster. Without this a sparse-data dow
    # (e.g. Mon with 3 candidates, k=4) produces all singleton clusters
    # that get dropped, yielding zero archetypes.
    max_supported_k = max(1, len(candidates) // MIN_CLUSTER_MEMBERS)
    k_eff = min(k, max_supported_k)

    break_mins = np.array([c["break_start_min"] for c in candidates], dtype=float)
    # Reshape for kmeans2 (needs 2D data).
    data = break_mins.reshape(-1, 1)

    # Deterministic init: seed cluster centers at evenly-spaced quantiles
    # of the data so reruns produce stable archetype IDs.
    quantiles = np.linspace(0, 1, k_eff + 2)[1:-1]
    init = np.quantile(break_mins, quantiles).reshape(-1, 1)

    try:
        centers, labels = kmeans2(data, init, minit="matrix", seed=1729)
    except Exception as e:  # pragma: no cover (defensive)
        logger.warning(f"kmeans2 failed ({e}); falling back to sort-buckets")
        order = np.argsort(break_mins)
        labels = np.zeros(len(candidates), dtype=int)
        bucket = np.array_split(order, k_eff)
        for i, idx in enumerate(bucket):
            labels[idx] = i
        centers = np.array([[break_mins[idx].mean()] for idx in bucket])

    archetypes: List[dict] = []
    for cluster_id in range(len(centers)):
        members = [candidates[i] for i, l in enumerate(labels) if l == cluster_id]
        if len(members) < MIN_CLUSTER_MEMBERS:
            continue

        member_breaks = np.array([m["break_start_min"] for m in members])
        member_signatures = np.array([m["signature"] for m in members])
        # nanmean so padded NaNs don't poison the mean.
        mean_sig = np.nanmean(member_signatures, axis=0)
        # Replace any still-NaN entries (column all-NaN) with sentinel 0
        # to keep cost-matrix math finite. Downstream cost masks NaN out.
        mean_sig = np.where(np.isnan(mean_sig), 0.0, mean_sig)

        archetypes.append(
            {
                "archetype_id": f"c{cluster_id}",
                "break_start_min": float(centers[cluster_id][0]),
                "median_break_start_min": float(np.median(member_breaks)),
                "signature": [float(x) for x in mean_sig],
                "member_count": int(len(members)),
                "member_vehicles": sorted(set(str(m["vehicle_id"]) for m in members)),
            }
        )

    archetypes.sort(key=lambda a: a["break_start_min"])
    return archetypes


def _extract_candidates_for_day(
    vid: str,
    union_visits_local: List[pd.Timestamp],
) -> dict | None:
    """Given a shuttle's Union visits for a single local day, compute a
    (break_start_min, signature) candidate. Returns None if no break
    window detected."""
    compressed = _compress_dwells(union_visits_local)
    if len(compressed) < 2:
        return None
    break_start = _find_break_start(compressed)
    if break_start is None:
        return None
    signature = _morning_signature(compressed)
    return {
        "vehicle_id": vid,
        "break_start_min": break_start.hour * 60 + break_start.minute,
        "signature": signature,
    }


def build_archetypes(csv_path: Path = _CSV_PATH,
                     out_dir: Path = _OUT_DIR) -> Dict[int, List[dict]]:
    """Read CSV, extract archetypes per day-of-week, write JSON files.
    Returns {dow_idx: [archetype, ...]} for testability.
    """
    logger.info(f"Loading {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    logger.info(f"Loaded {len(df)} rows, {df['vehicle_id'].nunique()} unique vehicles")

    # Timestamps are naive UTC — stamp, then convert to campus local for
    # day-of-week bucketing.
    df["ts_utc"] = df["timestamp"].dt.tz_localize("UTC")
    df["ts_local"] = df["ts_utc"].dt.tz_convert(CAMPUS_TZ)
    df["local_date"] = df["ts_local"].dt.date
    df["dow"] = df["ts_local"].dt.weekday  # 0=Mon..6=Sun

    # Filter to pings within UNION_RADIUS_M of the Union stop.
    dists = _haversine_m(df["latitude"].to_numpy(),
                         df["longitude"].to_numpy(),
                         UNION_LAT, UNION_LON)
    union_mask = dists <= UNION_RADIUS_M
    union_df = df.loc[union_mask].copy()
    logger.info(f"Filtered to {len(union_df)} Union pings (within {UNION_RADIUS_M}m)")

    # Group by (vehicle, local_date) and extract break candidates.
    candidates_by_dow: Dict[int, List[dict]] = defaultdict(list)
    groups = union_df.groupby(["vehicle_id", "local_date"], sort=False)
    for (vid, d), g in groups:
        visits = sorted(g["ts_local"].tolist())
        cand = _extract_candidates_for_day(str(vid), visits)
        if cand is None:
            continue
        cand["date"] = d.isoformat()
        dow = pd.Timestamp(d).weekday()
        candidates_by_dow[dow].append(cand)

    logger.info("Candidates per dow: " + ", ".join(
        f"{dow}:{len(v)}" for dow, v in sorted(candidates_by_dow.items())
    ))

    # Cluster per dow.
    archetypes_by_dow: Dict[int, List[dict]] = {}
    for dow in range(7):
        candidates = candidates_by_dow.get(dow, [])
        k = K_PER_DOW.get(dow, 3)
        archetypes_by_dow[dow] = _cluster_center_with_signature(candidates, k)
        logger.info(
            f"dow={dow} k={k}: "
            f"{len(candidates)} candidates -> {len(archetypes_by_dow[dow])} archetypes"
        )

    # Write per-dow files.
    out_dir.mkdir(parents=True, exist_ok=True)
    for dow, archetypes in archetypes_by_dow.items():
        out_path = out_dir / f"{dow}.json"
        payload = {
            "dow": dow,
            "dow_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow],
            "signature_window_start_hhmm": f"{MORNING_WINDOW_START_MIN // 60:02d}:{MORNING_WINDOW_START_MIN % 60:02d}",
            "signature_window_end_hhmm": f"{MORNING_WINDOW_END_MIN // 60:02d}:{MORNING_WINDOW_END_MIN % 60:02d}",
            "signature_len": SIGNATURE_LEN,
            "break_window_start_hhmm": f"{BREAK_WINDOW_START_MIN // 60:02d}:{BREAK_WINDOW_START_MIN % 60:02d}",
            "break_window_end_hhmm": f"{BREAK_WINDOW_END_MIN // 60:02d}:{BREAK_WINDOW_END_MIN % 60:02d}",
            "break_gap_min": BREAK_GAP_MIN,
            "archetypes": archetypes,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        logger.info(f"Wrote {out_path} ({len(archetypes)} archetypes)")

    return archetypes_by_dow


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_archetypes()
