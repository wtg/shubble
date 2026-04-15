"""Archetype-based break predictor (Phase 3 break detection).

Loads per-dow archetypes from shared/archetypes/<dow>.json and matches
today's active shuttles to archetypes via Hungarian assignment on morning
Union-departure signatures. Output: predicted on_break flag per vehicle.

Integrates with /api/locations as an OR with Phase 1's schedule-gap
detector (see backend/fastapi/routes.py:_build_locations_payload). The
combined flag is what the frontend reads; no API shape change.

Design decisions (from .planning/quick/260415-owx.../HANDOFF.md):
  1. Gap threshold fallback: 40 min
  2. Signature dim: multi-value (first 6 morning Union-departure times)
  3. Matching algorithm: Hungarian (pluggable via set_matcher())
  4. State location: none — per-cycle DB query (/api/locations already
     cached 3-15s so cost is bounded)
  5. Archetype persistence: shared/archetypes/<dow>.json written offline
     by ml/build_archetypes.py; reloaded once per process.

Concept provenance: cost-matrix + linear_sum_assignment approach inspired
by the earlier schedule-matching code in shared/schedules.py (deleted
2026-04-10 in f81d191) — correctness constraints differ (per-cycle
real-time here vs end-of-day offline there).
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sqlalchemy import select

from backend.models import VehicleLocation
from backend.time_utils import dev_now, get_campus_start_of_day

logger = logging.getLogger(__name__)


_ARCHETYPES_DIR = Path(__file__).parent.parent.parent / "shared" / "archetypes"

# Student Union coordinate (shared between NORTH and WEST routes).
UNION_LAT = 42.730711
UNION_LON = -73.676737
UNION_RADIUS_M = 75.0

# Morning signature window (campus local). Must match
# MORNING_WINDOW_START_MIN in ml/build_archetypes.py.
MORNING_WINDOW_START_MIN = 7 * 60
MORNING_WINDOW_END_MIN = 11 * 60
SIGNATURE_LEN = 6

# Dwell-compression: pings within 2 min collapse to one visit.
DWELL_GAP_SEC = 120

# Fallback gap threshold (decision #1).
FALLBACK_GAP_MIN = 40

# Plausible daytime window for fallback to fire.
FALLBACK_ACTIVE_START_MIN = 8 * 60
FALLBACK_ACTIVE_END_MIN = 20 * 60

# Archetype prediction window: start a few minutes early so the flag
# fires just before the shuttle's predicted break begins, and stays on
# through the archetype's typical dwell.
BREAK_PREDICT_GRACE_MIN = 5
BREAK_PREDICT_DWELL_MIN = 60

# Cost threshold — a match whose RMSE distance exceeds this is dropped
# (treated as unmatched). In signature space each component is
# minutes-from-07:00, so 30 means "archetype doesn't fit within 30 min
# per morning departure slot". Above that, fallback must carry the load.
MAX_MATCH_RMSE = 30.0


_ARCHETYPES_CACHE: Dict[int, List[dict]] = {}


def _load_archetypes(dow: int) -> List[dict]:
    if dow in _ARCHETYPES_CACHE:
        return _ARCHETYPES_CACHE[dow]
    path = _ARCHETYPES_DIR / f"{dow}.json"
    if not path.exists():
        _ARCHETYPES_CACHE[dow] = []
        return []
    try:
        data = json.loads(path.read_text())
        _ARCHETYPES_CACHE[dow] = data.get("archetypes", [])
    except Exception as e:
        logger.warning(f"Failed to load archetypes for dow={dow}: {e}")
        _ARCHETYPES_CACHE[dow] = []
    return _ARCHETYPES_CACHE[dow]


def reset_archetype_cache() -> None:
    """Drop the in-memory archetype cache (for tests and for the
    one-shot startup rebuild hook)."""
    _ARCHETYPES_CACHE.clear()


def _haversine_m_scalar(lat1: float, lon1: float,
                        lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _compress_dwells(times: List[datetime]) -> List[datetime]:
    if not times:
        return []
    times = sorted(times)
    out = [times[0]]
    for t in times[1:]:
        if (t - out[-1]).total_seconds() > DWELL_GAP_SEC:
            out.append(t)
    return out


def _extract_signature(visits_local: List[datetime]) -> List[float]:
    """First SIGNATURE_LEN morning visits in minutes since 07:00.
    Pad with NaN when fewer visits are present."""
    sig: List[float] = []
    for v in visits_local:
        m = v.hour * 60 + v.minute
        if MORNING_WINDOW_START_MIN <= m < MORNING_WINDOW_END_MIN:
            sig.append(float(m - MORNING_WINDOW_START_MIN))
        if len(sig) >= SIGNATURE_LEN:
            break
    while len(sig) < SIGNATURE_LEN:
        sig.append(float("nan"))
    return sig


def _signature_cost(vehicle_sig: np.ndarray, archetype_sig: np.ndarray) -> float:
    """RMSE over non-NaN components of vehicle_sig. Missing components
    don't penalize — a vehicle that started late still matches on what
    it has observed so far."""
    mask = ~np.isnan(vehicle_sig)
    if not mask.any():
        return float("inf")
    diff = vehicle_sig[mask] - archetype_sig[mask]
    return float(np.sqrt((diff ** 2).sum() / mask.sum()))


def match_signatures(
    vehicle_sigs: Dict[str, np.ndarray],
    archetypes: List[dict],
) -> Dict[str, dict]:
    """Hungarian assignment: vehicles -> archetypes by signature cost.
    Unmatched vehicles (no morning data, or cost > MAX_MATCH_RMSE) are
    absent from the returned dict."""
    if not vehicle_sigs or not archetypes:
        return {}

    vids = list(vehicle_sigs.keys())
    arch_vectors = [np.asarray(a["signature"], dtype=float) for a in archetypes]

    n_vids = len(vids)
    n_arch = len(arch_vectors)
    cost = np.zeros((n_vids, n_arch), dtype=float)
    for i, vid in enumerate(vids):
        for j, arch in enumerate(arch_vectors):
            cost[i, j] = _signature_cost(vehicle_sigs[vid], arch)

    # Pad when vehicles > archetypes so linear_sum_assignment runs on a
    # rectangular matrix with valid "no-match" slots.
    if n_vids > n_arch:
        pad = np.full((n_vids, n_vids - n_arch), fill_value=1e9, dtype=float)
        cost = np.hstack([cost, pad])

    # When archetypes > vehicles, linear_sum_assignment naturally returns
    # len(vids) matched pairs; excess archetypes are unassigned.
    row_ind, col_ind = linear_sum_assignment(cost)

    out: Dict[str, dict] = {}
    for r, c in zip(row_ind, col_ind):
        if c >= n_arch:
            continue
        if not math.isfinite(cost[r, c]) or cost[r, c] > MAX_MATCH_RMSE:
            continue
        out[vids[r]] = archetypes[c]
    return out


# Pluggable seam — sort-and-index or any drop-in replacement can swap
# in here. Kept as a module-level reference so tests and future scale
# work can override without patching callers.
_match_fn: Callable[[Dict[str, np.ndarray], List[dict]], Dict[str, dict]] = match_signatures


def set_matcher(fn: Callable[[Dict[str, np.ndarray], List[dict]], Dict[str, dict]]) -> None:
    """Swap the matching algorithm (default: Hungarian)."""
    global _match_fn
    _match_fn = fn


async def _fetch_today_visits(
    session_factory, vehicle_ids: List[str], start_of_day_utc: datetime
) -> Dict[str, List[tuple]]:
    """Return {vid: [(ts_utc, lat, lon), ...]} for today's pings.
    Filtering to Union radius happens Python-side after the fetch."""
    if not vehicle_ids:
        return {}
    async with session_factory() as db:
        q = (
            select(
                VehicleLocation.vehicle_id,
                VehicleLocation.timestamp,
                VehicleLocation.latitude,
                VehicleLocation.longitude,
            )
            .where(
                VehicleLocation.vehicle_id.in_(vehicle_ids),
                VehicleLocation.timestamp >= start_of_day_utc,
            )
            .order_by(VehicleLocation.vehicle_id, VehicleLocation.timestamp)
        )
        rows = (await db.execute(q)).all()

    by_vid: Dict[str, List[tuple]] = {vid: [] for vid in vehicle_ids}
    for vid, ts, lat, lon in rows:
        if vid in by_vid:
            by_vid[vid].append((ts, lat, lon))
    return by_vid


def _union_visits_local(
    pings: List[tuple], campus_tz
) -> tuple[List[datetime], Optional[datetime]]:
    """Return (compressed local-tz Union visits, last Union visit ts UTC).
    Compression uses DWELL_GAP_SEC; filtering is radius-based."""
    union_utc: List[datetime] = []
    for ts, lat, lon in pings:
        if _haversine_m_scalar(lat, lon, UNION_LAT, UNION_LON) <= UNION_RADIUS_M:
            # Normalize timestamp to UTC tz-aware so arithmetic works
            # regardless of how the driver returned it.
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            union_utc.append(ts)
    compressed_utc = _compress_dwells(union_utc)
    compressed_local = [t.astimezone(campus_tz) for t in compressed_utc]
    last_utc = compressed_utc[-1] if compressed_utc else None
    return compressed_local, last_utc


async def predict_on_break(
    vehicle_ids: List[str],
    session_factory,
    campus_tz,
) -> Dict[str, bool]:
    """Per-vehicle on_break flag.

    Returns {vid: True/False}. True when EITHER:
      (a) Hungarian match to an archetype whose break window brackets now.
      (b) Fallback: >=FALLBACK_GAP_MIN since last Union visit, during
          FALLBACK_ACTIVE_START..FALLBACK_ACTIVE_END local.
    """
    if not vehicle_ids:
        return {}

    now_utc = dev_now(timezone.utc)
    now_local = now_utc.astimezone(campus_tz)
    dow = now_local.weekday()
    archetypes = _load_archetypes(dow)

    start_of_day_utc = get_campus_start_of_day()
    visits_by_vid = await _fetch_today_visits(session_factory, vehicle_ids, start_of_day_utc)

    sigs: Dict[str, np.ndarray] = {}
    last_union_by_vid: Dict[str, Optional[datetime]] = {}
    for vid in vehicle_ids:
        compressed_local, last_utc = _union_visits_local(visits_by_vid.get(vid, []), campus_tz)
        sigs[vid] = np.asarray(_extract_signature(compressed_local), dtype=float)
        last_union_by_vid[vid] = last_utc

    matches = _match_fn(sigs, archetypes) if archetypes else {}

    now_local_min = now_local.hour * 60 + now_local.minute
    result: Dict[str, bool] = {}
    for vid in vehicle_ids:
        archetype = matches.get(vid)
        predicted = False
        if archetype is not None:
            start_min = archetype["break_start_min"] - BREAK_PREDICT_GRACE_MIN
            end_min = archetype["break_start_min"] + BREAK_PREDICT_DWELL_MIN
            if start_min <= now_local_min <= end_min:
                predicted = True

        fallback = False
        lu = last_union_by_vid[vid]
        if lu is not None:
            since_min = (now_utc - lu).total_seconds() / 60.0
            if (
                since_min >= FALLBACK_GAP_MIN
                and FALLBACK_ACTIVE_START_MIN <= now_local_min <= FALLBACK_ACTIVE_END_MIN
            ):
                fallback = True

        result[vid] = predicted or fallback

    return result
