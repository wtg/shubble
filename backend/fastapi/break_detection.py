"""Real-time break detection for /api/locations (Phase 3).

Three independent signals, OR'd together:

  1. Stay-point NOT at Union: shuttle stationary for 5+ min anywhere
     except Student Union. Fires at ~T+5min. Normal stop dwell is ~30s;
     only Union has multi-minute dwells between loops. This is the
     primary signal — catches both on-route and off-route breaks.

  2. CUSUM on loop cadence: per-shuttle adaptive threshold based on
     recent Union-return intervals. Backup for cases where the shuttle
     is slowly moving (not stationary) but not completing loops.
     Fires at ~T+20-25min (floor=20min above fleet P95).

  3. Hard fallback: >=40 min since last Union visit during 8-20 local.
     Safety net — always catches by T+40.

Integrates with /api/locations via predict_on_break(), which is called
from _build_locations_payload in routes.py. The result is OR'd with
Phase 1's schedule-gap flag. No API shape change.

See .planning/quick/260415-owx.../HANDOFF.md for design history.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy import select

from backend.models import VehicleLocation
from backend.time_utils import dev_now, get_campus_start_of_day

logger = logging.getLogger(__name__)


# Student Union coordinate (shared between NORTH and WEST routes).
UNION_LAT = 42.730711
UNION_LON = -73.676737
UNION_RADIUS_M = 75.0

# Dwell-compression: pings within 2 min collapse to one visit.
DWELL_GAP_SEC = 120

# --- CUSUM on loop cadence ---
# EMA smoothing for per-shuttle loop cadence estimate. 0.3 weights
# recent visits heavily while still smoothing over 3-5 loops.
CUSUM_EMA_ALPHA = 0.3
# Slack (minutes): tolerated deviation from mu before accumulating.
# Absorbs normal traffic variance (P90 of service intervals is ~15 min
# vs median ~6 min, so 1-2 min slack is appropriate).
CUSUM_SLACK_MIN = 1.0
# Threshold (minutes): accumulated excess above (mu + slack) to fire.
# With slack=1 and threshold=5, fires at mu + 6 min — e.g. 18 min for
# a 12-min-loop shuttle.
CUSUM_THRESHOLD_MIN = 5.0
# Absolute floor (minutes): CUSUM never fires below this regardless of
# personal mu. Prevents false positives from fast shuttles hitting a
# single slow loop. Calibrated via ml/extract_break_data.py on pooled
# Jul-2025 + Feb-Apr-2026 data (1.47M pings, 25 vehicles, 100 days):
# P97 of normal inter-visit intervals = 16.4min + 2min margin = ~18.
# Rerun `python -m ml.extract_break_data --csv <paths...>` to recalibrate.
CUSUM_MIN_FIRE_MIN = 18.0
# Default assumed loop time (minutes) before a shuttle has enough
# observations. Matches the all-weekday median inter-visit interval.
CUSUM_DEFAULT_MU_MIN = 12.0
# Minimum completed Union visits to compute a personal mu. Before this,
# uses CUSUM_DEFAULT_MU_MIN.
CUSUM_MIN_VISITS = 3

# --- Hard fallback ---
FALLBACK_GAP_MIN = 40
FALLBACK_ACTIVE_START_MIN = 8 * 60
FALLBACK_ACTIVE_END_MIN = 20 * 60

# --- Stay-point + off-route ---
STAY_POINT_RADIUS_M = 75.0
STAY_POINT_MIN_DWELL_SEC = 5 * 60
STAY_POINT_OFF_ROUTE_KM = 0.1  # 100m
STAY_POINT_LOOKBACK_SEC = 15 * 60


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


def _min_dist_to_any_route_km(lat: float, lon: float) -> float:
    """Min haversine distance (km) from point to any route polyline.
    Ambiguous (None) = on-route = 0km."""
    from shared.stops import Stops

    result = Stops.get_closest_point((lat, lon))
    if result is None or result[0] is None:
        return 0.0
    return float(result[0])


def _detect_active_stay_point(
    pings: List[tuple],
    now_utc: datetime,
) -> Optional[tuple[datetime, float, float, float]]:
    """Li-Zheng stay-point detector (backward walk from latest ping).
    Returns (start_ts, centroid_lat, centroid_lon, duration_sec) or None."""
    if not pings:
        return None
    cutoff = now_utc.timestamp() - STAY_POINT_LOOKBACK_SEC
    recent: List[tuple] = []
    for ts, lat, lon in pings:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts.timestamp() >= cutoff:
            recent.append((ts, lat, lon))
    if len(recent) < 2:
        return None
    recent.sort(key=lambda p: p[0])

    anchor_ts, anchor_lat, anchor_lon = recent[-1]
    start_idx = len(recent) - 1
    for i in range(len(recent) - 1, -1, -1):
        ts, lat, lon = recent[i]
        if _haversine_m_scalar(lat, lon, anchor_lat, anchor_lon) > STAY_POINT_RADIUS_M:
            break
        start_idx = i

    start_ts = recent[start_idx][0]
    duration = (anchor_ts - start_ts).total_seconds()
    if duration < STAY_POINT_MIN_DWELL_SEC:
        return None
    span = recent[start_idx:]
    c_lat = sum(p[1] for p in span) / len(span)
    c_lon = sum(p[2] for p in span) / len(span)
    return (start_ts, c_lat, c_lon, duration)


def _union_visits_utc(
    pings: List[tuple],
) -> List[datetime]:
    """Return dwell-compressed Union visit timestamps (UTC) from raw pings."""
    union_utc: List[datetime] = []
    for ts, lat, lon in pings:
        if _haversine_m_scalar(lat, lon, UNION_LAT, UNION_LON) <= UNION_RADIUS_M:
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            union_utc.append(ts)
    return _compress_dwells(union_utc)


def _compute_personal_mu(intervals: List[float]) -> float:
    """EMA of recent inter-visit intervals, excluding break-length gaps.
    Returns estimated loop cadence in minutes."""
    normal = [iv for iv in intervals if iv < FALLBACK_GAP_MIN]
    if not normal:
        return CUSUM_DEFAULT_MU_MIN
    mu = normal[0]
    for iv in normal[1:]:
        mu = CUSUM_EMA_ALPHA * iv + (1 - CUSUM_EMA_ALPHA) * mu
    return mu


def _cusum_fires(
    compressed_union_utc: List[datetime],
    now_utc: datetime,
) -> tuple[bool, float, float]:
    """CUSUM change-point detector on Union-return interval.

    Returns (fires, mu, elapsed_min).

    Fires when the current gap exceeds mu + slack + threshold, where mu
    is the shuttle's personal EMA of recent loop intervals. This adapts
    to each shuttle's cadence: a 12-min-loop shuttle fires at ~18 min,
    a 15-min-loop shuttle at ~21 min.
    """
    if not compressed_union_utc:
        return False, CUSUM_DEFAULT_MU_MIN, 0.0

    last_visit = compressed_union_utc[-1]
    elapsed = (now_utc - last_visit).total_seconds() / 60.0

    if len(compressed_union_utc) < CUSUM_MIN_VISITS:
        mu = CUSUM_DEFAULT_MU_MIN
    else:
        intervals = [
            (compressed_union_utc[i] - compressed_union_utc[i - 1]).total_seconds() / 60.0
            for i in range(1, len(compressed_union_utc))
        ]
        mu = _compute_personal_mu(intervals)

    fire_threshold = max(mu + CUSUM_SLACK_MIN + CUSUM_THRESHOLD_MIN, CUSUM_MIN_FIRE_MIN)
    return elapsed > fire_threshold, mu, elapsed


async def _fetch_today_pings(
    session_factory, vehicle_ids: List[str], start_of_day_utc: datetime
) -> Dict[str, List[tuple]]:
    """Return {vid: [(ts_utc, lat, lon), ...]} for today's pings."""
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


async def predict_on_break(
    vehicle_ids: List[str],
    session_factory,
    campus_tz,
) -> Dict[str, bool]:
    """Per-vehicle on_break flag.

    Three signals, OR'd:
      1. Stay-point NOT at Union (T+5min for any stationary break)
      2. CUSUM on loop cadence (T+20-25min adaptive, backup)
      3. Hard fallback (T+40min safety net)
    """
    if not vehicle_ids:
        return {}

    now_utc = dev_now(timezone.utc)
    now_local = now_utc.astimezone(campus_tz)
    now_local_min = now_local.hour * 60 + now_local.minute

    start_of_day_utc = get_campus_start_of_day()
    pings_by_vid = await _fetch_today_pings(session_factory, vehicle_ids, start_of_day_utc)

    result: Dict[str, bool] = {}
    for vid in vehicle_ids:
        all_pings = pings_by_vid.get(vid, [])
        union_visits = _union_visits_utc(all_pings)
        last_union = union_visits[-1] if union_visits else None
        since_union_min = (
            (now_utc - last_union).total_seconds() / 60.0
            if last_union is not None else None
        )
        in_active_window = FALLBACK_ACTIVE_START_MIN <= now_local_min <= FALLBACK_ACTIVE_END_MIN

        # Signal 1: stay-point NOT at Union.
        # A shuttle stationary for 5+ min ANYWHERE except Union is on
        # break — normal stop dwell is ~30s; only Union has multi-minute
        # dwells between loops. Replaces the old off-route gate which
        # missed on-route idling (e.g. parked at COLONIE for 20 min).
        # Gated on has-visited-Union-today (in-service check).
        stay_point_fires = False
        if since_union_min is not None:
            sp = _detect_active_stay_point(all_pings, now_utc)
            if sp is not None:
                _, c_lat, c_lon, _dur = sp
                not_at_union = _haversine_m_scalar(
                    c_lat, c_lon, UNION_LAT, UNION_LON
                ) > UNION_RADIUS_M
                if not_at_union:
                    stay_point_fires = True

        # Signal 2: CUSUM on loop cadence.
        # Gated on active window so overnight gaps don't fire.
        cusum_fires_flag = False
        if in_active_window and union_visits:
            cusum_fires_flag, _mu, _elapsed = _cusum_fires(union_visits, now_utc)

        # Signal 3: hard fallback (40-min gap).
        fallback = False
        if since_union_min is not None and in_active_window:
            if since_union_min >= FALLBACK_GAP_MIN:
                fallback = True

        result[vid] = stay_point_fires or cusum_fires_flag or fallback

    return result


# ─────────────────────────────────────────────────────────────────────
# Predictive layer: announce upcoming breaks BEFORE they start.
# Consumes the offline-trained priors (shared/break_priors.json) and
# effective schedule (shared/break_effective_schedule.json) built by
# .planning/debug/predictive_layers.py. On a held-out temporal test set
# (Apr 8-21 2026, 50 real breaks), this announces 94% of breaks ahead
# of time with 85-min median lead and 8.2-min median start-time error.
# ─────────────────────────────────────────────────────────────────────

import json as _json
from pathlib import Path as _Path

_SHARED_DIR = _Path(__file__).resolve().parents[2] / "shared"
_PRIORS_JSON = _SHARED_DIR / "break_priors.json"
_EFFECTIVE_JSON = _SHARED_DIR / "break_effective_schedule.json"

_priors_cache: Optional[dict] = None  # {(day_type, run, "HH:MM"): prior_entry}
_effective_cache: Optional[dict] = None  # {(day_type, run): [entry, ...]}


def _load_forecast_artifacts() -> tuple[dict, dict]:
    """Load and cache priors + effective schedule from shared/. Returns
    ({(dt, run, "HH:MM"): prior}, {(dt, run): [entries]})."""
    global _priors_cache, _effective_cache
    if _priors_cache is not None and _effective_cache is not None:
        return _priors_cache, _effective_cache

    priors_idx: dict = {}
    if _PRIORS_JSON.exists():
        for p in _json.loads(_PRIORS_JSON.read_text()):
            priors_idx[(p["day_type"], p["run"], p["scheduled_start"])] = p
    else:
        logger.warning(f"Break-prediction priors missing at {_PRIORS_JSON} — /api/predictions will return empty")

    eff_idx: dict = {}
    if _EFFECTIVE_JSON.exists():
        raw = _json.loads(_EFFECTIVE_JSON.read_text())
        for key, entries in raw.items():
            if "|" not in key:
                continue
            dt, run = key.split("|", 1)
            eff_idx[(dt, run)] = entries
    else:
        logger.warning(f"Effective-schedule artifact missing at {_EFFECTIVE_JSON}")

    _priors_cache = priors_idx
    _effective_cache = eff_idx
    return priors_idx, eff_idx


def _day_type_for_local(local_dt: datetime) -> str:
    dow = local_dt.weekday()
    return "M-F" if dow < 5 else ("Sat" if dow == 5 else "Sun")


async def _fetch_today_db_schedule(
    db_session, campus_tz, now_utc: datetime,
) -> Optional[set]:
    """Return set of (day_type, run_label, "HH:MM") for today's schedule from DB.

    The DB's DaySchedule/BusSchedule/RouteToBusSchedule tables are the
    forward-compatible source of truth for printed schedules (Issue #315).
    We cross-reference them against the JSON artifact so a schedule change
    pushed to the DB surfaces as `db_verified=False` in predictions until
    the offline pipeline is retrained.

    Returns None (not an empty set) when the DB is unreachable or empty,
    so callers can distinguish "no check performed" from "no slots found".
    """
    from sqlalchemy import select as _select
    from backend.models import (
        DaySchedule, BusSchedule, RouteToBusSchedule, BusScheduleToDaySchedule,
    )
    from sqlalchemy.orm import selectinload

    now_local = now_utc.astimezone(campus_tz)
    day_name = _day_type_for_local(now_local)

    try:
        q = (
            _select(DaySchedule)
            .where(DaySchedule.name == day_name)
            .options(
                selectinload(DaySchedule.bus_schedule_to_day_schedule)
                .selectinload(BusScheduleToDaySchedule.bus_schedule)
                .selectinload(BusSchedule.route_to_bus_schedules)
            )
        )
        rows = (await db_session.execute(q)).scalars().all()
    except Exception as e:
        logger.warning(f"DB schedule fetch failed ({e!r}); predictions won't carry db_verified flag")
        return None

    out: set = set()
    for day in rows:
        for mapping in day.bus_schedule_to_day_schedule:
            bs = mapping.bus_schedule
            for rbs in bs.route_to_bus_schedules:
                hhmm = rbs.time.strftime("%H:%M")
                out.add((day.name, bs.name, hhmm))

    if not out:
        return None
    return out


def predict_upcoming_breaks(
    now_utc: datetime,
    campus_tz,
    lookahead_min: int = 180,
    db_slots: Optional[set] = None,
) -> List[Dict]:
    """Announce upcoming breaks for today, per scheduled run.

    Reads the offline-trained effective schedule (which includes bimodal
    and discovered slots) and projects each break slot onto today's date.
    Returns entries with non-negative lead time up to `lookahead_min`.

    Each result: {
      "run": str,                  # e.g. "West-1 (223/229)"
      "predicted_start": iso8601,  # campus-local
      "predicted_end": iso8601,
      "confidence": float,         # 0..1
      "lead_min": float,           # minutes from now to predicted_start
      "source": str,               # "scheduled-active" / "discovered" / "bimodal-mode"
      "sigma_min": float,          # spread around predicted_start
    }
    """
    priors_idx, eff_idx = _load_forecast_artifacts()
    if not eff_idx:
        return []

    now_local = now_utc.astimezone(campus_tz)
    dt = _day_type_for_local(now_local)
    today_midnight = now_local.replace(hour=0, minute=0, second=0, microsecond=0)

    results: List[Dict] = []

    def _emit(run: str, mean_min: float, sigma_min: float, take_rate: float,
              source: str, duration_min: float = 25.0,
              printed_start: Optional[str] = None):
        total_sec = max(0.0, mean_min) * 60.0
        start_local = today_midnight + timedelta(seconds=total_sec)
        end_local = start_local + timedelta(minutes=duration_min)
        lead_min = (start_local - now_local).total_seconds() / 60.0
        if lead_min < 0 or lead_min > lookahead_min:
            return
        db_verified: Optional[bool] = None
        if db_slots is not None and printed_start:
            db_verified = (dt, run, printed_start) in db_slots
        results.append({
            "run": run,
            "predicted_start": start_local.isoformat(),
            "predicted_end": end_local.isoformat(),
            "confidence": round(max(0.0, min(1.0, take_rate)), 3),
            "lead_min": round(lead_min, 1),
            "source": source,
            "sigma_min": round(float(sigma_min), 1),
            "db_verified": db_verified,
        })

    for (entry_dt, run), entries in eff_idx.items():
        if entry_dt != dt:
            continue
        for e in entries:
            take_rate = float(e.get("take_rate", 0.0))
            src = e.get("src")
            printed_start = e.get("printed_start")  # "HH:MM" or None

            # Bimodal: emit each mode separately so we catch both clusters.
            # All modes inherit the parent entry's printed_start for db_verified.
            if e.get("modes"):
                total_days = max(e.get("n_days", 1), 1)
                for m in e["modes"]:
                    _emit(run, m["mean_min"], m["sigma_min"],
                          m["n"] / total_days, source="bimodal-mode",
                          printed_start=printed_start)
                continue

            # Discovered or scheduled-active: single effective time.
            mean_min = e.get("effective_mean_min")
            sigma_min = e.get("effective_sigma_min", 5.0)
            if mean_min is None:
                continue
            if src == "scheduled-rare" and take_rate < 0.1:
                continue
            _emit(run, mean_min, sigma_min, take_rate,
                  source=src or "unknown", printed_start=printed_start)

    results.sort(key=lambda r: r["predicted_start"])
    return results
