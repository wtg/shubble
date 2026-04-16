"""Back-test the break-detection matcher against all historical data.

For each (local_date, vehicle_id) with an observed break in
locations_raw.csv: extract the morning signature, run the production
Hungarian matcher against the per-dow archetypes, and compare the
predicted break start to the actual observed break start. Also measures
how many breaks the 40-min fallback would have caught.

Data-leakage note: archetypes were trained on the same CSV, so this is
an optimistic upper bound. The goal is to confirm the *shape* of the
matcher's output is sensible (predictions cluster near actuals), not to
claim out-of-sample accuracy.

Run:  python -m ml.eval_archetypes
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from ml.build_archetypes import (
    _haversine_m,
    _compress_dwells,
    _find_break_start,
    _morning_signature,
    UNION_LAT, UNION_LON, UNION_RADIUS_M,
    CAMPUS_TZ, BREAK_GAP_MIN,
    SIGNATURE_LEN, MORNING_WINDOW_START_MIN, MORNING_WINDOW_END_MIN,
)
from backend.fastapi.break_archetype import (
    match_signatures,
    _load_archetypes,
    reset_archetype_cache,
    MAX_MATCH_RMSE,
    BREAK_PREDICT_GRACE_MIN,
    BREAK_PREDICT_DWELL_MIN,
    FALLBACK_GAP_MIN,
)

logger = logging.getLogger(__name__)

_CSV_PATH = Path(__file__).parent / "cache" / "shared" / "locations_raw.csv"


def _load_union_visits() -> pd.DataFrame:
    df = pd.read_csv(_CSV_PATH, parse_dates=["timestamp"])
    df["ts_utc"] = df["timestamp"].dt.tz_localize("UTC")
    df["ts_local"] = df["ts_utc"].dt.tz_convert(CAMPUS_TZ)
    df["local_date"] = df["ts_local"].dt.date
    df["dow"] = df["ts_local"].dt.weekday
    dists = _haversine_m(df["latitude"].to_numpy(), df["longitude"].to_numpy(),
                         UNION_LAT, UNION_LON)
    return df.loc[dists <= UNION_RADIUS_M].copy()


def _build_sample(
    visits_local: List[pd.Timestamp],
) -> Dict | None:
    """Return dict with signature + actual break info, or None if no
    break was observable that shuttle-day."""
    compressed = _compress_dwells(visits_local)
    if len(compressed) < 2:
        return None
    bs = _find_break_start(compressed)
    if bs is None:
        return None
    # Signature computed identically to production.
    sig_floats = _morning_signature(compressed)
    sig_arr = np.asarray(sig_floats, dtype=float)

    # Find break END (next visit after break_start). Used for flag-window
    # accuracy measurement — did the predicted window bracket the actual?
    be = None
    for i in range(len(compressed) - 1):
        if compressed[i] == bs:
            be = compressed[i + 1]
            break
    actual_break_min = bs.hour * 60 + bs.minute
    return {
        "signature": sig_arr,
        "actual_break_start_min": actual_break_min,
        "actual_break_end_min": be.hour * 60 + be.minute if be is not None else None,
    }


def _eval_fallback_coverage(
    visits_local: List[pd.Timestamp],
) -> Tuple[bool, int | None]:
    """Would a 40-min-gap fallback have caught this break? Returns
    (caught, when_fallback_would_fire_min)."""
    compressed = _compress_dwells(visits_local)
    for i in range(len(compressed) - 1):
        gap_min = (compressed[i + 1] - compressed[i]).total_seconds() / 60.0
        if gap_min >= FALLBACK_GAP_MIN:
            fire_time = compressed[i] + pd.Timedelta(minutes=FALLBACK_GAP_MIN)
            return True, fire_time.hour * 60 + fire_time.minute
    return False, None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    reset_archetype_cache()

    logger.info("Loading historical CSV")
    u = _load_union_visits()
    logger.info(f"{len(u)} Union pings over {u['local_date'].nunique()} days")

    samples_by_dow: Dict[int, List[dict]] = defaultdict(list)
    groups = u.groupby(["vehicle_id", "local_date"], sort=False)
    for (vid, d), g in groups:
        visits = sorted(g["ts_local"].tolist())
        sample = _build_sample(visits)
        if sample is None:
            continue
        sample["vid"] = str(vid)
        sample["date"] = d.isoformat()
        sample["dow"] = pd.Timestamp(d).weekday()
        fb_caught, fb_min = _eval_fallback_coverage(visits)
        sample["fallback_caught"] = fb_caught
        sample["fallback_fire_min"] = fb_min
        samples_by_dow[sample["dow"]].append(sample)

    total = sum(len(v) for v in samples_by_dow.values())
    logger.info(f"Observed break samples: {total}")

    print("\n" + "=" * 72)
    print("BACK-TEST: archetype matcher vs observed breaks")
    print("=" * 72)

    # Aggregate trackers
    all_abs_errs: List[float] = []
    archetype_matched = 0
    archetype_unmatched = 0
    fallback_only = 0
    window_hits = 0
    fallback_caught_total = 0

    # Per-dow table
    rows = []
    for dow in sorted(samples_by_dow.keys()):
        samples = samples_by_dow[dow]
        archetypes = _load_archetypes(dow)
        dow_errs: List[float] = []
        dow_matched = 0
        dow_unmatched = 0
        dow_fb_caught = sum(1 for s in samples if s["fallback_caught"])

        for s in samples:
            vsigs = {s["vid"]: s["signature"]}
            matches = match_signatures(vsigs, archetypes)
            arch = matches.get(s["vid"])
            if arch is None:
                dow_unmatched += 1
                archetype_unmatched += 1
                # Unmatched but fallback caught -> flag still fires
                if s["fallback_caught"]:
                    fallback_only += 1
                continue
            dow_matched += 1
            archetype_matched += 1

            pred_min = arch["break_start_min"]
            actual = s["actual_break_start_min"]
            err = pred_min - actual
            dow_errs.append(err)
            all_abs_errs.append(abs(err))

            # Would the flag have fired during the actual break?
            # Flag is on from (pred - grace) to (pred + dwell).
            start = pred_min - BREAK_PREDICT_GRACE_MIN
            end = pred_min + BREAK_PREDICT_DWELL_MIN
            # Actual break spans [actual_break_start_min, actual_break_end_min]
            actual_end = s["actual_break_end_min"] or (actual + 45)
            # Overlap check
            if max(start, actual) <= min(end, actual_end):
                window_hits += 1

        fallback_caught_total += dow_fb_caught
        name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dow]
        if dow_errs:
            mae = float(np.mean(np.abs(dow_errs)))
            me = float(np.mean(dow_errs))
            p95 = float(np.percentile(np.abs(dow_errs), 95))
        else:
            mae = me = p95 = float("nan")
        rows.append((
            name, len(samples), len(archetypes),
            dow_matched, dow_unmatched, dow_fb_caught,
            mae, me, p95,
        ))

    print(f"{'dow':<5}{'N':>4}{'archs':>7}{'matched':>9}{'unmtch':>8}"
          f"{'fb_ok':>7}{'MAE':>8}{'ME':>8}{'P95':>8}")
    for r in rows:
        print(f"{r[0]:<5}{r[1]:>4}{r[2]:>7}{r[3]:>9}{r[4]:>8}"
              f"{r[5]:>7}{r[6]:>8.1f}{r[7]:>8.1f}{r[8]:>8.1f}")

    print()
    print(f"Total break samples:               {total}")
    print(f"  Archetype-matched:               {archetype_matched} "
          f"({100*archetype_matched/total:.0f}%)")
    print(f"  Unmatched (archetype):           {archetype_unmatched}")
    print(f"    ...of which fallback-caught:   {fallback_only}")
    print(f"  Fallback coverage (independent): {fallback_caught_total} "
          f"({100*fallback_caught_total/total:.0f}%)")
    print(f"Flag window hits (predicted bracketed actual): {window_hits} "
          f"({100*window_hits/total:.0f}%)")

    if all_abs_errs:
        print(f"\nArchetype-only MAE:       {np.mean(all_abs_errs):.1f} min")
        print(f"Archetype-only P50/P95:   "
              f"{np.percentile(all_abs_errs, 50):.1f} / "
              f"{np.percentile(all_abs_errs, 95):.1f} min")

    combined_caught = archetype_matched + (fallback_only)
    # fallback_only already subset of unmatched — add fallback_caught for
    # matched vehicles too (so we report "flag would have fired somehow")
    # Actually: archetype-matched flags fire via archetype; fallback is
    # independent signal. Worst case flag fires = matched+window_hit OR
    # fallback_caught. Simplest interpretation: count any sample where
    # EITHER signal would fire.
    either_fires = sum(
        1 for dow in samples_by_dow.values() for s in dow
        if s["fallback_caught"]
    )
    print(f"\nAny-signal coverage (flag fires, either path): "
          f"{either_fires} ({100*either_fires/total:.0f}%)")


if __name__ == "__main__":
    main()
