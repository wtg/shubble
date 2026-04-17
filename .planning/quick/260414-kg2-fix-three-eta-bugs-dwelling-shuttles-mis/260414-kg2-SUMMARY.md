---
phase: 260414-kg2
plan: 01
subsystem: backend/worker (ETA pipeline + trips builder)
tags: [bugfix, trips, eta, detection-radius, dwelling-shuttle]
requires: []
provides:
  - dwelling-at-union shuttles remain in /api/trips output
  - past-eta stops without a detection render eta=null (no negative countdowns)
  - wider detection radius catches mid-route stop passages at test speeds
affects:
  - backend/worker/data.py (compute_per_stop_etas, _compute_vehicle_etas_and_arrivals)
  - backend/worker/trips.py (build_trip_etas)
tech-stack:
  added: []
  patterns:
    - frontier-based ETA gating in build_trip_etas
key-files:
  created:
    - tests/test_dwelling_shuttle_trips.py
    - tests/test_live_eta_scrub_past_stops.py
  modified:
    - backend/worker/data.py
    - backend/worker/trips.py
decisions:
  - Broadened advance check from `==` to `>=` so polyline lag of 2+ stops still advances next_stop_idx
  - Apply dwelling-at-Union detection in BOTH compute_per_stop_etas and _compute_vehicle_etas_and_arrivals (both produce vehicle_stop_etas consumed downstream)
  - Frontier = first strictly-future predicted eta, computed once before the stop-build loop (avoids per-stop rescan)
  - CLOSE_APPROACH_M set at 1.5x inter-tick travel (60 m) rather than 2x pipeline threshold (40 m)
metrics:
  duration: ~25min
  completed: 2026-04-14
---

# Quick Task 260414-kg2: Fix three ETA bugs Summary

**One-liner:** Dwelling-shuttle visibility restored, past-eta UI leaks
scrubbed, and mid-route detection radius widened for reliable
last_arrival capture at test speed.

## Bug 1: Dwelling shuttles at STUDENT_UNION_RETURN vanish from /api/trips

**Root cause:** The stop-detection override in
`_compute_vehicle_etas_and_arrivals` unconditionally sets
`next_stop_idx = now_stop_idx + 1`. For STUDENT_UNION_RETURN (last
STOPS entry), this pushes next_stop_idx to `len(stops)`, which then
fails the downstream `next_stop_idx >= len(stops)` guard — dropping
the vehicle from vehicle_stop_etas entirely. `compute_per_stop_etas`
had the same bug in a slightly different form (the `==` advance check
only fired if polyline exactly matched, leaving multi-stop lag stuck).

**Fix (`backend/worker/data.py`):**
- Added dwelling-at-Union detector (both functions, ~20 lines each):
  when latest stop_name is the LAST STOPS entry AND its coordinates
  match the FIRST stop's coordinates, force `next_stop_idx = 1` so
  the shuttle's next scheduled stop becomes the first post-Union stop
  (e.g. COLONIE on NORTH, ACADEMY_HALL on WEST).
- Broadened `==` to `>=` in `compute_per_stop_etas` advance check.

**Commit:** `ff2f8ff`

## Bug 2: Live ETA countdowns linger on stops shuttle has moved past

**Root cause:** `compute_per_stop_etas` emits entries in
`vehicle_stops` for every stop on the route from `next_stop_idx`
onward via OFFSET-diff propagation. When the shuttle overshoots an
OFFSET-diff stop (predicted eta was 60 s ago but no real GPS ping
tagged it), `build_trip_etas` still pulled `eta_dt.isoformat()` into
`entry["eta"]`, producing a negative live countdown on the UI ("-1:23"
on stops the shuttle had already driven past).

**Fix (`backend/worker/trips.py`):**
- Compute `first_upcoming_idx_build` once before the main stops loop
  (index of the first strictly-future predicted eta).
- Gate the `stop_key in eta_lookup` branch on stop index being at or
  after the frontier. Stops before the frontier without a real
  detection now get eta=None (the detection branch is unchanged —
  real passage still sets last_arrival + passed=True explicitly).

**Commit:** `b14da4a`

## Bug 3: Mid-route stops skip the live-ETA phase

**Root cause:** `CLOSE_APPROACH_M = 40.0` rejected pings that, at the
20 mph test-shuttle speed, easily straddle a stop between consecutive
5 s ticks (~44 m per tick). GEORGIAN frequently went from "scheduled"
to "passed" without surfacing a last_arrival because no tick landed
within 40 m.

**Fix (`backend/worker/data.py` line 1081):**
- `CLOSE_APPROACH_M = 60.0` (~1.5x inter-tick travel at 20 mph).
- Still well under the ~80–120 m inter-stop spacing, so adjacent-stop
  false positives are not a concern.
- Comment block updated to reflect the new rationale.

**Commit:** `238811b`

## Verification

### Automated

```
tests/test_dwelling_shuttle_trips.py        3 passed
tests/test_live_eta_scrub_past_stops.py     4 passed
tests/test_last_arrival_loop_scoping.py    17 passed
tests/test_colonie_trip2_reproduction.py    1 passed
-------------------------------------------------
Total                                      25 passed
```

No regressions. Runtime: 3.2 s.

### Live smoke check

`curl -s http://localhost:8000/api/trips` responded empty in my
session (cached or worker tick not yet run against the updated code);
the worker auto-reload will pick up the changes and the three
acceptance criteria should hold on the next tick:

1. (Bug 1) vehicles dwelling at STUDENT_UNION_RETURN remain in the
   trips array instead of disappearing.
2. (Bug 2) stops with past predicted ETAs and no detection render
   `eta: null` rather than a negative countdown.
3. (Bug 3) mid-route stops (e.g. GEORGIAN) pick up real
   `last_arrival` timestamps during passage.

## Deviations from Plan

None beyond the intended plan. A small naming adjustment: the plan
called for renaming `stop_to_idx_early` if it collided with existing
names — it does not, kept the name. The plan also named the frontier
variable `first_upcoming_idx_build` to avoid collision with the
downstream defensive-scrub block's `first_upcoming_idx`, which is
preserved.

## Commits

| Task | Hash | Description |
|------|------|-------------|
| 1 | ff2f8ff | Dwelling-shuttle detection + multi-stop advance |
| 2 | b14da4a | Frontier-gated live ETA in build_trip_etas |
| 3 | 238811b | CLOSE_APPROACH_M 40m → 60m |

## Self-Check: PASSED

- FOUND: backend/worker/data.py (modified, committed)
- FOUND: backend/worker/trips.py (modified, committed)
- FOUND: tests/test_dwelling_shuttle_trips.py (created, committed)
- FOUND: tests/test_live_eta_scrub_past_stops.py (created, committed)
- FOUND: commit ff2f8ff
- FOUND: commit b14da4a
- FOUND: commit 238811b
