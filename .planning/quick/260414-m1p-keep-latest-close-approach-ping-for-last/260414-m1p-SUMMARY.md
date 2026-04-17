---
phase: 260414-m1p
plan: 01
subsystem: backend/worker (per-vehicle last_arrivals dedup)
tags: [bugfix, trips, eta, last-arrival, loop-scoping, dedup]
requires: []
provides:
  - active trips surface last_arrival on stops the shuttle passes every loop (HFH on NORTH, WEST return stops)
  - dedup contract matches the downstream loop_cutoff filter contract in build_trip_etas
affects:
  - backend/worker/data.py (_compute_vehicle_etas_and_arrivals close_rows dedup)
tech-stack:
  added: []
  patterns:
    - latest-within-gate dedup (max-timestamp among already-validated rows)
key-files:
  created:
    - tests/test_latest_close_approach.py
  modified:
    - backend/worker/data.py
decisions:
  - Tiebreak close_rows by max(timestamp) not min(_dist_m) — both are post-60m-gate rows, latest is the only choice that passes loop_cutoff
  - Leave _dist_m column in place (unused by dedup now but cheap and useful for future distance-based debugging)
  - Scope stays confined to the dedup block: CLOSE_APPROACH_M stays 60.0, haversine pipeline unchanged, loop_cutoff contract untouched
metrics:
  duration: ~20min
  completed: 2026-04-14
---

# Quick Task 260414-m1p: Keep latest close-approach ping for last_arrival Summary

**One-liner:** Switched the (vehicle, stop) dedup in
`_compute_vehicle_etas_and_arrivals` from globally-closest distance to
latest timestamp among pings already inside the 60 m gate, so the
current loop's detection survives the downstream `loop_cutoff` filter.

## Root cause

`backend/worker/data.py` built `last_arrivals_by_vehicle` by sorting
`close_rows` by `_dist_m` ascending and keeping the first row per
`(vehicle_id, stop_name)`. Over a multi-loop day, this preferred
whichever prior loop's ping happened to land geometrically closest to
the stop — typically an earlier loop, not the current one.

The downstream `build_trip_etas` then applies a `loop_cutoff` filter
(`= actual_departure` for active trips, `prior_departure` for
completed) that drops any `last_arrival` older than the cutoff as a
stale prior-loop leak. Net: the current loop's ping was discarded by
the dedup, and the earlier loop's ping was discarded by `loop_cutoff` —
so the trip's `stop_etas[stop].last_arrival` was `None` even seconds
after the shuttle passed the stop.

Live evidence (vid=001, NORTH, HOUSTON_FIELD_HOUSE):

| time | dist to HFH | outcome under old dedup |
|------|-------------|-------------------------|
| 18:15:04 | 2 m  | kept (min distance wins) |
| 19:48:08 | 18 m | silently discarded (current loop) |

Result: HFH never surfaced a "Last:" timestamp on the active trip.

## Fix

### Dedup block (`backend/worker/data.py`)

**BEFORE:**
```python
if not close_rows.empty:
    # Sort by distance ASCENDING within each (vehicle, stop)
    # group, then drop_duplicates keeping the first row —
    # that's the closest-approach ping. Its timestamp is the
    # real last_arrival.
    close_rows = close_rows.sort_values('_dist_m', kind='mergesort')
    closest_approach = close_rows.drop_duplicates(
        subset=['vehicle_id', 'stop_name'], keep='first'
    )
    ts_series = pd.to_datetime(closest_approach['timestamp'], utc=True)
    for (vid, stop_key, ts) in zip(
        closest_approach['vehicle_id'].values,
        closest_approach['stop_name'].astype(str).values,
        ts_series,
    ):
```

**AFTER:**
```python
if not close_rows.empty:
    # Sort by timestamp ASCENDING within each (vehicle, stop)
    # group, then drop_duplicates keeping the LAST row —
    # that's the most recent within-60m ping. Its timestamp
    # is the current loop's last_arrival (see rationale
    # above). Every row in close_rows already passed the
    # 60 m CLOSE_APPROACH_M gate, so tiebreaking among
    # already-valid rows is semantically free — "latest" is
    # the only choice that survives the downstream
    # `loop_cutoff` filter in `build_trip_etas`.
    close_rows = close_rows.sort_values('timestamp', kind='mergesort')
    latest_approach = close_rows.drop_duplicates(
        subset=['vehicle_id', 'stop_name'], keep='last'
    )
    ts_series = pd.to_datetime(latest_approach['timestamp'], utc=True)
    for (vid, stop_key, ts) in zip(
        latest_approach['vehicle_id'].values,
        latest_approach['stop_name'].astype(str).values,
        ts_series,
    ):
```

### Rationale comment block rewrite (`backend/worker/data.py`)

The block above the dedup was rewritten to describe LATEST-within-60m
semantics and cite the downstream `loop_cutoff` contract as the reason.
Key substitutions:

- Section header: "use the ACTUAL CLOSEST-APPROACH timestamp" →
  "use the LATEST within-60m timestamp — not the max timestamp over
  every ping tagged with a given stop_name, and not the globally
  closest-distance ping either."
- Fix step 1 extended to state explicitly that the 60 m filter is now
  the ONLY geometric gate: "every row that survives it is a genuine
  arrival at the stop."
- Fix step 2 replaced entirely (was "SMALLEST distance ... Not max,
  not min — closest-approach") with a paragraph explaining why `max(timestamp)`
  is the right tiebreaker: it aligns with the downstream loop_cutoff
  filter, cites HFH as the concrete symptom, and notes that completed
  trips still behave correctly (their `loop_cutoff = prior_departure`
  selects the tip of the prior loop).

## Tests

New file: `tests/test_latest_close_approach.py`

| Test | Purpose |
|------|---------|
| `test_latest_close_approach_wins_over_closest` | Two pings, earlier-closer vs later-farther, both within 60 m. Asserts later ts wins. Fails pre-fix with old "closest-distance" code. |
| `test_both_pings_out_of_range_yields_no_last_arrival` | Confirms the 60 m CLOSE_APPROACH_M gate still works independently of the tiebreaker. |
| `test_single_within_range_ping_still_surfaces` | Smoke check that the trivial single-row case still produces the expected last_arrival. |

All tests use real NORTH stop coordinates from `shared/routes.json`
(HOUSTON_FIELD_HOUSE, COLONIE) so `stop_coord_lookup` resolves, and
mock `predict_eta` to isolate the last_arrivals path from the velocity
predictor.

## Test results

```
./.venv/Scripts/python.exe -m pytest tests/test_latest_close_approach.py \
  tests/test_last_arrival_loop_scoping.py \
  tests/test_dwelling_shuttle_trips.py \
  tests/test_live_eta_scrub_past_stops.py -v

============================= test session starts =============================
collected 21 items

tests/test_latest_close_approach.py::test_latest_close_approach_wins_over_closest PASSED
tests/test_latest_close_approach.py::test_both_pings_out_of_range_yields_no_last_arrival PASSED
tests/test_latest_close_approach.py::test_single_within_range_ping_still_surfaces PASSED
tests/test_last_arrival_loop_scoping.py ... 11 tests PASSED
tests/test_dwelling_shuttle_trips.py ... 3 tests PASSED
tests/test_live_eta_scrub_past_stops.py ... 4 tests PASSED

============================= 21 passed in 3.23s ==============================
```

Pre-fix RED confirmation:
`test_latest_close_approach_wins_over_closest` fails with
`AssertionError: Expected latest ping 2026-04-10T19:48:00+00:00 to
win dedup, got 2026-04-10T18:15:00+00:00`, proving the bug.

Post-fix GREEN: all 21 tests pass including the 18 pre-existing
regression tests (11 in test_last_arrival_loop_scoping.py, 3 in
test_dwelling_shuttle_trips.py, 4 in test_live_eta_scrub_past_stops.py).

Note: the plan estimated 17 tests in `test_last_arrival_loop_scoping.py`
but the file contains 11 — the 17 figure was a stale estimate. The
important regression coverage (loop_cutoff contract, spurious-scrub,
boundary-stop detection, backfill) is all intact.

## Live sanity

`curl -s http://localhost:8000/api/trips` returned valid trip JSON
after commit, confirming the worker auto-reload picked up the change.
Deeper verification (confirming HFH flips to non-null `last_arrival`
on active NORTH trips after a shuttle pass) requires a Redis flush
and at least one full loop of mock-shuttle telemetry, which was not
run as part of this task.

## Commits

- `db276e3` - fix(worker): keep latest close-approach ping for last_arrival dedup

## Deviations from Plan

None — plan executed exactly as written. The only minor adjustment
was a pytest environment bootstrap: the worktree was missing a `.env`
file (needed by `backend.config.Settings` for `DATABASE_URL`), so one
was copied from the main repo. The `.env` is gitignored and was not
part of the commit.

## Self-Check: PASSED

- `backend/worker/data.py` modified (contains `latest_approach = close_rows.drop_duplicates(..., keep='last')` sorted by `timestamp` ASC): FOUND
- `tests/test_latest_close_approach.py` created (3 tests, all passing): FOUND
- Commit `db276e3` in git log: FOUND
- No changes to `backend/worker/trips.py`, `CLOSE_APPROACH_M`, or the haversine pipeline: VERIFIED (git diff scope limited to lines 1105-1141 + 1192-1206 of data.py, plus the new test file)
