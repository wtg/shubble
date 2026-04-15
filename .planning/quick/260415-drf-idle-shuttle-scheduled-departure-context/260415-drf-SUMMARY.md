---
task: 260415-drf
subsystem: backend+frontend
tags: [eta, trip-tracking, idle-binding, schedule-ui]
requirements: [DRF-01, DRF-02]
dependency_graph:
  requires:
    - quick-260415-0vt (boundary-stop la drop within 5 min of actual_departure)
    - quick-260414-m1p (latest close-approach la per vid/stop)
    - 02-01 (per-trip ETA model with build_trip_etas + compute_trips_from_vehicle_data)
  provides:
    - defense-in-depth la < actual_departure guard at every entry-write site
    - idle-at-Union vid capture + next-scheduled-slot binding
    - "waiting" pill on schedule rows when idle vid is bound
  affects:
    - backend/worker/trips.py (build_trip_etas, compute_trips_from_vehicle_data)
    - frontend/src/schedule/Schedule.tsx (scheduled-row render)
    - frontend/src/schedule/styles/Schedule.css (waiting-pill style)
tech_stack:
  added: []
  patterns:
    - post-hoc invariant enforcement via entry-write guards
    - idle-filter side-channel (idle_at_union_by_route) drains into scheduled-trip pre-pass
    - setdefault semantics to enforce "first-wins, no double-assignment"
key_files:
  created:
    - tests/test_last_arrival_loop_scoping.py (+ 5 new tests, 11 → 16)
  modified:
    - backend/worker/trips.py
    - frontend/src/schedule/Schedule.tsx
    - frontend/src/schedule/styles/Schedule.css
decisions:
  - "Task 1 guard at entry-write time, not upstream: tolerates any caller passing loop_cutoff=None or a stale loop_cutoff — invariant holds post-hoc regardless of how last_arrivals were derived."
  - "Task 2 idle-vid capture hooks Filter 3 (no-movement) + Filter 1 (long-idle), gated on at_first_stop. Reality check: continuously-dwelling shuttles never fire Filter 1 because actual_departure = cluster_end = now; Filter 3 is the one that actually short-circuits them."
  - "Only FIRST idle vid per route claims the next scheduled slot (setdefault). Additional idle vids stay invisible, preserving the single-row-per-slot invariant."
  - "Idle binding only consumes unclaimed future slots; slots already claimed by active/dwelling shuttles are left alone."
metrics:
  duration_min: 55
  completed: "2026-04-15T14:30:00Z"
  task_count: 3
  commit_count: 5
---

# Quick Task 260415-drf: Idle-shuttle scheduled-departure context + Colonie false-Passed bug Summary

One-liner: Defense-in-depth la < actual_departure guard fixes Colonie false-"Passed"; idle shuttles at Union now get bound to their next scheduled slot and render a "waiting" pill on the schedule page.

## What Changed

### Task 1 — Defensive la < actual_departure guard (backend)

Added a post-hoc invariant guard in `build_trip_etas` that ensures no entry's `last_arrival` can ever appear strictly older than the trip's own `actual_departure`, regardless of which code path wrote it.

**Guard points in `backend/worker/trips.py`:**
1. **Pre-filter** (~line 259): `last_arrivals` is pruned of entries older than `actual_departure_dt` before any downstream pass runs. Catches the case where `loop_cutoff=None` (legacy callers) leaves the upstream 215-226 filter inactive.
2. **Clamp-anchor write** (line ~659): if the monotonic clamp drags a real detection's `la` down to an anchor that predates `actual_departure`, the entry is nulled instead of written.
3. **Interpolated backfill write** (line ~681): if `_interpolated_la` would fabricate a timestamp earlier than `actual_departure`, the entry is nulled rather than producing an impossible-earlier-than-departure "Passed" display.

**Source of truth for the cutoff:** parses `trip["actual_departure"]` (ISO string) at the top of the function; falls back to `loop_cutoff` if absent; no-op when both are None (preserves unassigned-scheduled-trip legacy behavior).

Implements the Colonie live-repro fix: active NORTH trip with `actual_departure=13:41` showed `COLONIE.last_arrival=13:33` (eight minutes before its own departure) — now surgically impossible.

### Task 2 — Idle-shuttle → next-scheduled-slot binding (backend)

Added a side-channel capture in `compute_trips_from_vehicle_data` that lets the idle-filter branches record (route → vid) before the trip-emission `continue`, then a post-loop pre-pass binds each captured vid to its route's next unclaimed future scheduled slot.

**Changes in `backend/worker/trips.py`:**
- Declared `idle_at_union_by_route: Dict[str, str]` next to `assigned_scheduled_times`.
- Hooked Filter 1 (long-idle) AND Filter 3 (no-movement) with `idle_at_union_by_route.setdefault(route, str(vid))` before the `continue`. Filter 3 is gated on `at_first_stop` so only Union-parked shuttles are captured.
- Added `idle_slot_assignments: Dict[tuple, str]` pre-pass before scheduled-trip emission: for each idle route, walks sorted `sched_deps`, skips past/claimed slots, and remembers the first unclaimed future slot.
- Scheduled-trip emission reads `idle_slot_assignments.get((route, iso))` and uses it as the row's `vehicle_id` (falls back to `None` when no binding). `trip_id` includes the vid suffix when bound, keeping frontend dedup keys stable across tick boundaries.

**Deviation from plan (Rule 3 auto-fix):** The plan specified hooking only Filter 1. In practice, a continuously-dwelling shuttle has `actual_departure = cluster_end = now_utc`, so Filter 1's `idle_seconds` is ~0 and it never fires — Filter 3 is what actually short-circuits real-world idle shuttles. Hooking both filters (with Filter 3 gated on `at_first_stop`) is necessary for the feature to work.

### Task 3 — "Waiting" pill on schedule rows (frontend)

`frontend/src/schedule/Schedule.tsx` — sibling render to the existing vehicle-badge, gated on `loopTrip?.status === 'scheduled' && loopTrip?.vehicle_id`. Renders nothing for active/completed/unassigned rows.

`frontend/src/schedule/styles/Schedule.css` — new `.waiting-pill` rule (10px uppercase, amber `#b08c00` to distinguish from source-live blue and trip-completed gray).

## Files Touched

| File | Type | Description |
|---|---|---|
| `backend/worker/trips.py` | modified | defensive la guard (Task 1) + idle-vid capture & scheduled-slot binding (Task 2) |
| `tests/test_last_arrival_loop_scoping.py` | modified | +5 new tests (3 for Task 1 guard, 2 for Task 2 binding) |
| `frontend/src/schedule/Schedule.tsx` | modified | waiting-pill render |
| `frontend/src/schedule/styles/Schedule.css` | modified | .waiting-pill style |

## Test Results

### New tests (all GREEN)

- `test_stale_la_dropped_by_defensive_guard` — Colonie live-repro: all-stale `last_arrivals` + `loop_cutoff=None` must not produce any `passed=True` on the output.
- `test_stale_la_never_resurrected_by_monotonic_clamp` — mixed stale+valid: stale SU la must be dropped even when a valid later anchor (GEORGIAN) could backfill; the la < actual_departure invariant must hold.
- `test_defensive_guard_is_noop_when_actual_departure_missing` — pure-scheduled trips with no vehicle data preserve legacy behavior.
- `test_idle_shuttle_assigned_to_next_scheduled_trip` — idle vid at Union binds to the NEAREST unclaimed future slot; later slots stay unassigned; past slots unaffected.
- `test_multiple_idle_shuttles_same_route_claim_sequentially` — only one idle vid per route claims the next slot (no double-assignment).

### Full suite

- `./.venv/Scripts/pytest.exe tests/ --ignore=tests/simulation/test_frontend_ux.py`: **94 passed**
- `./.venv/Scripts/pytest.exe tests/test_last_arrival_loop_scoping.py`: **16 passed** (11 pre-existing + 5 new)
- `cd frontend && npm run build`: PASS (no new TypeScript errors; one pre-existing dynamic-import warning unrelated to this change)

### Red-before-green validation

Confirmed Task 1 tests fail without the guard by stashing the `trips.py` changes and running the tests — both Task 1 tests FAIL with `la=13:33:00` leaking onto the output, then PASS once the guard is restored. Task 2 tests FAIL without the idle-binding implementation (empty `assigned` list).

## Live Verification (curl)

Backend/worker hadn't been restarted with the new code at time of summary (user will restart):

```bash
# Task 1: no active trip has la < actual_departure  (always held; invariant
# tightened by the guard for future occurrences)
curl -s http://localhost:8000/api/trips | python -c "...invariant check..."
Active-trip la-before-actual_departure violations: 0
```

After worker restart with the new code, the same curl check will continue to return 0 violations but now with the guarantee that the underlying code path cannot emit a violation.

```bash
# Task 2 (pending worker restart): scheduled trips with bound vid
Scheduled trips with vehicle_id: 0  (expected 0 pre-restart — no idle shuttle to bind)
# After restart and with a test-server shuttle parked at Union > 10 min:
# expected >= 1 scheduled trip with vehicle_id != null on its route's next slot
```

## Commits

| Commit | Message |
|---|---|
| `830806f` | test(quick-260415-drf): add RED regression tests for defensive la<actual_departure guard |
| `fd24946` | fix(quick-260415-drf): defensive la < actual_departure guard at every entry-write path |
| `0acc70c` | test(quick-260415-drf): add RED tests for idle-shuttle scheduled-slot binding |
| `7a11b98` | feat(quick-260415-drf): bind idle-at-Union shuttles to next scheduled slot |
| `8ab4142` | feat(quick-260415-drf): render "waiting" pill on idle-bound scheduled rows |

## Deviations from Plan

### [Rule 3 - Blocking] Plan specifies Filter 1 hook for idle-vid capture; reality requires Filter 3 too

- **Found during:** Task 2 implementation, debugging why RED tests still failed after initial implementation.
- **Issue:** The plan targeted Filter 1 (`at_first_stop and not_moving and idle_seconds > IDLE_THRESHOLD_SEC`) as the idle-capture hook point. But Filter 1's `idle_seconds = (now_utc - actual_departure).total_seconds()`, and for a continuously-dwelling shuttle, `actual_departure = _detect_vehicle_departures(...)[−1] = cluster_end = last_ping ≈ now`. So `idle_seconds ≈ 0`, Filter 1 never fires, and the feature was dead on arrival.
- **Fix:** Hooked BOTH filters. Filter 1 still captures (rare case where a gap-in-cluster shifts actual_departure to an older cluster). Filter 3 (no-movement) is the filter that actually fires for real dwelling shuttles; gated on `at_first_stop` so only Union-parked vids are captured, not mid-route-parked ones.
- **Files modified:** `backend/worker/trips.py` (both filter branches).
- **Commit:** `7a11b98` (single commit that includes both hooks and the pre-pass).

### [Rule 1 - Bug] Test assertions in Task 1 initially too strict

- **Found during:** First GREEN run of Task 1's two new tests — they still failed because STUDENT_UNION was being backfilled via `_interpolated_la` from GEORGIAN's valid later detection (a semantically correct behavior the plan's assertion didn't anticipate).
- **Fix:** Relaxed the SU-specific `last_arrival=None, passed=False` assertions to the plan's core invariant: "no stop's la predates actual_departure." The backfill may still mark earlier stops as `passed=True` via interpolation using a valid later anchor's timestamp — that's physically correct (if GEORGIAN was reached, SU was passed on the way). The guard's true contract is the timestamp invariant, not the passed-flag shape. Also added a separate "all-stale" test case that DOES satisfy the strict `passed=False` assertion (no valid anchor for the backfill to use).
- **Files modified:** `tests/test_last_arrival_loop_scoping.py`.
- **Commit:** `fd24946` (folded into the Task 1 GREEN commit).

## Auth Gates

None — all work was local file edits + pytest + npm build. No external auth required.

## Deferred Follow-ups

1. **Pre-existing test failure** (not fixed, out of scope):
   `tests/simulation/test_frontend_ux.py::TestUnionStopDisplay::test_frontend_deviation_logic_for_union`
   asserts `"deviationMinutes" in schedule_tsx` but current Schedule.tsx uses `delta`/`deviationSec` inside `getDepartureLabel`. See `deferred-items.md` in this task's folder. Reproduced on base commit before any edits; not caused by this task.

2. **Map-marker "waiting" indicator** (explicitly deferred in CONTEXT):
   The CONTEXT locked "Schedule page only. Add a pill/indicator on the next scheduled row for the route. Do not touch the live map marker in this change." — remains deferred for a future task if users also want a visible indicator on the map pin.

3. **Worker restart required** for the backend changes (Task 1 guard + Task 2 idle-binding) to take effect on the running services. User has handled restarts before; not blocking this task's completion. Post-restart verification:

```bash
# Should return 0 going forward even under prior-loop la leaks
curl -s http://localhost:8000/api/trips | jq '[.[] | select(.status=="active") | . as $t | .stop_etas | to_entries[] | select(.value.last_arrival != null and ($t.actual_departure != null) and (.value.last_arrival < $t.actual_departure))] | length'

# With a test-server shuttle idle at Union > 10 min, should return >= 1
curl -s http://localhost:8000/api/trips | jq '[.[] | select(.status=="scheduled" and .vehicle_id != null)] | length'
```

## Self-Check: PASSED

**Files created/modified (all exist):**
- FOUND: backend/worker/trips.py
- FOUND: tests/test_last_arrival_loop_scoping.py
- FOUND: frontend/src/schedule/Schedule.tsx
- FOUND: frontend/src/schedule/styles/Schedule.css
- FOUND: .planning/quick/260415-drf-idle-shuttle-scheduled-departure-context/deferred-items.md
- FOUND: .planning/quick/260415-drf-idle-shuttle-scheduled-departure-context/260415-drf-SUMMARY.md

**Commits (all present in git log):**
- FOUND: 830806f (RED Task 1 tests)
- FOUND: fd24946 (Task 1 GREEN guard)
- FOUND: 0acc70c (RED Task 2 tests)
- FOUND: 7a11b98 (Task 2 GREEN idle-binding)
- FOUND: 8ab4142 (Task 3 GREEN waiting-pill)
