---
phase: quick-260415-nm1
plan: 01
subsystem: test-server
tags: [test-server, simulation, breaks, fleet-rotation]
requires:
  - existing ShuttleAction.ON_BREAK enum (kept — not a new variant)
  - existing _schedule_worker + setup_schedule_shuttles scaffolding
provides:
  - BREAK_SPOT off-route parking coord constant
  - three-phase ON_BREAK state machine (drive_out -> waiting -> drive_back)
  - Shuttle.interrupt_for_break() preemption helper
  - LUNCH_WINDOW_START / LUNCH_WINDOW_END / BREAK_DURATION_SEC constants
  - _fleet_size_for_today() and _lunch_window_today() helpers
  - per-day fleet sizing (Sat=1, Sun=2/1, Weekday=3/1) in setup_schedule_shuttles
  - rotation-aware ON_BREAK queuing inside _schedule_worker
affects:
  - /api/shuttles response now shows state="on_break" during three-phase break
  - test server now simulates off-route + stationary GPS data for break detection
tech-stack:
  patterns:
    - threading.Thread daemon helpers for time-delayed action pushing
    - straight-line path generation via _follow_path interpolation
    - preemption via interrupt_for_break to handle non-strict forever-loop LOOPING
key-files:
  modified:
    - test/server/shuttle.py
    - test/server/shuttles.py
decisions:
  - Upgraded existing ShuttleAction.ON_BREAK rather than adding a new variant — enum value "on_break" is already in /api/shuttles JSON, renaming would break downstream consumers
  - BREAK_SPOT = (42.7265, -73.672) — ~474m off NORTH polyline, ~498m off WEST, well past Filter 2's off-route threshold
  - Lunch window 11:30 AM - 2:30 PM campus time; round-robin slot length = 3h/fleet_size
  - Default break duration 2700s (45 min); override via DEV_BREAK_DURATION_SEC env var
  - Introduced Shuttle.interrupt_for_break because non-strict LOOPING never completes, so push_action(ON_BREAK) would queue behind a forever-looping action and never fire
metrics:
  duration: ~45min
  completed: 2026-04-15
  tasks_completed: 3
  files_modified: 2
---

# Phase quick-260415-nm1: Simulate shuttle break behavior in test server Summary

Upgraded the test server's `ON_BREAK` state machine to drive off-route to a fixed break spot, sit stationary, then drive back on-route — and wired per-day fleet sizing (Sat=1, Sun=2-lunch/1-other, Weekday=3-lunch/1-other) with rotation-driven break queuing so backend break-detection logic can be exercised against realistic simulated data.

## What Was Built

**test/server/shuttle.py (+110 lines)**

- Added module-level constant `BREAK_SPOT = (42.7265, -73.672)` — verified during planning to be >400m off both NORTH (474m) and WEST (498m) polylines, so the backend's Filter 2 (off-route) will trigger cleanly.
- Added `_BREAK_PATH_SEGMENTS = 1` (straight-line segment for drive_out / drive_back paths).
- Added three break-phase fields to `Shuttle.__init__`: `_break_phase` (`None | "drive_out" | "waiting" | "drive_back"`), `_break_wait_start` (wall-clock stamp for the stationary phase), `_break_return_route` (snapshotted at break start so later mutations of `_current_route` can't misdirect the return).
- Relaxed `push_action` so `ON_BREAK` may optionally carry a route (validated against `Stops.active_routes` if provided) for first-stop return-path generation.
- Replaced `_start_action` `ON_BREAK` branch to build a straight-line path from current location to `BREAK_SPOT`.
- Replaced `_handle_on_break` body with a three-phase state machine:
  - `drive_out`: follow straight-line path to `BREAK_SPOT`; on arrival, snap to `BREAK_SPOT`, set `_break_phase="waiting"`, stamp `_break_wait_start = time.time()`.
  - `waiting`: stationary; `duration` is measured from `_break_wait_start` so drive-out travel time doesn't eat into the stationary window.
  - `drive_back`: build path from `BREAK_SPOT` to the first on-route stop of `_break_return_route` (falls back to Union `(42.730711, -73.676737)` if route can't be resolved); on arrival, mark action completed and return to idle.
- Added `Shuttle.interrupt_for_break(route, duration)` — preemption helper that marks the current action as `interrupted`, drops the `_loop_dwell_until` timer, inserts a new `ON_BREAK` at the front of the queue tail, and forces `_current_action = None` so next tick promotes the break. Needed because non-strict `LOOPING` never completes on its own.

**test/server/shuttles.py (+138 lines)**

- Added `import os`.
- Added constants `LUNCH_WINDOW_START = (11, 30)`, `LUNCH_WINDOW_END = (14, 30)`, `BREAK_DURATION_SEC = int(os.environ.get("DEV_BREAK_DURATION_SEC", "2700"))`.
- Added `_fleet_size_for_today()` returning `(peak_fleet, base_fleet)` per D-03: Sat=(1,1), Sun=(2,1), Weekday=(3,1). Uses JS getDay semantics (matching `_load_today_schedule`).
- Added `_lunch_window_today()` returning today's `(window_start, window_end)` as campus-tz datetimes.
- Extended `_schedule_worker` signature with `fleet_size` + `shuttle_slot`. When `fleet_size>=2` a daemon thread waits until the shuttle's round-robin slot time (`window_start + slot_len * shuttle_slot` where `slot_len = total_window / fleet_size`), then calls `interrupt_for_break(route, duration=min(BREAK_DURATION_SEC, slot_len))`. Uses a 60-second grace so a slot whose nominal time is a fraction of a second in the past (e.g. server boots right at `window_start`) still schedules.
- Replaced the hard-coded 2-shuttle split (`future_deps[0::2] / future_deps[1::2]`) in `setup_schedule_shuttles` with N-way interleaved split where N = `peak_fleet` (in-lunch) or `base_fleet` (outside). Passes `peak_fleet` and `slot` through `_schedule_worker` kwargs so break timing is rotation-aware even when `fleet_size == base_fleet` at boot.
- Added docstring comment block inside `setup_schedule_shuttles` documenting fleet-sizing + break semantics.

## Verification Results

All three task verification scripts pass:

- **Task 1** — Shuttle state machine transitions `drive_out -> waiting -> drive_back -> idle` on a solo `ON_BREAK` with a 2-second duration; final location within 200m of Union. (Plan's canned verification script used `_speed = 20` default and a 15s deadline, which was too short — ~650m at 20mph through the pseudo-degree math exceeds 30s. Re-ran with `_speed = 500.0` and 30s deadline: all phases observed, final distance to Union 0m.)
- **Task 2** — `_fleet_size_for_today()` returns `(1,1)` for Sat, `(2,1)` for Sun, `(3,1)` for Mon; `LUNCH_WINDOW_START/END` and `BREAK_DURATION_SEC` constants correctly wired; `DEV_BREAK_DURATION_SEC` env override works.
- **Task 3** (end-to-end) — Simulated Monday at 11:30 AM with `DEV_BREAK_DURATION_SEC=1`. `setup_schedule_shuttles(strict=False)` spawns 6 shuttles (3 per active route × 2 routes NORTH/WEST). Slot 0 break fires immediately (grace period); at least one shuttle transitions to `on_break` and its location reaches `BREAK_SPOT` within 50m. Sample `to_dict()` during break returned `state='on_break'`, `loc=(42.7268, -73.6724)` (3m from `BREAK_SPOT`).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Added `Shuttle.interrupt_for_break()`**
- **Found during:** Task 3 integration verification
- **Issue:** `push_action(ON_BREAK)` appends to the queue; `_handle_idle` only promotes next action when `_current_action is None`. In non-strict mode, `LOOPING` actions have `single_loop=False` so they never complete — the shuttle loops forever. Any `ON_BREAK` queued after `LOOPING` would be stranded in the queue, never executed. The plan's integration test calls `setup_schedule_shuttles(strict=False)` and expects `on_break` to fire, which is impossible without preemption.
- **Fix:** Added `interrupt_for_break(route, duration)` on `Shuttle` that marks the in-progress action as `interrupted`, clears `_loop_dwell_until`, inserts `ON_BREAK` at `_action_queue[_action_index]`, and sets `_current_action = None`. Next `_handle_idle` tick promotes the `ON_BREAK`. Wired `_break_pusher` to call this instead of `push_action`.
- **Files modified:** test/server/shuttle.py (new method), test/server/shuttles.py (_break_pusher body)
- **Commit:** ed48a1e

**2. [Rule 1 - Bug] Widened break-start time check from strict `>` to `>= now - 60s` grace**
- **Found during:** Task 3 integration verification
- **Issue:** Plan's `if break_start > dev_now(CAMPUS_TZ)` excluded slot 0 (whose nominal start equals `window_start`) any time there was sub-millisecond drift between computing `window_start` and re-reading `dev_now`. Effectively: slot 0 never schedules unless you get lucky on timing.
- **Fix:** Changed to `if break_start >= now_for_check - timedelta(seconds=60)` so a slot whose nominal time is within the last 60 seconds still schedules. Since `_break_pusher` uses `if wait > 0: time.sleep(wait)`, a past `break_start` just fires immediately.
- **Files modified:** test/server/shuttles.py (_schedule_worker)
- **Commit:** ed48a1e

## Known Stubs

None — all break-behavior code is wired end-to-end. No placeholder data.

## Out of Scope / Follow-ups

- **Live fleet resizing at the lunch boundary.** Current implementation only spawns `base_fleet` outside lunch; if the server is running continuously and crosses the 11:30 AM boundary, the "extras" needed for the rotation don't auto-spawn. Explicitly out of scope per CONTEXT.md ("Fine-grained timing not in scope — the lunch rotation window is the source of truth"). A restart of the test server at any time during the window will correctly populate the peak fleet.
- **Driver-assignment closing as an alternative break representation.** D-01 chose off-route-drive only; closing `driver_vehicle_assignment` for the break window is a future enhancement for exercising the driver-assignment gate.
- **Per-route break-spot coords.** Currently a single `BREAK_SPOT` is shared across all routes for simulation simplicity. If we later want each route's break spot to look more plausible (e.g. different lots), make `BREAK_SPOT` a route-indexed mapping.
- **Frontend test failure unrelated to this task.** `tests/simulation/test_frontend_ux.py::TestUnionStopDisplay::test_frontend_deviation_logic_for_union` was already failing before this plan — it grep's `Schedule.tsx` for the literal string `deviationMinutes` which no longer exists after a recent refactor. Logged for future cleanup; not caused by this task.

## Operational Notes

**Test server restart required to pick up new behavior.** The test server process already running on main-branch code has the OLD `_schedule_worker` and `_start_action` in memory. To exercise the new break behavior end-to-end:

```bash
docker-compose --profile test restart server
# or run the mock server directly
./.venv/Scripts/python.exe -m test.server.server
```

**Smoke test commands**

```bash
# Dev server with 10-second breaks for fast iteration
DEV_BREAK_DURATION_SEC=10 ./.venv/Scripts/python.exe -m test.server.server

# Then, during the lunch window (or with DEV_TARGET_HOUR set to land inside it):
curl http://localhost:4000/api/shuttles | jq '.[] | {id, state, location, current_route}'
# Expect: one shuttle per route with state="on_break" and location near (42.7265, -73.672)
```

## Commits

| Task | Commit   | Description                                                                          |
| ---- | -------- | ------------------------------------------------------------------------------------ |
| 1    | 8de9fbd  | upgrade ON_BREAK to drive off-route, sit, return on-route                            |
| 2    | 47a5401  | per-day fleet sizing and rotation-driven ON_BREAK queuing                            |
| 3    | ed48a1e  | integrate break rotation + interrupt-aware LOOPING (interrupt_for_break + grace)     |

## Self-Check: PASSED

- [x] `test/server/shuttle.py` modified (BREAK_SPOT constant, break state machine, interrupt_for_break)
- [x] `test/server/shuttles.py` modified (constants, fleet-size helpers, _schedule_worker extension, N-way fleet sizing)
- [x] Commits `8de9fbd`, `47a5401`, `ed48a1e` exist on branch `test-server-break-simulation`
- [x] Task 1 verification passed (state transitions drive_out -> waiting -> drive_back -> idle)
- [x] Task 2 verification passed (fleet sizes, LUNCH_WINDOW, BREAK_DURATION_SEC env override)
- [x] Task 3 verification passed (6 shuttles on Monday lunch, shuttle reaches BREAK_SPOT, state=on_break surfaces via to_dict)
