---
name: Simulate shuttle break behavior in test server
description: Discussion-phase decisions for quick task 260415-nm1
type: quick-task-context
---

# Quick Task 260415-nm1 — Context

**Branch:** test-server-break-simulation (not test/server/test- …; branch name only)
**Gathered:** 2026-04-15
**Status:** Ready for planning

<domain>
## Task Boundary

Add break-taking behavior to `test/server/` so we can develop + validate backend break-detection without waiting for real Samsara data. No backend detection code yet — this task is strictly simulation-side. Breaks are driven by the route schedule and day-of-week pattern.

Operating model to simulate (source of truth: user's friend who takes the shuttle):

| Day | Fleet per route | Lunch coverage | Dinner coverage |
|---|---|---|---|
| Saturday | 1 shuttle, all day | lunch = 1-hour schedule gap (single shuttle takes break, no coverage) | dinner = 1-hour schedule gap (same shuttle) |
| Sunday | 2 during lunch rotation | one on break, the other covers (schedule stays continuous) | 1 shuttle; dinner = schedule gap |
| Weekday | 3 during lunch rotation | two cover, one breaks; round-robin through the lunch window | 1 shuttle; dinner = schedule gap |

Single-shuttle windows (Sat all day, Sun/Weekday dinner, and the single shuttle after the lunch rotation ends) produce SCHEDULE GAPS — visible in the aggregated schedule as gaps >~25 min — and need no per-shuttle "on break" flag in the sim because there's simply no shuttle active. Multi-shuttle rotation windows (Sun lunch 2-shuttle rotation, Weekday lunch 3-shuttle rotation) need per-shuttle break-taking so a shuttle individually goes on break while others cover.

</domain>

<decisions>
## Implementation Decisions

### Break representation (D-01)
- **Drive to break spot off-route, sit stationary.** Shuttle physically drives to a fixed off-route break location (configurable per route; default a plausible RPI-adjacent parking lot coord that's off both NORTH and WEST polylines), sits there for the break duration, then drives back to the first stop on-route to resume. GPS pings continue during the break.
- Exercises Filter 2 (off-route) + Filter 3 (stationary) simultaneously; visually debuggable (shuttle marker visible at break spot).
- **Not chosen:** closing driver_vehicle_assignment. Future work if we want to exercise the driver-assignment gate — noted in follow-ups but NOT in scope for this task.

### Break triggering (D-02)
- **Automatic — derived from schedule + per-day rotation pattern.** No manual API surface this task. Test server reads today's schedule, spawns the right fleet count, and each shuttle's scheduler thread queues a BREAK action at the appropriate time according to the rotation pattern.
- Lunch rotation (Sun 2-shuttle / Weekday 3-shuttle): distribute break slots round-robin across the lunch window (~12:00-14:30 is a reasonable placeholder; exact window to be derived from schedule density or configurable constant).
- Each break is ~45 min (configurable constant).
- Saturday / single-shuttle windows: no BREAK action needed — the schedule itself has the gap (setup_schedule_shuttles already handles this since future_deps is filtered to scheduled times).

### Per-day fleet count (D-03)
- **Apply now.** setup_schedule_shuttles spawns: Saturday=1, Sunday=2 (during lunch window)/1 (outside), Weekday=3 (during lunch window)/1 (outside).
- The "during lunch" vs "outside lunch" distinction is resolved by: during lunch window spawn all N; outside it, N-of-them continue running, the rest idle out naturally as they complete their current loop and match no more future slots. Simplest implementation. Fine-grained timing (exactly when the extra shuttles join/leave) not in scope — the lunch rotation window is the source of truth.

### Break spot coordinate (Claude's discretion)
- Pick one shared break spot coordinate (e.g., a parking lot near Union that's off both NORTH and WEST polylines by ≥100m). Same coord for all routes for simplicity. Location is a constant in `test/server/shuttle.py` or `test/server/shuttles.py`.

### Claude's Discretion
- Exact lunch window times (e.g., 11:30-13:30 or 12:00-14:30). Pick a reasonable window that matches typical RPI dining hours.
- Round-robin break ordering: simple cyclic by shuttle index is fine. No need for load-balancing or behavioral optimization in this task.
- Whether a BREAK action auto-re-queues a LOOPING after break ends, or the scheduler thread handles that transition. Executor picks whichever fits the existing ShuttleAction machinery best.

</decisions>

<specifics>
## Specific Ideas

### Break spot coord candidate
- Parking lot east of the athletic complex at roughly (42.735, -73.665) — off-route for both NORTH (which goes to STAC area) and WEST (which goes further west). Verify haversine distance to nearest polyline point is >100m before finalizing.

### Existing plumbing to reuse
- `ShuttleAction` enum + `shuttle.push_action()`: add a `BREAK` action variant.
- `_schedule_worker` thread (test/server/shuttles.py:227): currently queues LOOPING at each scheduled departure. Extend to also queue BREAK at rotation-determined times.
- `setup_schedule_shuttles`: already day-aware via `_load_today_schedule`. Spawn count should branch on `js_day` (0=Sun, 6=Sat).

### Verification
- Run test server with simulated "today = Saturday" override (via a dev env var or request param) → observe 1 shuttle spawned per route, schedule gaps honored naturally.
- Run with "today = weekday" during lunch window → observe 3 shuttles spawned, one at a time driving off-route to break spot, returning after break.
- `curl /api/shuttles` during break window → 1 shuttle has `state: break` (or equivalent) at the break-spot coord.

</specifics>

<canonical_refs>
## Canonical References

- `test/server/shuttles.py:setup_schedule_shuttles` — fleet-size + schedule wiring
- `test/server/shuttles.py:_schedule_worker` — per-shuttle departure-time queue
- `test/server/shuttle.py:ShuttleAction` — action enum we extend
- `shared/aggregated_schedule.json` — source of schedule-gap info (dinner gaps already present)
- Conversation history leading up to this task (break-detection design discussion).

</canonical_refs>
