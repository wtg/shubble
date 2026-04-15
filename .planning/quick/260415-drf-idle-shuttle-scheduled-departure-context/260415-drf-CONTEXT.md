---
name: Idle-shuttle scheduled-departure context + Colonie false-Passed bug
description: Discussion-phase decisions for quick task 260415-drf
type: quick-task-context
---

# Quick Task 260415-drf: Idle-shuttle scheduled-departure context + Colonie false-Passed bug - Context

**Gathered:** 2026-04-15
**Status:** Ready for planning

<domain>
## Task Boundary

Two issues on schedule/UI for shuttles (verified via live test-server :4000, backend :8000, frontend :3000):

1. **FEATURE — Idle-shuttle scheduled departure**: When a shuttle is idle at Union and hasn't departed yet, the schedule page should associate it with its next scheduled departure so students see that a shuttle is ready and waiting for a specific upcoming slot. Today the `scheduled` trips in `/api/trips` have `vehicle_id: null` and there is no other signal tying the idle shuttle to its next slot.

2. **BUG — Stale "Passed" on Colonie**: Active NORTH trip (vid=000000000000002) that departed at minute 41 shows `COLONIE.last_arrival=33+00` — 8 minutes BEFORE actual_departure. The shuttle just left Union and hasn't reached Colonie yet, but the UI treats Colonie as Passed because `last_arrival` is populated. Root cause: per-trip `stop_etas` inherit `last_arrival` timestamps from rolling history instead of being scoped to the current trip's window. (Live data confirmed — see gray-area block in conversation.)

</domain>

<decisions>
## Implementation Decisions

### Idle-shuttle UI location
- **Schedule page only.** Add a pill/indicator on the next scheduled row for the route, e.g. "001 waiting → leaves 2:30". Do not touch the live map marker in this change.

### Idle-shuttle → trip association wiring
- **Backend assigns `vehicle_id` to the next "scheduled" trip** for each idle Union shuttle. `/api/trips` remains the single source of truth. Frontend just renders `vehicle_id` on the scheduled row when present.
- Scope: in `backend/worker/trips.py` (trip computation), after the current active-trip assignments are made, iterate remaining idle vehicles (those near Union with no active trip) and assign each to the next `status=scheduled` trip for their historical route.
- Idle-shuttle detection: vehicle is within Union proximity threshold AND has no current active trip AND has recent GPS pings (not stale). Reuse existing Union-proximity logic from `shared/stops.py` / `backend/worker/data.py` if present.

### Colonie false-Passed bug fix
- **Backend: drop `last_arrival` older than trip's `actual_departure`** when building per-trip `stop_etas`.
- Scope: in `backend/worker/trips.py`, at the point where `stop_etas[stop]['last_arrival']` is populated, guard with `if last_arrival_dt >= actual_departure_dt`. When the trip has no `actual_departure` yet (pure scheduled trip), keep existing behavior.
- This is a surgical fix at the data source — no frontend change needed. `passed` flag derivation downstream will flow correctly.

### Claude's Discretion
- Exact pill copy for the idle-shuttle indicator (e.g. "waiting — leaves 2:30" vs "001 @ Union → 2:30"). Plan/executor picks a clean short format.
- Union-proximity threshold for idle detection — reuse whatever constant is already used in the stop-centric pass if one exists; otherwise 60m (matches CLOSE_APPROACH_M).
- Whether to also zero out `passed: True` on a stop whose last_arrival was just dropped (likely yes — keeping `passed: True` with `last_arrival: None` would be inconsistent).

</decisions>

<specifics>
## Specific Ideas

### Live evidence of the bug (snapshot 2026-04-15 ~13:54 UTC)
```
ACTIVE NORTH trip vid=000000000000002 dep=00+00 actual=41+00
  STUDENT_UNION: last_arrival=33+00    ← stale, before actual_departure
  COLONIE:       last_arrival=33+00    ← stale, before actual_departure
  GEORGIAN:      eta=07+00              ← correct, forward-looking
```

### Files most likely touched
- `backend/worker/trips.py` — trip computation; both the idle-assignment and the last_arrival guard land here
- `frontend/src/schedule/Schedule.tsx` — render new "waiting" pill on scheduled row when vehicle_id is populated
- `frontend/src/hooks/useTrips.ts` — Trip type may need no change if vehicle_id is optional already; otherwise update type
- `frontend/src/schedule/styles/Schedule.css` — style the waiting pill

### Verification
- `curl http://localhost:8000/api/trips | jq '.[] | select(.status=="scheduled") | .vehicle_id'` — should return non-null for the next scheduled trip per route when a shuttle is idle
- `curl http://localhost:8000/api/trips | jq '.[] | select(.status=="active" and .route=="NORTH") | .stop_etas | to_entries[] | select(.value.last_arrival != null) | "\(.key): \(.value.last_arrival)"'` — no stale entries older than the trip's `actual_departure`

</specifics>

<canonical_refs>
## Canonical References

- `backend/worker/trips.py` lines 241, 280, 344 (existing comments about trip-boundary/passed-flag edge cases — same class of issue)
- `.planning/autonomous/PERF-FIXES-PLAN.md` (prior session that shipped 8 perf improvements, left this unrelated bug)
- Live test state: test-server :4000, backend :8000 all running; 4 shuttles on NORTH/WEST

</canonical_refs>
