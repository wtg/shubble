---
name: Fix-forward — strip trip_id suffixes, gate LIVE on active-only, reorder DONE to top
description: Discussion-phase decisions for quick task 260415-emp
type: quick-task-context
---

# Quick Task 260415-emp Context

**Gathered:** 2026-04-15
**Status:** Ready for planning

<domain>
## Task Boundary

Three tightly-coupled UI regressions from quick-260415-drf + one new UX ask. User explicitly picked the "fix forward" path (preserve idle-binding feature intent; don't mutate trip_id; fix frontend badge gating; reorder DONE).

**Regression 1 — Duplicate rows per slot (frontend "many stale trips")**
- Cause: commit 7a11b98 in backend/worker/trips.py appended `:vid`/`:done` to trip_id. Frontend dedups on trip_id → completed + active versions of the same slot now coexist as 2 rows.
- Live evidence (snapshot):
  ```
  NORTH @ 14:30:00
    NORTH:14:30:00:000000000000001        active
    NORTH:14:30:00:000000000000001:done   completed (ghost)
  WEST @ 14:25:00
    WEST:14:25:00:000000000000003:done    completed (shuttle 003 prior loop)
    WEST:14:25:00:000000000000004         active (shuttle 004 now in this slot)
  ```

**Regression 2 — LIVE badge on future scheduled trips**
- Cause: quick-260415-drf's idle-binding assigned `vehicle_id` to FUTURE scheduled trips (status=scheduled, vehicle_id=<idle shuttle>). Frontend's LIVE badge logic at Schedule.tsx:1013 and 1046 fires whenever `loopTrip` has an `etaTime` — now LIVE appears on future scheduled rows because backend computes forward-projected ETAs using the bound vehicle.
- User's words: "Future trips say 'live' when they should say 'sched' from the static schedule. These 'live' numbers are all based off current time and 'last'."

**Regression 3 — Stale Colonie "Passed"**
- Downstream of Regression 1: the user sees a `DONE` ghost row for the current slot where Colonie WAS passed in the prior loop (valid historical record). Fixing trip_id dedup collapses the ghost and the false-Passed disappears as a visible artifact.

**New UX ask — DONE trips at top**
- User: "all DONE trips should be moved to the top to not confuse users."
- Today DONE rows sit in their chronological position interspersed with active/upcoming. Autoscroll skips past them but they remain in the list causing visual clutter.

</domain>

<decisions>
## Implementation Decisions

### LIVE-vs-SCHED gating (frontend)
- **Gate LIVE badge on `loopTrip.status === 'active'`.** If status is `scheduled`, never render LIVE — even when vehicle_id is bound by idle-binding. Scheduled rows keep their static schedule-time display and either render nothing (simpler) or the existing scheduled-time formatting; no SCHED badge added in this task (user approved Recommended option, not the "+SCHED badge" variant).
- Scope: Schedule.tsx sites at approximately lines 1003-1048 (two `<span className="source-badge source-live">LIVE</span>` sites). Wrap each LIVE emission in a status check.
- Do NOT remove the "waiting" pill introduced in 8ab4142 — that's a different surface and still desired for idle-bound scheduled rows.

### DONE trips placement
- **Sort DONE rows to the TOP of the per-route timeline**, preserving relative order within completed (chronological, most-recent last). Active/upcoming rows keep their current ordering below.
- Scope: the trip-rendering loop in Schedule.tsx (around the `{timelineItems.map(...)}` region). Add a pre-sort step: partition into `[DONE, ...rest]`, concat.
- Existing `.trip-completed-badge` DONE marker is the visual separator; no new section header. This is Recommended option 1 (Sort-only; not Option 2's separate collapsible section).

### Backend trip_id shape
- **Strip all suffixes. Canonical trip_id = `{route}:{departure_time}`.** No `:vid`, no `:done`.
- `vehicle_id` remains a separate field on the trip record (already the case). Idle-binding continues to populate `vehicle_id` on the scheduled trip; no trip_id mutation needed to disambiguate, because the frontend dedup on unsuffixed trip_id is already correct (prior state).
- Scope: trips.py emissions — 2 sites approximately:
  1. Scheduled-trip emission (around line ~1076 from prior SUMMARY) — remove `:{vid}` suffix.
  2. Completed-trip emission — remove `:done` suffix.
- Net effect: completed + active + scheduled for the same `(route, departure_time)` collapse to ONE trip_id on the wire. The trip's status field (`active|completed|scheduled`) is the only per-lifecycle-state discriminator needed. Backend should ensure only ONE row per (route, departure_time) appears in /api/trips output at any given time — if that's not already guaranteed, verify and fix in the same task.

### Claude's Discretion
- Whether to add a single `SCHED` badge next to scheduled-trip times (not needed per user; revisit if UI feels ambiguous).
- Whether to hide DONE rows older than N minutes (user didn't ask; keep all DONE visible for now — sort-to-top is the whole ask).
- If backend produces duplicate `(route, departure_time)` trip_ids for legitimate reasons (e.g. same slot re-run after a reset), document and leave as follow-up; this task assumes one row per slot is the invariant.

</decisions>

<specifics>
## Specific Ideas

### Files most likely touched
- `backend/worker/trips.py` — strip trip_id suffixes. Two emission sites from prior SUMMARY.
- `frontend/src/schedule/Schedule.tsx` — gate LIVE badge on `loopTrip?.status === 'active'`; sort DONE to top of timelineItems.
- Possibly new tests in `tests/test_last_arrival_loop_scoping.py` or a new `tests/test_trip_id_shape.py` — one row per (route, departure_time); trip_id has no `:vid`/`:done` suffix.

### Existing surfaces worth preserving
- The "waiting" pill (8ab4142) — still valid, still renders when `loopTrip.status === 'scheduled' && loopTrip.vehicle_id`.
- The defensive la < actual_departure guard (fd24946) — still valid; no changes here.

### Verification
- `curl -s http://localhost:8000/api/trips | python -c "from collections import Counter; import sys, json; trips=json.load(sys.stdin); c=Counter((t['route'], t['departure_time']) for t in trips); print('duplicates:', {k:v for k,v in c.items() if v>1})"` → should print empty dict after fix.
- `curl ... | python -c "...; [print(t['trip_id']) for t in trips[:10]]"` → no `:done` or `:000000000000001` suffixes.
- Open `http://localhost:3000/schedule` — no LIVE badges on future scheduled rows; DONE rows appear at the top; no duplicate rows per slot.

</specifics>

<canonical_refs>
## Canonical References

- quick-260415-drf SUMMARY: `.planning/quick/260415-drf-idle-shuttle-scheduled-departure-context/260415-drf-SUMMARY.md` (what was added and why)
- Live diagnosis (this conversation): 28 trips in /api/trips with 2 duplicate (route, dep) pairs, both caused by trip_id suffix collision
- Schedule.tsx LIVE-badge sites: approximately lines 1007-1048, two `source-live` renders

</canonical_refs>
