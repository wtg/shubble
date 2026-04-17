---
task: 260415-emp
subsystem: backend+frontend
tags: [trip-id, dedup, schedule-ui, live-badge, done-reorder]
requirements: [EMP-01, EMP-02, EMP-03]
key_files:
  modified:
    - backend/worker/trips.py
    - frontend/src/schedule/Schedule.tsx
    - tests/test_trip_id_shape.py (NEW)
metrics:
  commit_count: 5
  completed: "2026-04-15"
---

# Quick Task 260415-emp Summary

Fix-forward for 3 regressions from quick-260415-drf + 1 new UX ask.

## Commits

| SHA | Change |
|-----|--------|
| d531199 | strip :vid/:done from trip_id; tests (later refined — see below) |
| 5768a85 | gate LIVE badge on trip.status === 'active' |
| 596fa5f | sort DONE (completed) trips to top of timeline |
| feec663 | dedupe completed trips sharing the same slot |
| 8fb3ed3 | restore :done suffix for completed (needed for active+completed coexistence); update tests |

## What changed

### Backend (trips.py)
- `trip_id` no longer includes `:vid` (was causing cross-vehicle duplicate rows). Canonical forms:
  - active/scheduled: `{route}:{iso_dep}`
  - completed: `{route}:{iso_dep}:done` (disambiguates from new active loop on same slot)
- Added slot-dedup for completed trips: when two shuttles complete loops matching the same scheduled slot, keep only the one with the latest `actual_departure`.
- Preserved: defensive la-before-actual_departure guard (fd24946), idle-binding side-channel on `vehicle_id` field (7a11b98 minus the trip_id mutation).

### Frontend (Schedule.tsx)
- LIVE badge gated on `loopTrip.status === 'active'`. Scheduled trips — even idle-bound ones with a `vehicle_id` — render static schedule times / no LIVE badge.
- Per-stop expanded view: same status gate on `stopInfo` construction.
- DONE (completed) trips partition-sorted to the top of each route's timeline.
- Preserved: "waiting" pill on idle-bound scheduled rows (8ab4142).

## Tests (all passing)

- `tests/test_trip_id_shape.py` (NEW, 3 tests): banned-suffix regex self-test, no `:vid` on emitted trip_ids, trip_id uniqueness + per-status slot uniqueness.
- Full suite: 97 pass (1 pre-existing unrelated failure in tests/simulation/test_frontend_ux.py carried over).
- `cd frontend && npm run build` passes.

## Live verification (after backend + worker restart)

```
Total trips: 29
trip_ids with :vid suffix: 0
Duplicate trip_ids: 0
Duplicate active slots: 0
Duplicate scheduled slots: 0
```

Sample:
```
active     NORTH:2026-04-15T16:50:00+00:00        vid=000000000000002
completed  WEST:2026-04-15T16:50:00+00:00:done    vid=000000000000003
active     NORTH:2026-04-15T16:53:44+00:00        vid=000000000000001
scheduled  NORTH:2026-04-15T17:00:00+00:00        vid=None
```

The single `:done`-suffixed entry correctly distinguishes the just-completed WEST loop from the active WEST loop continuing in the next slot. No LIVE badges on scheduled rows.
