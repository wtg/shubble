---
quick_task: 260415-oeb
plan: 01
type: execute
wave: 1
subsystem: backend/fastapi, frontend/live-map
tags: [break-detection, schedule-gap, mapkit, on-break-flag]
requirements:
  - D-01-on-break-flag
  - D-02-35-min-gap-threshold
  - D-03-frontend-greyout
dependency_graph:
  requires:
    - shared/aggregated_schedule.json (existing)
    - backend/fastapi/_build_locations_payload (existing)
    - frontend ShuttleIcon component (existing)
  provides:
    - on_break:bool on every /api/locations + SSE stream payload
    - SCHEDULE_GAP_THRESHOLD_SEC constant (35*60)
    - _load_today_gap_windows / _in_schedule_gap / _compute_gap_windows helpers
    - muted SVG shuttle icon variant (grayscale 0.8 + opacity 0.5)
  affects:
    - Live map marker rendering (muted style when on_break=true)
tech_stack:
  added: []
  patterns:
    - Per-day cached gap-window computation keyed on (y,m,d,tz) tuple
    - Baked-in SVG <filter> + group opacity for MapKit ImageAnnotation styling
    - Union-across-routes gap semantic (Phase 1 target: all-routes breaks only)
key_files:
  created: []
  modified:
    - backend/fastapi/utils.py
    - backend/fastapi/routes.py
    - frontend/src/types/vehicleLocation.ts
    - frontend/src/locations/components/LiveLocationMapKit.tsx
decisions:
  - on_break is a plain bool in Phase 1; break_reason deferred to Phase 2
  - 35-min threshold constant lives in backend/fastapi/utils.py:SCHEDULE_GAP_THRESHOLD_SEC
  - Muted marker = embedded SVG feColorMatrix saturate(0.2) + group opacity 0.5
  - Union-across-routes gap windows (safe for Phase 1: all gaps in scope share across routes)
metrics:
  duration_sec: 217
  tasks_completed: 2
  files_modified: 4
  completed_at: "2026-04-15T21:47:50Z"
---

# Quick Task 260415-oeb Summary

Phase 1 break detection — schedule-gap detector + `on_break` flag on `/api/locations`, plus muted shuttle marker on the live map when flagged.

## One-liner

`/api/locations` now emits `on_break: bool` (true when now falls inside a >=35-min scheduled-departure gap), and the frontend renders those shuttles with a grayscale + 50% opacity variant of the route-colored marker.

## What Shipped

### Task 1 — Backend gap detector + on_break flag (commit `ad58758`)

**`backend/fastapi/utils.py`:**
- `SCHEDULE_GAP_THRESHOLD_SEC = 35 * 60` — single location for the threshold constant.
- `_compute_gap_windows(sched_deps)` — pair-wise scan of sorted departures, returns `(gap_start, gap_end)` tuples for gaps `>= 35 min`.
- `_load_today_gap_windows(campus_tz)` — reads `shared/aggregated_schedule.json`, parses today's routes (JS `getDay()` indexing), unions per-route gap windows, caches per `(year, month, day, tz)` tuple. Day-rollover-safe.
- `_in_schedule_gap(now_utc, gap_windows)` — strict `start < now < end` check (scheduled departure minutes count as in-service on both sides).

No imports from `backend.worker.*` — FastAPI/worker boundary preserved by duplicating the small schedule-loading shape.

**`backend/fastapi/routes.py`:**
- `_build_locations_payload` now computes `gap_windows` once per call (not per vehicle) and sets `on_break: bool` on every vehicle entry.
- Since both REST `/api/locations` and SSE `/api/locations/stream` use this helper, one edit covers both transports.

### Task 2 — Frontend type + muted marker (commit `e04a490`)

**`frontend/src/types/vehicleLocation.ts`:**
- Added `on_break?: boolean` to `VehicleLocationData` (optional for backward safety — undefined treated as false).
- `VehicleCombinedData` inherits the field via its existing `VehicleLocationData & {...}` intersection; the spread in `LiveLocationMapKit.tsx:111` carries it through unchanged.

**`frontend/src/locations/components/LiveLocationMapKit.tsx`:**
- `getShuttleIconUrl(color, size, muted)` now takes a `muted` flag; cache key is `${color}|${size}|${muted ? "m" : "n"}`.
- Muted branch strips the inner `<svg>` wrapper and re-wraps in a filtered `<g>`:
  ```
  <defs><filter id="m"><feColorMatrix type="saturate" values="0.2"/></filter></defs>
  <g filter="url(#m)" opacity="0.5">{inner shuttle shapes}</g>
  ```
  `feColorMatrix saturate(0.2)` ≈ `grayscale(0.8)`, group `opacity="0.5"` matches D-03 exactly. Filter baked into the data-URL because MapKit `ImageAnnotation` doesn't expose a reliable CSS filter hook on the rendered `<img>`.
- Per-vehicle annotation build selects the muted variant when `vehicle.on_break === true`.
- `MapKitOverlays.tsx` already re-applies `annotation.url = overlay.url` on overlay updates, so transitions between normal/muted flow through automatically on each poll.

## Muted SVG Variant Construction (for Phase 2 reference)

The muted marker is constructed at icon-URL build time, not at render time. Steps:

1. Render the existing `<ShuttleIcon color size />` React component via `renderToStaticMarkup` → full `<svg>...</svg>` string.
2. Strip the outer `<svg>` open/close tags to get just the shape children.
3. Wrap in a new outer `<svg>` with `<defs><filter id="m"><feColorMatrix type="saturate" values="0.2"/></filter></defs>` and `<g filter="url(#m)" opacity="0.5">...</g>`.
4. Base64-encode and produce `data:image/svg+xml;base64,...`.
5. Cache by `(color, size, muted)` key so each unique variant is produced exactly once per session.

Phase 2 can add additional visual states (e.g. a red `break_reason=driver_logoff` variant) by extending the muted branch with another cache key suffix and parallel SVG composition.

## Field Contract (Phase 1)

```
GET /api/locations response:
{
  "<vehicle_id>": {
    ...existing fields...,
    "driver": { "id": "...", "name": "..." } | null,
    "on_break": bool        // true iff now ∈ any route's >=35min gap today
  },
  ...
}
```

`on_break` is **per-response uniform for Phase 1** (all vehicles in the geofence share the same value), because the in-scope gaps (Saturday lunch/dinner, Sat/Sun/Weekday dinner) are all-routes gaps. Phase 2 will add per-vehicle signals (driver assignment, Hungarian+break-column) that distinguish individual shuttles during rotation windows.

## Follow-up Notes

- **Phase 2** will add `break_reason: Optional[str]` (e.g. `"schedule_gap"`, `"driver_logoff"`, `"rotation"`) to distinguish break causes. The boolean remains sufficient for the target Phase 1 scenarios.
- **Phase 2** will handle weekday/Sunday lunch rotation where some routes have continuous coverage — that scenario is explicitly out of scope here and currently returns `on_break=false`.
- The `_GAP_WINDOWS_CACHE` is process-local in-memory only. Multiple uvicorn workers each pay one cold-load per day; acceptable because cold load is <5ms and the data only changes on midnight rollover.

## Verification

### Automated checks (all passed)
- Unit check: `SCHEDULE_GAP_THRESHOLD_SEC == 35*60`, `_compute_gap_windows([t0,t1,t2,t3])` returns the correct `(t2,t3)` pair, `_in_schedule_gap` correctly excludes boundaries. Output: `gap detector OK`.
- Real-data check on today's (Wednesday) schedule produced 2 gap windows (18:30→19:20 and 18:40→19:30, each 50 min — dinner break). Cache load succeeded.
- pytest: `./.venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/simulation/test_frontend_ux.py` → **97 passed** in 33.13s.
- Frontend: `npx tsc --noEmit` → clean (no errors).
- Frontend build: `cd frontend && npm run build` → succeeded, 89 modules transformed, dist bundle generated.

### Manual (deferred)
Full end-to-end smoke test with a running test stack + sim clock set to a Saturday schedule-gap minute was not exercised here — services may need a restart to pick up the new code (see below). The plan's manual verify steps are documented in `260415-oeb-PLAN.md` if a human-eyes pass is desired.

## Restart Notice

**Any currently running backend/worker containers will NOT pick up the new `on_break` field until restarted.** If `docker-compose up` is running in the background, restart the `backend` service:

```bash
docker-compose restart backend
```

Frontend is unaffected by backend restarts — it will start consuming `on_break` on its next /api/locations response.

## Deviations from Plan

None — plan executed exactly as written. All four done-criteria items for each task are satisfied.

## Deferred Issues

None. Plan scope fully landed.

## Known Stubs

None introduced. The `on_break` field is wired end-to-end from schedule source → backend → REST+SSE → type → marker rendering.

## Self-Check: PASSED

- [x] `backend/fastapi/utils.py` contains `SCHEDULE_GAP_THRESHOLD_SEC`, `_compute_gap_windows`, `_in_schedule_gap`, `_load_today_gap_windows` (verified via import + runtime check).
- [x] `backend/fastapi/routes.py` `_build_locations_payload` computes `gap_windows` once and writes `on_break` into every vehicle dict (verified by inspection).
- [x] `frontend/src/types/vehicleLocation.ts` has `on_break?: boolean` on `VehicleLocationData`.
- [x] `frontend/src/locations/components/LiveLocationMapKit.tsx` `getShuttleIconUrl` takes `muted` flag; per-vehicle build sets `muted = vehicle.on_break === true`.
- [x] Commit `ad58758` exists (Task 1 — backend).
- [x] Commit `e04a490` exists (Task 2 — frontend).
- [x] pytest passes 97/97.
- [x] tsc --noEmit passes clean.
- [x] npm run build succeeds.
