---
quick_task: 260415-oeb
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - backend/fastapi/utils.py
  - backend/fastapi/routes.py
  - frontend/src/types/vehicleLocation.ts
  - frontend/src/locations/components/LiveLocationMapKit.tsx
autonomous: true
requirements:
  - D-01-on-break-flag
  - D-02-35-min-gap-threshold
  - D-03-frontend-greyout
must_haves:
  truths:
    - "GET /api/locations returns on_break: bool for every vehicle in the payload"
    - "A shuttle whose current UTC time falls inside a >=35min scheduled-departure gap (for any route today) gets on_break=true"
    - "When the schedule has no >=35min gap covering 'now' for any route, every vehicle in the response has on_break=false"
    - "The frontend shuttle marker renders muted (desaturated + 50% opacity) when on_break=true"
    - "The frontend shuttle marker renders at full color/opacity when on_break=false"
    - "SSE stream (/api/locations/stream) emits on_break on each pushed payload (uses the same _build_locations_payload)"
  artifacts:
    - path: "backend/fastapi/utils.py"
      provides: "SCHEDULE_GAP_THRESHOLD_SEC constant, _compute_gap_windows, _in_schedule_gap, _load_today_gap_windows (per-day cached)"
      contains: "SCHEDULE_GAP_THRESHOLD_SEC"
    - path: "backend/fastapi/routes.py"
      provides: "on_break: bool populated in _build_locations_payload for every vehicle entry"
      contains: "on_break"
    - path: "frontend/src/types/vehicleLocation.ts"
      provides: "on_break: boolean on VehicleLocationData (propagates into VehicleCombinedData)"
      contains: "on_break"
    - path: "frontend/src/locations/components/LiveLocationMapKit.tsx"
      provides: "muted SVG variant selection + onBreak propagated to annotation.url"
      contains: "on_break"
  key_links:
    - from: "backend/fastapi/routes.py:_build_locations_payload"
      to: "backend/fastapi/utils.py:_in_schedule_gap"
      via: "computes gap_windows once per payload build, then sets on_break per vehicle"
      pattern: "_in_schedule_gap"
    - from: "backend/fastapi/utils.py:_load_today_gap_windows"
      to: "shared/aggregated_schedule.json"
      via: "reads JSON, parses today's routes into datetimes, derives gap pairs"
      pattern: "aggregated_schedule"
    - from: "frontend/src/locations/components/LiveLocationMapKit.tsx"
      to: "on_break field on VehicleCombinedData"
      via: "switches SVG data-URL to muted variant inside annotation.url"
      pattern: "on_break"
---

<objective>
Phase 1 break detection: schedule-gap detector + on_break flag for frontend greyout.

Purpose: When a shuttle's cycle time falls in a long (>=35min) scheduled-departure gap (e.g. Saturday dinner, weekend lunch), expose an `on_break` boolean through `/api/locations` so the map can visually distinguish break shuttles without hiding them.

Output: Backend gap-window computation + `on_break` field in the REST+SSE locations payload, frontend type update, and muted marker rendering (desaturated + 50% opacity) when `on_break=true`.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/quick/260415-oeb-phase-1-break-detection-schedule-gap-det/260415-oeb-CONTEXT.md
@backend/fastapi/routes.py
@backend/fastapi/utils.py
@backend/worker/trips.py
@frontend/src/types/vehicleLocation.ts
@frontend/src/locations/components/LiveLocationMapKit.tsx
@frontend/src/locations/components/ShuttleIcon.tsx
@shared/aggregated_schedule.json

<interfaces>
<!-- Contracts extracted from the codebase so the executor doesn't need to go spelunking. -->

Existing schedule-loading pattern (mirror this shape, don't import from worker/trips):
From backend/worker/trips.py (lines 33, 68-112) — reference only:
```python
SCHEDULE_PATH = Path(__file__).parent.parent.parent / "shared" / "aggregated_schedule.json"
_SCHEDULE_CACHE: Dict[tuple, Dict[str, List[datetime]]] = {}

def _load_today_schedule(campus_tz) -> Dict[str, List[datetime]]:
    """Parse today's route schedule to UTC datetimes.
       aggregated_schedule is indexed by JS getDay() (0=Sun).
       js_day = (now.weekday() + 1) % 7."""
    # ... caches per (year, month, day, str(campus_tz))
```
Use the same day-key and JS-getDay indexing so the FastAPI gap cache stays consistent with the worker.

Timezone/time helpers:
From backend/config.py:
```python
settings.CAMPUS_TZ  # zoneinfo.ZoneInfo("America/New_York")
```
From backend/time_utils.py:
```python
def dev_now(tz=None) -> datetime  # timezone-aware "now" (respects DEV_TIME_SHIFT overrides for sim)
```

Existing response-building entry point (the one spot to add `on_break`):
From backend/fastapi/routes.py (lines 53-104):
```python
async def _build_locations_payload(session_factory) -> tuple[dict[str, Any], Optional[str]]:
    # iterates `results` and writes response_data[vehicle_id] = { ... }
    # `on_break: bool` must be added inside that dict literal
```
Used by BOTH GET /api/locations AND GET /api/locations/stream — one edit covers both transports.

Existing vehicle type (add `on_break` here):
From frontend/src/types/vehicleLocation.ts:
```ts
export type VehicleLocationData = {
    address_id: string;
    // ...existing fields...
    vin: string;
    driver?: { id: string; name: string; } | null;
}
// on_break flows through VehicleCombinedData via the spread in LiveLocationMapKit.tsx line 111
```

Existing SVG icon generator (needs a muted variant):
From frontend/src/locations/components/LiveLocationMapKit.tsx (lines 10-19):
```tsx
const _shuttleIconCache = new Map<string, string>();
function getShuttleIconUrl(color: string, size: number): string { ... }
```
Annotations consume this via `url: { 1: svgShuttle }` (line 245) on each AnimatedAnnotation. The URL update path in MapKitOverlays.tsx (lines 87-110) already re-applies `annotation.url = overlay.url` when the overlay for an existing key changes — so switching the cached URL between normal/muted will flow through without any other wiring.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="false">
  <name>Task 1: Backend — schedule-gap detector + on_break in /api/locations</name>
  <files>backend/fastapi/utils.py, backend/fastapi/routes.py</files>
  <behavior>
    Required observable behavior after this task:
    - GET /api/locations returns `on_break: bool` for every vehicle entry.
    - GET /api/locations/stream (SSE) emits the same field on every pushed payload (same helper).
    - With simulated Saturday 6:45 AM (e.g. DEV_TIME_SHIFT=1 DEV_TARGET_HOUR=6 DEV_TARGET_MINUTE=45), at least one vehicle in the response has on_break=true (any route whose Sat schedule has a >=35min gap bracketing 6:45 AM triggers the flag for vehicles currently inside the geofence).
    - During a weekday lunch rotation window (schedule continuous), on_break=false for every vehicle.
    - Response shape remains backward compatible: all existing fields still present; `on_break` is additive.
  </behavior>
  <action>
  Implement the gap detector in `backend/fastapi/utils.py` and wire it into `_build_locations_payload` in `backend/fastapi/routes.py`. Do NOT import from `backend.worker.trips` — FastAPI currently has zero deps on worker; keep that boundary clean by duplicating the small schedule-loading shape here.

  Per D-01: field name is `on_break: bool` on each vehicle entry in the `/api/locations` response. Do NOT add `break_reason` in Phase 1 (CONTEXT says "Phase 1 can ship with just the boolean"). A later phase can add it.

  Per D-02: threshold constant is `SCHEDULE_GAP_THRESHOLD_SEC = 35 * 60` (kept with other threshold constants in utils.py, near the top after imports).

  Step 1 — Add helpers to `backend/fastapi/utils.py` (at module scope, above `smart_closest_point`; keep them private with leading underscore):

  ```python
  import json
  from pathlib import Path
  from typing import Tuple
  from backend.config import settings

  # Path mirrors backend/worker/trips.py:SCHEDULE_PATH so both stay in sync.
  _AGGREGATED_SCHEDULE_PATH = Path(__file__).parent.parent.parent / "shared" / "aggregated_schedule.json"

  # Per D-02: 35-minute threshold. Conservative — above the normal
  # 10-min weekday cadence so only intentional lunch/dinner breaks fire.
  SCHEDULE_GAP_THRESHOLD_SEC = 35 * 60

  # Cached per (campus_date, campus_tz). Gap computation is trivial but
  # re-reading aggregated_schedule.json on every /api/locations call (once
  # per worker tick + SSE fan-out) is still pure waste. Day-rollover-safe
  # because the key includes the date.
  _GAP_WINDOWS_CACHE: dict[tuple, list[Tuple[datetime, datetime]]] = {}


  def _compute_gap_windows(sched_deps: List[datetime]) -> List[Tuple[datetime, datetime]]:
      """Return (gap_start, gap_end) pairs between consecutive scheduled
      departures whose gap exceeds SCHEDULE_GAP_THRESHOLD_SEC."""
      windows: List[Tuple[datetime, datetime]] = []
      for prev, nxt in zip(sched_deps, sched_deps[1:]):
          if (nxt - prev).total_seconds() >= SCHEDULE_GAP_THRESHOLD_SEC:
              windows.append((prev, nxt))
      return windows


  def _load_today_gap_windows(campus_tz) -> List[Tuple[datetime, datetime]]:
      """Return the union (across all routes) of today's >=35min scheduled-
      departure gap windows, in UTC. Cached per campus-day.

      We union across routes because the flag says 'this shuttle is in a
      break window' — not 'this shuttle's specific route is in a gap'. If
      ANY route scheduled to run today has a long gap that brackets 'now',
      a shuttle currently in the geofence is presumed on break (Phase 1
      scope per CONTEXT.md: Saturday lunch/dinner, Sat/Sun/Weekday dinner
      gaps — all cases where ALL routes share a gap).
      """
      now = dev_now(campus_tz)
      cache_key = (now.year, now.month, now.day, str(campus_tz))
      cached = _GAP_WINDOWS_CACHE.get(cache_key)
      if cached is not None:
          return cached

      try:
          with open(_AGGREGATED_SCHEDULE_PATH) as f:
              schedule = json.load(f)
      except Exception as e:
          logger.error(f"Failed to load aggregated_schedule for gap detection: {e}")
          _GAP_WINDOWS_CACHE.clear()
          _GAP_WINDOWS_CACHE[cache_key] = []
          return []

      # aggregated_schedule indexed by JS getDay() (0=Sun). Mirror trips.py:91.
      js_day = (now.weekday() + 1) % 7
      today = schedule[js_day] if js_day < len(schedule) else {}

      all_windows: List[Tuple[datetime, datetime]] = []
      for route_name, times in today.items():
          parsed: List[datetime] = []
          for time_str in times:
              try:
                  hm = datetime.strptime(time_str, "%I:%M %p")
                  dt = now.replace(hour=hm.hour, minute=hm.minute, second=0, microsecond=0)
                  if time_str.strip() == "12:00 AM":
                      dt += timedelta(days=1)
                  parsed.append(dt.astimezone(timezone.utc))
              except ValueError:
                  logger.warning(f"Could not parse schedule time for gap detect: {time_str}")
          parsed.sort()
          all_windows.extend(_compute_gap_windows(parsed))

      # Drop prior-day entries so the cache doesn't grow unbounded.
      _GAP_WINDOWS_CACHE.clear()
      _GAP_WINDOWS_CACHE[cache_key] = all_windows
      return all_windows


  def _in_schedule_gap(now_utc: datetime, gap_windows: List[Tuple[datetime, datetime]]) -> bool:
      """True if now_utc falls strictly inside any (gap_start, gap_end) window.

      Strict inequality (start < now < end) so the exact scheduled-departure
      boundary minute counts as in-service on both sides.
      """
      return any(start < now_utc < end for start, end in gap_windows)
  ```

  Import `datetime, timedelta, timezone` at the top of utils.py if not already present — `datetime` and `timezone` are imported but `timedelta` is not currently. Add it.

  Step 2 — Update `_build_locations_payload` in `backend/fastapi/routes.py`:

  Add import at the top of routes.py (next to the other backend.fastapi.utils imports):
  ```python
  from backend.fastapi.utils import (
      smart_closest_point,
      get_latest_vehicle_locations,
      get_current_driver_assignments,
      get_latest_velocities,
      _load_today_gap_windows,
      _in_schedule_gap,
  )
  ```

  Inside `_build_locations_payload`, compute the gap windows ONCE per payload build (not per vehicle), then set `on_break` in the per-vehicle dict:

  ```python
  async def _build_locations_payload(session_factory) -> tuple[dict[str, Any], Optional[str]]:
      results = await get_latest_vehicle_locations(session_factory)
      vehicle_ids = [loc["vehicle_id"] for loc in results]

      current_assignments = await get_current_driver_assignments(
          vehicle_ids, session_factory
      )

      # Schedule-gap detection (Phase 1 break detection). Computed once
      # per payload build and reused for every vehicle. Cached per-day
      # inside _load_today_gap_windows so this is effectively free after
      # the first call.
      gap_windows = _load_today_gap_windows(settings.CAMPUS_TZ)
      now_utc = dev_now(timezone.utc)
      in_gap_now = _in_schedule_gap(now_utc, gap_windows)

      response_data: dict[str, Any] = {}
      oldest_iso = min((loc["timestamp"] for loc in results), default=None)

      for loc in results:
          vehicle = loc["vehicle"]
          driver_info = None
          assignment = current_assignments.get(loc["vehicle_id"])
          if assignment and assignment.get("driver"):
              driver_data = assignment["driver"]
              driver_info = {
                  "id": driver_data["id"],
                  "name": driver_data["name"],
              }

          response_data[loc["vehicle_id"]] = {
              "name": loc["name"],
              "latitude": loc["latitude"],
              "longitude": loc["longitude"],
              "timestamp": loc["timestamp"],
              "heading_degrees": loc["heading_degrees"],
              "speed_mph": loc["speed_mph"],
              "is_ecu_speed": loc["is_ecu_speed"],
              "formatted_location": loc["formatted_location"],
              "address_id": loc["address_id"],
              "address_name": loc["address_name"],
              "license_plate": vehicle["license_plate"],
              "vin": vehicle["vin"],
              "asset_type": vehicle["asset_type"],
              "gateway_model": vehicle["gateway_model"],
              "gateway_serial": vehicle["gateway_serial"],
              "driver": driver_info,
              "on_break": in_gap_now,  # Phase 1 break detection (D-01, D-02)
          }

      return response_data, oldest_iso
  ```

  Note: Phase 1 says the flag is a property of "is now in a schedule gap" for ALL shuttles in the geofence. That's correct because the in-scope gaps (Sat lunch/dinner, Sat/Sun/Weekday dinner) are ALL-routes gaps — every route has the same pause. Weekday lunch rotation, where routes interleave and there IS no overall gap, is explicitly out of scope (Phase 2). The union-across-routes in `_load_today_gap_windows` is the correct semantic for Phase 1.

  Security / error handling: the schedule file already exists in the repo and is required for `/api/aggregated-schedule`; if the file is missing `_load_today_gap_windows` logs and returns `[]` (gap_windows empty → `_in_schedule_gap` always False → on_break always False, which is the safe/conservative fallback).

  Do NOT modify the `@cache` decorator on `get_locations` — `on_break` is deterministic per cache key (it's a function of server time, which changes the underlying `get_latest_vehicle_locations` result cache anyway on a 3s soft TTL). The existing cache stays correct.
  </action>
  <verify>
    <automated>cd "C:/Users/Jzgam/OneDrive/Documents/GitHub/shubble" && uv run python -c "from backend.fastapi.utils import _compute_gap_windows, _in_schedule_gap, SCHEDULE_GAP_THRESHOLD_SEC, _load_today_gap_windows; from datetime import datetime, timedelta, timezone; assert SCHEDULE_GAP_THRESHOLD_SEC == 35*60; t0 = datetime(2026,4,18,12,0,tzinfo=timezone.utc); t1 = t0 + timedelta(minutes=20); t2 = t0 + timedelta(minutes=40); t3 = t0 + timedelta(minutes=120); w = _compute_gap_windows([t0, t1, t2, t3]); assert w == [(t2, t3)], f'gap computation wrong: {w}'; assert _in_schedule_gap(t2 + timedelta(minutes=30), w) is True; assert _in_schedule_gap(t2, w) is False; assert _in_schedule_gap(t1, w) is False; print('gap detector OK')"</automated>
    <manual>Run `docker-compose --profile test --profile backend up` (or equivalent dev stack) with `DEV_TIME_SHIFT=1 DEV_TARGET_HOUR=6 DEV_TARGET_MINUTE=45` set so sim clock lands mid-Saturday-morning gap; then `curl -s http://localhost:8000/api/locations | jq '[.[] | {name, on_break}]'` — at least one entry should show `on_break: true`. Curl the same endpoint at a mid-afternoon weekday sim time — every entry shows `on_break: false`.</manual>
  </verify>
  <done>
    - `backend/fastapi/utils.py` exports `SCHEDULE_GAP_THRESHOLD_SEC=35*60`, `_compute_gap_windows`, `_in_schedule_gap`, `_load_today_gap_windows`; no imports from `backend.worker.*`.
    - The automated verify one-liner above prints `gap detector OK` (pure unit check — no DB, no Redis).
    - `_build_locations_payload` computes `gap_windows` ONCE per call and writes `on_break` into every vehicle dict.
    - `/api/locations` JSON includes `on_break: bool` on every vehicle; existing fields unchanged.
    - SSE `/api/locations/stream` emits the same field (automatic — no additional code needed since it reuses `_build_locations_payload`).
  </done>
</task>

<task type="auto" tdd="false">
  <name>Task 2: Frontend — on_break type + muted shuttle marker</name>
  <files>frontend/src/types/vehicleLocation.ts, frontend/src/locations/components/LiveLocationMapKit.tsx</files>
  <behavior>
    Required observable behavior after this task:
    - When `/api/locations` returns `on_break: true` for a vehicle, that shuttle's marker on the live map renders with desaturation (grayscale effect) AND 50% opacity, per D-03.
    - When `on_break: false` (or undefined, for safety), the marker renders at full saturation and full opacity — identical to current appearance.
    - Marker position / heading / animation are unchanged — only visual style is muted.
    - TypeScript compilation passes with no errors; `VehicleLocationData` and `VehicleCombinedData` both carry the new field, and the existing spread pipeline in `LiveLocationMapKit.tsx` line 111 propagates it automatically.
  </behavior>
  <action>
  Step 1 — Update `frontend/src/types/vehicleLocation.ts`:

  Add `on_break?: boolean` to `VehicleLocationData` (use optional to stay backward-safe if a stale backend briefly omits it — the frontend will default-treat missing as false):

  ```ts
  // Raw vehicle location data from /api/locations
  export type VehicleLocationData = {
      address_id: string;
      address_name: string;
      asset_type: string;
      formatted_location: string;
      gateway_model: string;
      gateway_serial: string;
      heading_degrees: number;
      is_ecu_speed: boolean;
      latitude: number;
      license_plate: string;
      longitude: number;
      name: string;
      speed_mph: number;
      timestamp: string;
      vin: string;
      driver?: { id: string; name: string; } | null;
      on_break?: boolean;  // Phase 1 break detection (D-01)
  }
  ```

  No change needed to `VehicleCombinedData` — it extends `VehicleLocationData` via intersection (`VehicleLocationData & {...}`) and the existing spread in `LiveLocationMapKit.tsx` line 111 (`combined[vehicleId] = { ...location, ... }`) already carries `on_break` through.

  Step 2 — In `frontend/src/locations/components/LiveLocationMapKit.tsx`, produce a muted SVG data-URL variant and select between normal/muted per vehicle.

  Per D-03, the visual effect is "opacity 0.5 + grey tint (filter: grayscale(0.8) opacity(0.5) equivalent)". MapKit's ImageAnnotation renders the `url` as an image; CSS filter on the outer image won't reliably apply. The cleanest way that honors the decision semantics: bake the grayscale + opacity into a second SVG variant. Inline SVG `<filter>` + wrapper `opacity` attribute produces exactly the same visual effect as the CSS filter described in D-03.

  Modify the icon cache + getter, then use on-break state to pick the URL:

  ```tsx
  // PERF: SVG shuttle icons only depend on (color, size, muted), so cache
  // by all three. Muted variant embeds an SVG <feColorMatrix> filter to
  // match the D-03 spec (grayscale(0.8) opacity(0.5)) inside the data-URL
  // itself — MapKit ImageAnnotation doesn't expose a reliable CSS filter
  // hook on the rendered <img> element, so baking it into the SVG is the
  // straightforward way to honor the decision.
  const _shuttleIconCache = new Map<string, string>();
  function getShuttleIconUrl(color: string, size: number, muted: boolean): string {
    const key = `${color}|${size}|${muted ? "m" : "n"}`;
    const cached = _shuttleIconCache.get(key);
    if (cached) return cached;
    const inner = renderToStaticMarkup(<ShuttleIcon color={color} size={size} />);
    let svg: string;
    if (muted) {
      // Wrap the existing shuttle SVG in a filter + lowered opacity.
      // feColorMatrix type="saturate" 0.2 ≈ grayscale(0.8); group opacity
      // 0.5 matches the D-03 spec. Using a group wrapper (<g>) avoids
      // re-parsing the inner SVG — we just wrap it in a filtered group
      // and a <filter> definition.
      svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 50 50">
        <defs><filter id="m"><feColorMatrix type="saturate" values="0.2"/></filter></defs>
        <g filter="url(#m)" opacity="0.5">${inner.replace(/^<svg[^>]*>|<\/svg>$/g, "")}</g>
      </svg>`;
    } else {
      svg = inner;
    }
    const url = `data:image/svg+xml;base64,${btoa(svg)}`;
    _shuttleIconCache.set(key, url);
    return url;
  }
  ```

  Update the per-vehicle annotation build inside `vehicleAnnotationProps` to pass the muted flag:

  ```tsx
  // Cached SVG data URL — renders once per unique (color, size, muted).
  const muted = vehicle.on_break === true;
  const svgShuttle = getShuttleIconUrl(routeColor, shuttleIconSize, muted);
  ```

  No change needed to `MapKitOverlays.tsx` — it already re-applies `annotation.url = overlay.url` on each update for existing keys (lines 87-110), so a shuttle that transitions from on_break=false → on_break=true between polls will swap its icon on the next tick.

  Edge case handling:
  - If `on_break` is `undefined` (backend response predates this field), the `=== true` check treats it as false → full-color rendering. No runtime error.
  - If a shuttle is selected / has an open callout when its style changes, MapKit handles the URL change in place via the existing overlays update path.
  </action>
  <verify>
    <automated>cd "C:/Users/Jzgam/OneDrive/Documents/GitHub/shubble/frontend" && npx tsc --noEmit</automated>
    <manual>Run the frontend against a backend with at least one `on_break: true` shuttle (use the Sat-morning DEV_TIME_SHIFT scenario from Task 1). Open the live map: the flagged shuttle's marker is visibly desaturated and semi-transparent; non-flagged shuttles look identical to before. Flip the sim clock to a weekday non-gap time and confirm every marker returns to full color/opacity.</manual>
  </verify>
  <done>
    - `VehicleLocationData` has `on_break?: boolean`; `VehicleCombinedData` carries the field via intersection; TypeScript compiles clean (`npx tsc --noEmit` exits 0).
    - `getShuttleIconUrl(color, size, muted)` returns a correctly cached, filtered SVG data-URL when `muted=true`.
    - `vehicleAnnotationProps` selects the muted variant iff `vehicle.on_break === true`.
    - No changes to marker position, heading, speed, animation, selected-route logic, or callout behavior.
  </done>
</task>

</tasks>

<verification>
End-to-end phase check (after both tasks):
1. Backend unit: `uv run python -c "..."` one-liner from Task 1 prints `gap detector OK`.
2. Backend integration: `curl /api/locations | jq 'first(.[]) | has("on_break")'` returns `true` (field always present).
3. Frontend type: `cd frontend && npx tsc --noEmit` exits 0.
4. End-to-end manual: with sim clock set to a Saturday schedule-gap minute, the flagged shuttle on the live map is visibly muted; flip to a continuous-schedule minute and it returns to full color.
5. SSE parity: `curl -N http://localhost:8000/api/locations/stream` shows `on_break` on every message (uses the same payload helper).
</verification>

<success_criteria>
- GET /api/locations and the SSE stream both emit `on_break: bool` on every vehicle entry.
- The flag is `true` iff "now" (sim clock) falls inside a >=35-minute scheduled-departure gap for today (union across routes).
- The flag is `false` during normal in-service windows (weekday lunch rotation stays false — Phase 2 scope).
- Frontend shuttle markers render muted (grayscale + 50% opacity) when `on_break=true`, full-color otherwise.
- Zero changes to existing fields, response shape compatibility preserved, no new runtime dependencies (no imports from worker/trips in FastAPI).
- All existing tests (if any) still pass; TypeScript compiles with no new errors or warnings.
</success_criteria>

<output>
After completion, create `.planning/quick/260415-oeb-phase-1-break-detection-schedule-gap-det/260415-oeb-SUMMARY.md` documenting:
- The final field contract (`on_break: bool`, no `break_reason` in Phase 1).
- The 35-min constant location (`backend/fastapi/utils.py:SCHEDULE_GAP_THRESHOLD_SEC`).
- How the muted SVG variant is constructed (for whoever ships Phase 2's driver-assignment signal).
- Follow-up notes: Phase 2 will add `break_reason` and weekday lunch rotation handling; the current union-across-routes semantic holds for Phase 1's target scenarios.
</output>
</content>
</invoke>