---
phase: quick-260415-drf
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - backend/worker/trips.py
  - tests/test_last_arrival_loop_scoping.py
  - frontend/src/schedule/Schedule.tsx
  - frontend/src/schedule/styles/Schedule.css
autonomous: true
requirements:
  - DRF-01
  - DRF-02

must_haves:
  truths:
    - "For an active trip, no stop's `last_arrival` is older than that trip's `actual_departure`."
    - "When a shuttle is idle at Union (passes the idle-filter guards), the next `status=scheduled` trip for that shuttle's historical route has `vehicle_id` populated instead of null."
    - "Only one idle shuttle can claim the NEXT future scheduled slot per route (no double-assignment); further idle shuttles on the same route remain unassigned."
    - "On the schedule page, scheduled rows whose trip has a `vehicle_id` render a small 'waiting' pill next to the vehicle badge; rows with `vehicle_id: null` render unchanged."
  artifacts:
    - path: "backend/worker/trips.py"
      provides: "Defensive per-entry guard + idle-to-next-scheduled vehicle assignment"
      contains: "actual_departure"
    - path: "tests/test_last_arrival_loop_scoping.py"
      provides: "Regression lock for Colonie stale-la bug via surgical guard"
      contains: "def test_"
    - path: "frontend/src/schedule/Schedule.tsx"
      provides: "Waiting pill rendered on scheduled rows with assigned vehicle_id"
      contains: "waiting-pill"
    - path: "frontend/src/schedule/styles/Schedule.css"
      provides: "Waiting pill styling"
      contains: ".waiting-pill"
  key_links:
    - from: "backend/worker/trips.py build_trip_etas"
      to: "per-stop entry assignment (lines ~486-496, ~592-606)"
      via: "defensive guard that drops/scrubs la < actual_departure at entry-write time"
      pattern: "last_arrival.*actual_departure|actual_departure.*last_arrival"
    - from: "backend/worker/trips.py compute_trips_from_vehicle_data (scheduled-trip emission ~1076-1119)"
      to: "idle-vehicle pool captured upstream"
      via: "post-loop pass that assigns vid to the first unclaimed future scheduled dep per route"
      pattern: "idle_at_union|idle_vehicles|next_scheduled"
    - from: "frontend/src/schedule/Schedule.tsx (~line 989-991)"
      to: "Trip.vehicle_id on status=scheduled rows"
      via: "conditional 'waiting' pill render"
      pattern: "waiting-pill"
---

<objective>
Fix two real-service issues observed against test-server :4000 / backend :8000 / frontend :3000:

1. **BUG — Colonie shows stale "Passed"**: Active NORTH trip shows `COLONIE.last_arrival` with a timestamp BEFORE the trip's own `actual_departure`, which flips the stop to `passed=true` on the UI even though the shuttle just left Union. The existing `loop_cutoff` filter at the top of `build_trip_etas` is meant to catch this, but evidence from live state shows it doesn't always hold — either the filter is being bypassed or a downstream path (monotonic-clamp anchor, backfill interpolation) is resurrecting the stale la on the entry. Add a defensive guard at every point where `entry["last_arrival"]` is written, dropping any la strictly older than the trip's `actual_departure`.

2. **FEATURE — Idle shuttle not bound to next scheduled slot**: When a shuttle is idle at Union (passes the existing idle-filter short-circuit in `compute_trips_from_vehicle_data`), its trip row is suppressed but the NEXT `status=scheduled` trip for that route still has `vehicle_id: null`. Students can't see that a physical shuttle is parked and waiting for a specific upcoming departure. Capture the idle-at-Union vehicles during the filter check and, after the main assignment loop, bind each to the next unclaimed future scheduled slot for its historical route. Render a compact "waiting" pill on the schedule-page row when `status=scheduled` and `vehicle_id` is populated.

Purpose: Colonie fix restores the correctness invariant "no stop inside the current loop shows a Last: timestamp older than the shuttle's own departure". Idle-shuttle feature closes the current UX hole where a visibly parked shuttle has no schedule-row presence at all.

Output:
- `backend/worker/trips.py` — defensive per-entry guard + idle-vehicle capture + post-loop scheduled-slot assignment
- `tests/test_last_arrival_loop_scoping.py` — regression test that passes a stale la in `last_arrivals` but verifies it never appears on the built entry
- `frontend/src/schedule/Schedule.tsx` — conditional "waiting" pill on scheduled-with-vehicle rows
- `frontend/src/schedule/styles/Schedule.css` — pill style
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/quick/260415-drf-idle-shuttle-scheduled-departure-context/260415-drf-CONTEXT.md
@backend/worker/trips.py
@tests/test_last_arrival_loop_scoping.py
@frontend/src/schedule/Schedule.tsx
@frontend/src/hooks/useTrips.ts

<interfaces>
<!-- Key contracts extracted from codebase so the executor needs no extra exploration. -->

### Trip schema (from `frontend/src/hooks/useTrips.ts:20-29`)
```typescript
export interface Trip {
  trip_id: string;
  route: string;
  departure_time: string;       // ISO datetime (scheduled slot time)
  actual_departure: string | null;
  scheduled: boolean;
  vehicle_id: string | null;    // ← Task 2 populates this on status='scheduled' for idle shuttles
  status: 'scheduled' | 'active' | 'unassigned' | 'completed';
  stop_etas: Record<string, TripStopETA>;
}
```
No type change is needed — `vehicle_id` is already `string | null` on the Trip type.

### `build_trip_etas` signature (from `backend/worker/trips.py:176-183`)
```python
def build_trip_etas(
    trip: Dict[str, Any],
    vehicle_stops: List,          # List[Tuple[stop_key, eta_datetime]]
    last_arrivals: Dict[str, str],
    stops_in_route: List[str],
    now_utc: datetime,
    loop_cutoff: Optional[datetime] = None,
) -> Dict[str, Dict[str, Any]]:
```
The `trip` dict already contains `actual_departure` (ISO string) — Task 1 reads it from there (or parses from `loop_cutoff`) to build the defensive guard.

### Where entries are written in `build_trip_etas`
- **Primary write path** at lines 486–496 (`if stop_key in last_arrivals: entry["last_arrival"] = la_iso`).
- **Monotonic-clamp write path** at lines 592–606 (`entry["last_arrival"] = clamped` for real detections) and line 619 (interpolated la for gap stops).

### Idle-filter short-circuit (from `backend/worker/trips.py:743-763`)
```python
# Filter 1: idle at the route's first stop for longer than a full loop
if at_first_stop and not_moving and idle_seconds > IDLE_THRESHOLD_SEC:
    logger.debug(f"Skipping trip for vehicle {vid} on {route}: idle at {first_stop} for {idle_seconds:.0f}s")
    continue
```
Task 2 hooks IN AT THIS POINT — before the `continue`, record the (route, vid, now_utc) into an `idle_at_union_by_route` dict, then continue as before.

### Scheduled-trip emission block (from `backend/worker/trips.py:1076-1122`)
```python
# Add scheduled trips that don't have a vehicle assigned yet
for route, sched_deps in schedule.items():
    ...
    for dep in sched_deps:
        if (route, dep.isoformat()) in assigned_scheduled_times:
            continue
        ...
        trips.append({
            ...
            "vehicle_id": None,      # ← Task 2 replaces this with pop from idle pool
            "status": "scheduled" if dep > now_utc else "unassigned",
            ...
        })
```
Task 2 adds a pre-pass that assigns one idle vid per route to the FIRST unclaimed future scheduled slot for that route, marking that (route, iso) as claimed BEFORE this emission loop walks it.

### Schedule row render anchor (from `frontend/src/schedule/Schedule.tsx:989-991`)
```tsx
{loopTrip?.vehicle_id && (
  <span className="vehicle-badge" aria-label={`Shuttle ${loopTrip.vehicle_id.slice(-3)}`}>#{loopTrip.vehicle_id.slice(-3)}</span>
)}
```
Task 3 adds a sibling `<span className="waiting-pill">waiting</span>` gated on `loopTrip?.status === 'scheduled' && loopTrip?.vehicle_id`. The existing `vehicle-badge` already renders whenever `vehicle_id` is set — no change to that.

### Existing test conventions (from `tests/test_last_arrival_loop_scoping.py:1-40`)
- Uses `pytest` + `build_trip_etas` + `compute_trips_from_vehicle_data` directly.
- `_iso()` helper formats tz-aware datetimes exactly as the worker does.
- Tests pass `loop_cutoff=actual_departure` and assert on the returned `entry["last_arrival"]`.

### Live services assumed running
- `http://localhost:4000` — mock Samsara test server
- `http://localhost:8000` — backend (FastAPI)
- `http://localhost:3000` — frontend (Vite)

Use `curl http://localhost:8000/api/trips | jq ...` for black-box verification.
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Drop last_arrival older than trip's actual_departure at every entry-write point</name>
  <files>backend/worker/trips.py, tests/test_last_arrival_loop_scoping.py</files>
  <behavior>
    Regression test added to `tests/test_last_arrival_loop_scoping.py` named `test_stale_la_dropped_by_defensive_guard` that:
    1. Constructs `trip = {"trip_id": "NORTH:...", "actual_departure": "<41+00 ISO>", ...}`.
    2. Builds `last_arrivals = {"STUDENT_UNION": "<33+00 ISO>", "COLONIE": "<33+00 ISO>", "GEORGIAN": "<post-actual-departure ISO>"}` — i.e., exactly the bug snapshot from CONTEXT where two stops have la BEFORE actual_departure and one has la AFTER.
    3. Calls `build_trip_etas(trip=trip, vehicle_stops=[...], last_arrivals=last_arrivals, stops_in_route=NORTH_STOPS, now_utc=..., loop_cutoff=actual_departure)`.
    4. Asserts:
       - `stop_etas["STUDENT_UNION"]["last_arrival"] is None`
       - `stop_etas["STUDENT_UNION"]["passed"] is False`
       - `stop_etas["COLONIE"]["last_arrival"] is None`
       - `stop_etas["COLONIE"]["passed"] is False`
       - `stop_etas["GEORGIAN"]["last_arrival"]` equals the post-actual-departure ISO (the one valid la IS preserved)
       - Any interpolated `last_arrival` written by the backfill (lines 540-620) is ALSO strictly `>= actual_departure`

    Then a second test `test_stale_la_never_resurrected_by_monotonic_clamp` that:
    - Passes `last_arrivals = {"STUDENT_UNION": "<pre-actual-departure ISO>", "GEORGIAN": "<post-actual-departure ISO>"}` (later stop valid, earlier stop stale).
    - Asserts `stop_etas["STUDENT_UNION"]["last_arrival"] is None` AND `stop_etas["STUDENT_UNION"]["passed_interpolated"]` is either False (if left unpassed) or any interpolated la is `>= actual_departure`.
    - Guards against the backfill-anchor path from re-writing the stale la.

    Tests fail BEFORE implementation (RED), pass AFTER (GREEN).
  </behavior>
  <action>
    **RED first**: Add both tests to `tests/test_last_arrival_loop_scoping.py` following existing patterns (see the top-of-file helpers `_iso()`, `_make_vehicle_df_loop_detections`, the `NORTH_STOPS` import). Run the tests and confirm they FAIL against the current implementation. (If the first test coincidentally passes because the existing `loop_cutoff` filter at lines 215-226 catches the pre-filter case, wire the test to bypass that filter by also passing `loop_cutoff=None` in a third sibling test and confirming the new guard still catches stale la — that is the real regression lock.)

    **GREEN**: In `backend/worker/trips.py`, add a defensive `actual_departure_dt` derivation at the top of `build_trip_etas` (parse from `trip.get("actual_departure")` if non-null; else fall back to `loop_cutoff`). Then at every point where `entry["last_arrival"]` is set to a non-None value, add a guard that drops/nulls the la if it is strictly earlier than `actual_departure_dt`:

    - **Write site 1 (lines 486-496)**: before `entry["last_arrival"] = la_iso`, parse `la_iso` and skip the write + DO NOT set `passed=True` if `la_dt < actual_departure_dt`. When skipping, also skip `entry["eta"] = None` so a future ETA can still surface for the unreached stop. (This is belt-and-suspenders on top of the existing 215-226 filter — the filter should have already dropped this, but if any path gets past it, this is the last line of defense.)

    - **Write site 2 (lines 592-606, real-detection branch)**: `clamped` is read from `anchors`. If `clamped < actual_departure_dt` (parsed), do NOT write la/passed; instead null them and let the stop fall through as "not yet reached" in the current loop.

    - **Write site 3 (line 619, interpolated branch)**: if the interpolated la is earlier than `actual_departure_dt`, null it (`entry["last_arrival"] = None`) and set `entry["passed"] = False`, `entry["passed_interpolated"] = False` — the backfill shouldn't fabricate a "passed" timestamp that predates the trip's own departure.

    **Important**: When `actual_departure_dt` is None (no trip.actual_departure, no loop_cutoff — e.g. pure scheduled trips with no vehicle data) the guard is a no-op. Preserves existing behavior for unassigned scheduled trips.

    Add a module-level comment block at the top of the new guard explaining WHY (defense in depth: the 215-226 filter is the first line, but monotonic-clamp anchors and interpolation can reintroduce values that originated from another path, and we want the invariant "no entry's la predates the trip's actual_departure" to be enforceable post-hoc).

    **REFACTOR (only if needed)**: extract the la-parse-and-compare into a small `_la_is_before(la_iso: str, cutoff: datetime) -> bool` helper inside `build_trip_etas` scope to avoid four copy-pasted try/except blocks.
  </action>
  <verify>
    <automated>cd "C:/Users/Jzgam/OneDrive/Documents/GitHub/shubble" && uv run pytest tests/test_last_arrival_loop_scoping.py -x -v</automated>
    Also curl verify against live state after implementation:
    `curl -s http://localhost:8000/api/trips | jq '[.[] | select(.status=="active") | .stop_etas | to_entries[] | select(.value.last_arrival != null and .value.last_arrival < (.. | .actual_departure? // empty))] | length'` should return 0 (no active trip has any stop whose la predates its own actual_departure).
  </verify>
  <done>
    - Both new tests pass.
    - All pre-existing tests in `test_last_arrival_loop_scoping.py` still pass (no regression).
    - Live `/api/trips` shows zero stops with `last_arrival < actual_departure` on any active trip.
    - No changes to the legacy behavior for unassigned scheduled trips (`actual_departure: null`).
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Bind idle-at-Union shuttles to next scheduled trip's vehicle_id</name>
  <files>backend/worker/trips.py, tests/test_last_arrival_loop_scoping.py</files>
  <behavior>
    Regression test added to `tests/test_last_arrival_loop_scoping.py` named `test_idle_shuttle_assigned_to_next_scheduled_trip` that:
    1. Synthesizes a vehicle dataframe where vid='000000000000099' is parked at STUDENT_UNION on NORTH with `speed_kmh=0` and `idle_seconds > IDLE_THRESHOLD_SEC` (e.g. 1800s of first-stop dwell).
    2. Uses a schedule for NORTH with slots at `now-5min`, `now+10min`, `now+25min`.
    3. Calls `compute_trips_from_vehicle_data(...)`.
    4. Asserts:
       - The idle vehicle has no `status='active'` trip (idle-filter still fires — regression guard).
       - Exactly ONE scheduled trip for NORTH in the future (`status='scheduled'`, `departure_time==now+10min`) has `vehicle_id == '000000000000099'`.
       - The NEXT scheduled trip (now+25min) still has `vehicle_id: null`.
       - Past scheduled slot (now-5min) is unaffected.

    A second test `test_multiple_idle_shuttles_same_route_claim_sequentially` where two vehicles are idle on NORTH: only ONE of them gets assigned to the NEXT future slot; the other remains unassigned (stays with `vehicle_id: null` on the row after next, OR is not assigned at all — the simpler contract is "at most one idle vid per route claims the NEXT slot"). Assert exactly one scheduled NORTH trip has `vehicle_id != null` among future slots within the 2-hour window.
  </behavior>
  <action>
    **RED first**: Write both tests and confirm they FAIL.

    **GREEN**: In `backend/worker/trips.py`, inside `compute_trips_from_vehicle_data`:

    1. At the top of the function, alongside `assigned_scheduled_times: set = set()` (around line 662), add:
       ```python
       # route -> vid that is idle at first_stop and should claim the next
       # scheduled slot for this route. Populated during Pass 1's idle filter,
       # consumed after the main assignment loop before scheduled-trip emission.
       idle_at_union_by_route: Dict[str, str] = {}
       ```

    2. Inside the idle filter branch (around line 758-763), BEFORE the `continue`, record the vehicle:
       ```python
       if at_first_stop and not_moving and idle_seconds > IDLE_THRESHOLD_SEC:
           logger.debug(
               f"Skipping trip for vehicle {vid} on {route}: "
               f"idle at {first_stop} for {idle_seconds:.0f}s"
           )
           # But remember it so the next scheduled slot on this route gets
           # this vid assigned — students should see "waiting → leaves HH:MM"
           # on the upcoming row. First idle vid per route wins (O(1) by
           # Python dict insertion order; deterministic because vehicle_stop_etas
           # iteration order is stable within a worker cycle).
           idle_at_union_by_route.setdefault(route, str(vid))
           continue
       ```
       `setdefault` ensures only the FIRST idle vid per route claims the slot — additional idle vids on the same route are silently ignored (no double-assignment, no error). This implements the "at most one idle vid per route" invariant required by the second test.

    3. Directly BEFORE the scheduled-trip emission loop at line 1076 (`# Add scheduled trips that don't have a vehicle assigned yet`), add a pre-pass that resolves each route's idle vid to the first unclaimed future scheduled slot:
       ```python
       # Bind idle-at-Union shuttles to their next scheduled slot. For each
       # route that had an idle shuttle recorded in Pass 1, find the earliest
       # UNCLAIMED future scheduled departure and remember (route, iso_dep) ->
       # vid. The scheduled-trip emission loop below consumes this mapping
       # when constructing the trip dict.
       idle_slot_assignments: Dict[tuple, str] = {}  # (route, iso_dep) -> vid
       for route, idle_vid in idle_at_union_by_route.items():
           sched_deps = schedule.get(route, [])
           for dep in sorted(sched_deps):
               if dep <= now_utc:
                   continue
               slot_key = (route, dep.isoformat())
               if slot_key in assigned_scheduled_times:
                   continue
               idle_slot_assignments[slot_key] = idle_vid
               assigned_scheduled_times.add(slot_key)  # reserve so no double-emit
               break  # only the FIRST future slot per route claims this vid
       ```
       The `assigned_scheduled_times.add(slot_key)` call here is important ONLY as a bookkeeping reservation — but the scheduled-trip emission at line 1086 has `if (route, dep.isoformat()) in assigned_scheduled_times: continue`, which would SKIP this slot entirely. So DO NOT add to `assigned_scheduled_times` — instead, pass the `idle_slot_assignments` dict into the emission loop and read-only reference it there. Revised snippet:
       ```python
       idle_slot_assignments: Dict[tuple, str] = {}
       for route, idle_vid in idle_at_union_by_route.items():
           sched_deps = schedule.get(route, [])
           for dep in sorted(sched_deps):
               if dep <= now_utc:
                   continue
               slot_key = (route, dep.isoformat())
               if slot_key in assigned_scheduled_times:
                   # Slot already claimed by a real active/dwelling shuttle;
                   # idle vid steps aside and does not claim anything.
                   continue
               idle_slot_assignments[slot_key] = idle_vid
               break
       ```

    4. In the scheduled-trip emission block (lines 1110-1119), replace the hardcoded `"vehicle_id": None` with a lookup:
       ```python
       vid_for_slot = idle_slot_assignments.get((route, dep.isoformat()))
       trips.append({
           "trip_id": f"{route}:{dep.isoformat()}" + (f":{vid_for_slot}" if vid_for_slot else ""),
           "route": route,
           "departure_time": dep.isoformat(),
           "actual_departure": None,
           "scheduled": True,
           "vehicle_id": vid_for_slot,   # string if idle vid claimed this slot, else None
           "status": "scheduled" if dep > now_utc else "unassigned",
           "stop_etas": stop_etas,
       })
       ```
       Note: incorporate the vid into `trip_id` ONLY when a vid was assigned — this keeps trip_ids stable for unassigned rows (frontend dedup keys in `Schedule.tsx:470` use `trip_id` as fallback).

    **Why setdefault over other approaches**:
    - We don't want a shuttle that's been parked all day to claim a slot 6 hours from now — the context says "next scheduled departure", interpreted as the NEXT FUTURE slot.
    - We don't want two idle shuttles on the same route both claiming slots — only the first idle vid per route claims its route's next slot. Remaining idle vehicles stay invisible (consistent with current behavior; they're not at a scheduled departure time anyway).

    **Union-proximity threshold**: reuse existing idle-at-first-stop detection (`at_first_stop and not_moving and idle_seconds > IDLE_THRESHOLD_SEC`) — no new constant needed. Claude's discretion per CONTEXT was "reuse existing 60m constant OR CLOSE_APPROACH_M" — using the already-working idle-filter's check is both cheaper and already-validated.
  </action>
  <verify>
    <automated>cd "C:/Users/Jzgam/OneDrive/Documents/GitHub/shubble" && uv run pytest tests/test_last_arrival_loop_scoping.py::test_idle_shuttle_assigned_to_next_scheduled_trip tests/test_last_arrival_loop_scoping.py::test_multiple_idle_shuttles_same_route_claim_sequentially -x -v</automated>
    Also full regression: `uv run pytest tests/ -x` — no pre-existing test should break.
    Live curl check: `curl -s http://localhost:8000/api/trips | jq '[.[] | select(.status=="scheduled" and .vehicle_id != null)] | length'` should return 1+ when a test shuttle is parked at Union (test server :4000 can simulate this).
  </verify>
  <done>
    - Both new tests pass.
    - All pre-existing tests pass (no regression in idle-filter behavior or scheduled-trip emission).
    - Live `/api/trips` shows at least one `status=scheduled` trip with a non-null `vehicle_id` when a test shuttle is parked at Union.
    - Only ONE scheduled trip per route per idle shuttle is bound (no double-assignment).
    - Scheduled trips with no idle shuttle still have `vehicle_id: null` (no regression to existing unassigned trips).
  </done>
</task>

<task type="auto">
  <name>Task 3: Render "waiting" pill on schedule rows with scheduled-status + assigned vehicle_id</name>
  <files>frontend/src/schedule/Schedule.tsx, frontend/src/schedule/styles/Schedule.css</files>
  <action>
    No changes to `frontend/src/hooks/useTrips.ts` — the `Trip` interface at `useTrips.ts:20-29` already has `vehicle_id: string | null` and `status` enum includes `'scheduled'`. Backend change in Task 2 simply populates `vehicle_id` on status=scheduled rows, and the frontend receives it transparently.

    **Schedule.tsx change**: Locate the existing vehicle-badge render (around line 989-991):
    ```tsx
    {loopTrip?.vehicle_id && (
      <span className="vehicle-badge" aria-label={`Shuttle ${loopTrip.vehicle_id.slice(-3)}`}>#{loopTrip.vehicle_id.slice(-3)}</span>
    )}
    ```
    Immediately AFTER this block (but before the `{showRelative ...}` block), add:
    ```tsx
    {loopTrip?.status === 'scheduled' && loopTrip?.vehicle_id && (
      <span className="waiting-pill" aria-label={`Shuttle waiting for this departure`}>waiting</span>
    )}
    ```
    This:
    - Renders nothing for active/completed trips (existing vehicle-badge still shows there).
    - Renders nothing for scheduled trips with no vehicle (existing behavior preserved — pure schedule rows are unchanged).
    - Renders ONLY for the new case introduced by Task 2: scheduled + vehicle assigned via idle binding.
    - The pill sits next to the vehicle badge so students read "#123 waiting" inline on the row.

    **Schedule.css change**: Append a new rule after the existing `.vehicle-badge` block (after line 363):
    ```css
    /* Waiting pill — shown on scheduled rows when an idle shuttle is
       parked at Union and bound to this upcoming departure. Visually
       distinguishes "schedule says 2:30 AND a specific shuttle is
       waiting for it" from "schedule says 2:30, no shuttle assigned yet". */
    .waiting-pill {
      display: inline-block;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #b08c00;
      background: rgba(176, 140, 0, 0.1);
      border: 1px solid rgba(176, 140, 0, 0.3);
      padding: 1px 5px;
      border-radius: 3px;
      vertical-align: middle;
      margin-left: 6px;
    }
    ```
    The amber color is distinct from the existing source-live blue (#578FCA) and trip-completed-badge gray (#7f8c8d). Rationale: "waiting" is a distinct semantic state deserving its own visual slot.

    **Copy choice**: "waiting" (not "idle", not "#001 waiting — leaves 2:30"). The row already shows the scheduled time in the `.timeline-time` element and the vehicle id in the adjacent `.vehicle-badge`, so adding either repeats info. A single-word pill is the minimal addition that communicates the new state.

    Do NOT touch `Schedule.tsx:418-500` (row construction), `Schedule.tsx:639-736` (next-ETA computation), or any other rendering path — the existing code already handles a Trip object with `status='scheduled'` and a non-null `vehicle_id` correctly (see comments at lines 989, 1008-1022 which document the scheduled+vehicle case). This change is purely additive.
  </action>
  <verify>
    <automated>cd "C:/Users/Jzgam/OneDrive/Documents/GitHub/shubble/frontend" && npm run build</automated>
    (Vite+TypeScript build must pass — catches any type error on the new JSX.)

    Manual smoke (optional, for the human):
    - Visit `http://localhost:3000/schedule`, pick the NORTH route, look for the next upcoming scheduled slot. When a test-server shuttle is parked at Union long enough to trip the idle filter, the row should show `#XYZ waiting` next to the time.
    - When no shuttle is parked, the same row shows no pill — no regression on the empty scheduled-slot case.
  </verify>
  <done>
    - `npm run build` passes with no new TypeScript errors.
    - Waiting pill renders ONLY on rows where `status==='scheduled' && vehicle_id != null`.
    - Pure scheduled rows (no vehicle) render unchanged.
    - Active/completed rows render unchanged (vehicle-badge still shows, no waiting pill).
  </done>
</task>

</tasks>

<verification>
End-to-end live verification (all three services running):

1. **Bug fix (Task 1)** — with an actively running NORTH shuttle mid-loop:
   ```bash
   curl -s http://localhost:8000/api/trips | \
     jq '[.[] | select(.status=="active") |
          . as $t |
          .stop_etas | to_entries[] |
          select(.value.last_arrival != null and
                 ($t.actual_departure != null) and
                 (.value.last_arrival < $t.actual_departure))] | length'
   ```
   Must return `0`. (Before the fix, the Colonie case in CONTEXT would have made this return >= 1.)

2. **Idle-shuttle feature (Task 2)** — with a test-server shuttle parked at Union > 20 min:
   ```bash
   curl -s http://localhost:8000/api/trips | \
     jq '[.[] | select(.status=="scheduled" and .vehicle_id != null) | {route, departure_time, vehicle_id}]'
   ```
   Must return at least one entry. Each entry's `departure_time` must be the NEXT future scheduled slot for that `route`, not a later one.

3. **Invariant (Task 2 correctness)**:
   ```bash
   curl -s http://localhost:8000/api/trips | \
     jq 'group_by(.vehicle_id) | map(select(.[0].vehicle_id != null) | {vid: .[0].vehicle_id, scheduled_count: [.[] | select(.status=="scheduled")] | length})'
   ```
   Each vid should have at most 1 scheduled trip bound to it (no double-assignment).

4. **UI (Task 3)** — load `http://localhost:3000/schedule`, pick a route with a parked test shuttle. The next scheduled row shows `#XYZ waiting` pill. Row before it (past) and row two slots in the future both show no waiting pill.

5. **Test regression**:
   ```bash
   cd C:/Users/Jzgam/OneDrive/Documents/GitHub/shubble && uv run pytest tests/ -x
   ```
   All existing + new tests pass.

6. **Frontend build**:
   ```bash
   cd frontend && npm run build
   ```
   No new TypeScript errors.
</verification>

<success_criteria>
- `uv run pytest tests/test_last_arrival_loop_scoping.py -x` passes (3+ new tests added, all pre-existing pass).
- `uv run pytest tests/ -x` passes (full-suite regression).
- `cd frontend && npm run build` passes.
- Live `/api/trips` invariant: no active trip has `stop_etas[*].last_arrival < actual_departure`.
- Live `/api/trips` invariant: each idle-at-Union shuttle is bound to the NEXT future scheduled trip for its route, and only to one trip.
- Schedule page renders `waiting` pill only on scheduled-status rows with an assigned vehicle_id.
- No pre-existing behavior (active trips, completed trips, unassigned scheduled rows, injected trips) regresses.
</success_criteria>

<output>
After completion, create `.planning/quick/260415-drf-idle-shuttle-scheduled-departure-context/260415-drf-SUMMARY.md` following the standard quick-task summary template: what changed, files touched, test results, curl verification output, and any deferred follow-ups (e.g., the map-marker "waiting" indicator was explicitly deferred in CONTEXT and should remain deferred).
</output>
