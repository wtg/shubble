---
task: 260415-emp
type: quick
wave: 1
depends_on: []
files_modified:
  - backend/worker/trips.py
  - frontend/src/schedule/Schedule.tsx
autonomous: true
requirements: [EMP-01, EMP-02, EMP-03]

must_haves:
  truths:
    - "Each (route, departure_time) pair appears as at most ONE row in /api/trips — no duplicate rows per slot across the active/completed/scheduled lifecycle."
    - "Every trip_id in /api/trips has the shape `{route}:{iso_departure_time}` with NO `:vid` or `:done` suffix."
    - "Future scheduled rows on /schedule never display the LIVE badge, even when idle-binding has populated their vehicle_id."
    - "Current-loop active rows still show LIVE badges at the first stop and at secondary stops with live ETAs/last-arrivals."
    - "Completed (DONE) rows sort to the TOP of each route's timeline; active and upcoming rows follow below in their existing chronological order."
    - "The waiting pill (commit 8ab4142) still renders when `loopTrip.status === 'scheduled' && loopTrip.vehicle_id` — unchanged by this task."
    - "The defensive la < actual_departure guard (commit fd24946) still holds — unchanged by this task."
  artifacts:
    - path: "backend/worker/trips.py"
      provides: "trip_id emissions without :vid or :done suffixes (3 sites)"
      contains: 'f"{route}:{trip_time.isoformat()}"'
    - path: "frontend/src/schedule/Schedule.tsx"
      provides: "LIVE badge gated on active-only status + DONE-first row ordering"
      contains: "status === 'active'"
  key_links:
    - from: "backend/worker/trips.py line ~1076"
      to: "frontend dedup by trip_id"
      via: "active trip emission"
      pattern: 'f"\{route\}:\{trip_time\.isoformat\(\)\}"'
    - from: "backend/worker/trips.py line ~1128"
      to: "frontend dedup by trip_id"
      via: "completed trip emission"
      pattern: 'f"\{route\}:\{prior_display\.isoformat\(\)\}"'
    - from: "backend/worker/trips.py line ~1233"
      to: "frontend dedup by trip_id"
      via: "scheduled trip emission (including idle-bound slots)"
      pattern: 'f"\{route\}:\{dep\.isoformat\(\)\}"'
    - from: "frontend/src/schedule/Schedule.tsx line ~1013"
      to: "LIVE badge conditional"
      via: "loopTrip.status === 'active'"
      pattern: "loopTrip\\?\\.status === 'active'"
    - from: "frontend/src/schedule/Schedule.tsx line ~768"
      to: "visibleItems ordering"
      via: "DONE-first partition"
      pattern: "status === 'completed'"
---

<objective>
Fix three tightly-coupled regressions from quick-260415-drf plus one new UX ask, all via the "fix forward" path locked in CONTEXT.md:

1. **Backend trip_id collision fix** — Strip `:vid` and `:done` suffixes from the 3 trip_id emission sites in `backend/worker/trips.py` so the canonical shape is `{route}:{iso_departure_time}`. This restores the single-row-per-slot invariant the frontend dedup relies on. Collapses ghost DONE rows + eliminates the stale "Passed" downstream artifact in one move.
2. **Frontend LIVE-gate** — Gate the LIVE badge on `loopTrip.status === 'active'` at the two first-stop emission sites (Schedule.tsx lines 1013 and 1046) AND the three secondary-stop sites (1185, 1223, 1228) that can fire for idle-bound scheduled rows. Scheduled rows with bound vehicle_ids must never display LIVE even though they carry forward-projected ETAs.
3. **Frontend DONE-reorder** — Pre-sort `visibleItems` at Schedule.tsx line ~768 so completed rows float to the TOP of each route's timeline, with active/upcoming rows retaining their current chronological order below.

Purpose: Restore trust in the schedule page. Students see exactly one row per slot, LIVE means live, and finished trips don't clutter the middle of the list.

Output:
- Commit 1: `fix(quick-260415-emp): strip trip_id :vid/:done suffixes` — 3 edits in trips.py
- Commit 2: `fix(quick-260415-emp): gate LIVE badge on active-only status` — 5 edits in Schedule.tsx
- Commit 3: `feat(quick-260415-emp): sort DONE trips to top of schedule timeline` — 1 edit in Schedule.tsx
- (Optional, Claude discretion) Test additions in tests/test_trip_id_shape.py

Preserves:
- The "waiting" pill at Schedule.tsx ~line 998 (commit 8ab4142) — still renders for idle-bound scheduled rows.
- The defensive la < actual_departure guard in build_trip_etas (commit fd24946) — untouched.
- The idle-binding side-channel in `compute_trips_from_vehicle_data` (commit 7a11b98) — vehicle_id still populated on scheduled rows; trip_id no longer gets the `:vid` suffix.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/quick/260415-emp-fix-forward-strip-trip-id-suffixes-gate-/260415-emp-CONTEXT.md
@.planning/quick/260415-drf-idle-shuttle-scheduled-departure-context/260415-drf-SUMMARY.md
@backend/worker/trips.py
@frontend/src/schedule/Schedule.tsx

<interfaces>
<!-- Key context for the executor. Extracted from live grep. -->

## Current trip_id emissions (backend/worker/trips.py)

Three f-string sites produce trip_id. ALL three need their suffixes stripped per D-03:

```python
# Line 1076 — active/completed trip (inside per-vehicle loop)
trip = {
    "trip_id": f"{route}:{trip_time.isoformat()}:{vid}",
    # ...
}

# Line 1128 — just-completed trip emitted alongside a newer active loop
completed_trip = {
    "trip_id": f"{route}:{prior_display.isoformat()}:{vid}:done",
    # ...
}

# Lines 1233-1237 — scheduled trip (including idle-bound slots) — CONDITIONAL suffix
trip_id = (
    f"{route}:{dep.isoformat()}:{vid_for_slot}"
    if vid_for_slot
    else f"{route}:{dep.isoformat()}"
)
```

After the fix, ALL three should produce `f"{route}:{<iso_dep>}"` with no vid/done suffix. The scheduled-trip site's conditional reduces to the else-branch — remove the ternary entirely.

The `vehicle_id` field remains on every trip dict (line 1081, 1133, and in the scheduled-trip dict further down). Idle-binding keeps populating `vehicle_id` on scheduled rows — that's what the "waiting" pill reads. Only the trip_id suffix is being removed.

## Current LIVE-badge sites (frontend/src/schedule/Schedule.tsx)

Five `source-badge source-live` spans render in the per-row map loop. Per D-01, LIVE must NEVER render on a `scheduled` row — even when idle-binding has set `loopTrip.vehicle_id`.

```tsx
// ~line 868 — loopTrip variable established per row
const loopTrip: Trip | null = trip;
// loopTrip.status can be 'active' | 'completed' | 'scheduled'

// ~line 1007-1015 — FIRST STOP, loopTrip branch (trip-based ETA)
if (loopTrip && firstStop) {
  const ti = getTripStopInfo(firstStop);
  if (ti?.etaTime) {
    return (
      <>
        <span className="live-eta"> - ETA: {ti.etaTime}</span>
        <span className="source-badge source-live">LIVE</span>  {/* line 1013 */}
      </>
    );
  }
  return null;
}

// ~line 1043-1048 — FIRST STOP, legacy fallback (no loopTrip at all)
return isCurrentLoop && fEta && fMatch ? (
  <>
    <span className="live-eta"> - ETA: {fEta}</span>
    <span className="source-badge source-live">LIVE</span>  {/* line 1046 */}
  </>
) : null;

// ~line 1182-1186 — SECONDARY STOP, hasLiveETA branch
{hasLiveETA ? (
  <>
    <span className="live-eta">{etaTime}</span>
    <span className="source-badge source-live">LIVE</span>  {/* line 1185 */}
    {/* deviation span follows */}
  </>
) : lastArrival ? (
// ~line 1220-1224 — SECONDARY STOP, lastArrival branch
  <>
    <span className="last-arrival">Last: {lastArrival}</span>
    <span className="source-badge source-live">LIVE</span>  {/* line 1223 */}
  </>
) : inferredPassed ? (
// ~line 1225-1229 — SECONDARY STOP, inferredPassed branch
  <>
    <span className="last-arrival">Passed</span>
    <span className="source-badge source-live">LIVE</span>  {/* line 1228 */}
  </>
) : ...
```

**Note on line 1046:** This branch only fires when `loopTrip` is null (pure scheduled row with no trip matched). When loopTrip is null, there's no status to check — the row is a pure-static schedule entry, not an idle-bound scheduled trip. The regression in CONTEXT is specifically about idle-bound scheduled trips WHERE `loopTrip.status === 'scheduled'`. So site 1046 technically does not need gating — but adding a belt-and-suspenders `isCurrentLoop` check is already in place there. Leave 1046 alone unless the fix is trivially consistent.

**Preferred implementation: gate via getTripStopInfo**

The cleanest single-point fix is to make `getTripStopInfo` (line 888) return null when `loopTrip.status === 'scheduled'`, because `ti?.etaTime` would then be falsy and lines 1013, 1185 auto-collapse. lastArrival at 1223 and inferredPassed at 1228 also derive from getTripStopInfo output and collapse. This is a ~2-line change:

```tsx
const getTripStopInfo = (stop: string) => {
  if (!loopTrip || !loopTrip.vehicle_id || !loopTrip.stop_etas[stop]) return null;
  // NEW: scheduled trips have no live data source — their stop_etas are
  // forward projections from a bound idle vehicle, not live predictions.
  // Return null so the row falls back to the static-schedule render path.
  if (loopTrip.status === 'scheduled') return null;
  // ... rest unchanged
```

This single guard covers sites 1013, 1185, 1223, 1228 at once. Site 1046 is untouched (its branch only fires when loopTrip is null).

## Current render loop & visibleItems (frontend/src/schedule/Schedule.tsx)

```tsx
// ~line 768 — visibleItems derived from timelineRows (already sorted by timeDate asc at line 500)
const visibleItems: TimelineRow[] = shouldTruncate
  ? timelineRows.slice(Math.max(0, anchorIdx - 1), anchorIdx + 6)
  : timelineRows;

// ~line 847 — render loop
return visibleItems.map((row) => {
  // ...
});
```

`TimelineRow` has `.trip: Trip | null` and `.trip?.status === 'completed'` identifies DONE rows. Pre-sort step needed before the map:

```tsx
// D-02 — sort DONE rows to the top while preserving relative order within each group
const orderedItems = [
  ...visibleItems.filter(r => r.trip?.status === 'completed'),
  ...visibleItems.filter(r => r.trip?.status !== 'completed'),
];
return orderedItems.map((row) => {
  // ...
});
```

Alternative: stable sort with comparator `(a, b) => (a.trip?.status === 'completed' ? 0 : 1) - (b.trip?.status === 'completed' ? 0 : 1)`. Either works; the filter-partition form is more readable.

## Existing scroll/focus logic to preserve

Line 361-372 has scrollable-row selection that skips `.trip-completed-badge` rows when looking for the current-loop anchor. After reordering, DONE rows at the top should NOT become the scroll anchor — verify the `filter(item => !item.querySelector('.trip-completed-badge'))` at line 372 still works on the reordered DOM. It should, since the filter operates on rendered DOM nodes regardless of their position in the list.

Line 677-682 explicitly excludes completed trips from the "soonest ETA" projection scan — unchanged by this task but worth noting the pattern: `if (row.trip.status === 'completed') continue;`

## Dedup site on frontend (where one-row-per-slot is enforced)

Search `frontend/src/schedule/Schedule.tsx` for how `timelineRows` are built. Rows with the same (route, dep) but different trip_ids coexist today because whatever dedup exists keys on trip_id. After the backend fix, the same (route, dep) will produce exactly one trip record, so the frontend's existing rendering naturally collapses to one row. If a defensive dedup-by-trip_id already exists in the frontend, no change needed. If /api/trips somehow still returns two rows for the same (route, dep) — that is a backend bug to investigate inside Task 1's verify step.
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Backend — Strip trip_id suffixes (3 emission sites)</name>
  <files>backend/worker/trips.py, tests/test_trip_id_shape.py</files>
  <action>
Per D-03 in CONTEXT.md: canonical trip_id shape is `{route}:{iso_departure_time}`. No `:vid`, no `:done`.

**Edit 1 — active trip emission (line 1076):**
```python
# BEFORE
"trip_id": f"{route}:{trip_time.isoformat()}:{vid}",
# AFTER
"trip_id": f"{route}:{trip_time.isoformat()}",
```

**Edit 2 — completed trip emission (line 1128):**
```python
# BEFORE
"trip_id": f"{route}:{prior_display.isoformat()}:{vid}:done",
# AFTER
"trip_id": f"{route}:{prior_display.isoformat()}",
```

**Edit 3 — scheduled trip emission (lines 1233-1237):** Collapse the ternary entirely. `vid_for_slot` remains stored on the `vehicle_id` field (that's what the waiting pill reads), but trip_id no longer encodes it.
```python
# BEFORE
trip_id = (
    f"{route}:{dep.isoformat()}:{vid_for_slot}"
    if vid_for_slot
    else f"{route}:{dep.isoformat()}"
)
# AFTER
trip_id = f"{route}:{dep.isoformat()}"
```
Also update the comment block above the assignment (lines 1228-1232) — remove the rationale about "append it to trip_id so the key is still unique" since the key is now always unique by construction (one row per `(route, dep)` per tick).

**Edit 4 (preserve invariant) — verify no other site adds a suffix.** Grep `backend/worker/trips.py` for `trip_id` assignments and confirm only the 3 above write to trip_id. Also confirm no downstream post-processing appends to trip_id.

**Edit 5 — optional but recommended: add a regression test.** Create `tests/test_trip_id_shape.py` with two tests:
1. `test_trip_id_has_no_vid_or_done_suffix` — build a minimal vehicle_data input that exercises active + completed + scheduled paths; assert every output trip_id matches `^[A-Z]+:[0-9T:.+-]+$` (no `:000000` vid, no `:done`).
2. `test_no_duplicate_route_departure_pairs` — same input; assert `Counter((t["route"], t["departure_time"]) for t in trips)` has no values > 1.

Use the existing test scaffolding style in `tests/test_last_arrival_loop_scoping.py` (fixtures, async test pattern, `compute_trips_from_vehicle_data` entry point).

**Do NOT touch:**
- The `vehicle_id` field on any trip dict (idle-binding reads it).
- The defensive la-guard at lines ~259, ~659, ~681 (commit fd24946).
- The idle-binding side-channel setup (`idle_at_union_by_route`, `idle_slot_assignments`, lines ~1186-1210).

**Why this is safe:** The `vehicle_id` field remains populated on every trip with a bound/assigned vehicle. The frontend's waiting-pill gate (`loopTrip?.status === 'scheduled' && loopTrip?.vehicle_id`) reads `vehicle_id`, not trip_id. Only the trip_id's dedup-key role is affected, and that's the bug we're fixing.
  </action>
  <verify>
    <automated>
./.venv/Scripts/pytest.exe tests/test_trip_id_shape.py tests/test_last_arrival_loop_scoping.py -v
    </automated>
    Also (manual, post worker-restart):
    ```bash
    # Duplicate-pair check (MUST print empty dict)
    curl -s http://localhost:8000/api/trips | python -c "from collections import Counter; import sys, json; trips=json.load(sys.stdin); c=Counter((t['route'], t['departure_time']) for t in trips); print('duplicates:', {k:v for k,v in c.items() if v>1})"

    # Suffix check (MUST print zero matches)
    curl -s http://localhost:8000/api/trips | python -c "import sys, json; trips=json.load(sys.stdin); bad=[t['trip_id'] for t in trips if ':done' in t['trip_id'] or t['trip_id'].count(':') > 2]; print('bad trip_ids:', bad); assert not bad"
    ```
  </verify>
  <done>
    - 3 trip_id emission sites in trips.py produce `{route}:{iso_dep}` only.
    - `vehicle_id` field still populated on active, completed, and idle-bound scheduled trips.
    - `pytest tests/test_trip_id_shape.py` passes (2 new tests, if added).
    - `pytest tests/test_last_arrival_loop_scoping.py` still passes (16 tests from prior task).
    - Commit message: `fix(quick-260415-emp): strip trip_id :vid/:done suffixes; canonical shape is {route}:{departure_time}`.
  </done>
</task>

<task type="auto">
  <name>Task 2: Frontend — Gate LIVE badge on active-only status</name>
  <files>frontend/src/schedule/Schedule.tsx</files>
  <action>
Per D-01 in CONTEXT.md: LIVE badge must NEVER render on a `scheduled` row, even when idle-binding has populated `loopTrip.vehicle_id`.

**Strategy: single-point gate in `getTripStopInfo`** (preferred — covers 4 of 5 sites at once).

**Edit 1 — `getTripStopInfo` guard (line ~888):**
```tsx
// BEFORE
const getTripStopInfo = (stop: string) => {
  if (!loopTrip || !loopTrip.vehicle_id || !loopTrip.stop_etas[stop]) return null;
  const info = loopTrip.stop_etas[stop];
  // ...

// AFTER
const getTripStopInfo = (stop: string) => {
  if (!loopTrip || !loopTrip.vehicle_id || !loopTrip.stop_etas[stop]) return null;
  // Scheduled trips have no live data source — their stop_etas are
  // forward projections from an idle-bound vehicle, not live predictions.
  // Gating here collapses the LIVE badge at all four in-row sites
  // (first-stop ~1013, secondary hasLiveETA ~1185, lastArrival ~1223,
  // inferredPassed ~1228) back to the static-schedule render path.
  // Per D-01 of quick-260415-emp.
  if (loopTrip.status === 'scheduled') return null;
  const info = loopTrip.stop_etas[stop];
  // ... (rest unchanged, including completed-trip branch at line ~900)
```

**Why this works:**
- Line 1013 (first stop trip branch) reads `ti = getTripStopInfo(firstStop)` → null → `ti?.etaTime` falsy → falls through past the LIVE render.
- Line 1185 (secondary hasLiveETA) depends on `si.hasETA`, which derives from `ti.etaTime` via `stopInfo.map(stop => { if (loopTrip) { const ti = getTripStopInfo(stop); ...` at line 1085. With `ti = null`, the map falls into the legacy-fallback else-branch, which is gated on `isCurrentLoop` — a scheduled row is not `isCurrentLoop` (see line 849: `currentRowKeys` only includes rows with `trip?.status === 'active' || 'completed'`). So the legacy fallback produces no LIVE badge either.
- Lines 1223, 1228 follow the same path — `si.lastArrival` and the derived `inferredPassed` will both be falsy for scheduled rows.
- Line 1046 (pure-static fallback) only fires when `loopTrip` is null, unaffected.

**Edit 2 (belt-and-suspenders) — explicit guard at line 1013:** For maximum clarity, also add an explicit check at the first-stop render site in case future refactors bypass `getTripStopInfo`:
```tsx
// BEFORE
if (loopTrip && firstStop) {
  const ti = getTripStopInfo(firstStop);
  if (ti?.etaTime) {

// AFTER
if (loopTrip && loopTrip.status === 'active' && firstStop) {
  const ti = getTripStopInfo(firstStop);
  if (ti?.etaTime) {
```
This makes the intent explicit at the render site AND matches the "Wrap each LIVE emission in a status check" directive in CONTEXT.md.

**Preserve explicitly:**
- The "waiting" pill render at line ~998-1000 (`{loopTrip?.status === 'scheduled' && loopTrip?.vehicle_id && (...)}`) — this is the surface users need to see an idle-bound shuttle. Do NOT touch.
- The `isCompletedTrip` handling at line 886 and the completed-trip branch in getTripStopInfo at ~line 900 — completed trips should still show `Last:` times with the LIVE badge (line 1223). Completed is not scheduled; the gate at Edit 1 only filters `status === 'scheduled'`.
- The DONE badge render at line 1002 — unrelated.
- The deviation span at lines ~1186-1218 — nested inside the hasLiveETA branch which already collapses when getTripStopInfo returns null; nothing to edit.

**Do NOT:**
- Add a new SCHED badge on scheduled rows (CONTEXT explicitly states user chose the Recommended option, NOT the "+SCHED badge" variant).
- Hide the scheduled row's scheduled-time display or other chrome. Only the LIVE badge is gated.
- Touch the legacy-fallback branch at line 1046 (it only fires when loopTrip is null, which by definition cannot be a scheduled-bound row).
  </action>
  <verify>
    <automated>
cd frontend && npm run build
    </automated>
    Also (manual, after backend + frontend reload):
    1. Open http://localhost:3000/schedule — select a route (e.g., NORTH).
    2. Find a FUTURE scheduled row that has a vehicle badge (`#NNN`) and the "waiting" pill.
    3. Confirm: NO `LIVE` badge anywhere on that row (not at first stop, not at secondary stops).
    4. Confirm: the waiting pill IS still visible on that row.
    5. Find an ACTIVE row (same route, current loop).
    6. Confirm: LIVE badges ARE present at first stop + secondary stops with live ETAs.
    7. Find a COMPLETED (DONE) row.
    8. Confirm: `Last: HH:MM` entries still show LIVE badge (completed trips still have live data).
  </verify>
  <done>
    - `getTripStopInfo` returns null for `status === 'scheduled'` trips.
    - First-stop render site at line ~1013 has an explicit `loopTrip.status === 'active'` guard.
    - `npm run build` passes with no new TypeScript errors.
    - Waiting pill (commit 8ab4142) still renders on idle-bound scheduled rows.
    - LIVE badges still render on active and completed rows where live data exists.
    - Commit message: `fix(quick-260415-emp): gate LIVE badge on loopTrip.status === 'active'; scheduled rows no longer show LIVE even when idle-bound`.
  </done>
</task>

<task type="auto">
  <name>Task 3: Frontend — Sort DONE rows to top of timeline</name>
  <files>frontend/src/schedule/Schedule.tsx</files>
  <action>
Per D-02 in CONTEXT.md: DONE (completed) rows float to the TOP of each route's timeline; active and upcoming rows follow below in their existing chronological order. Sort-only — no separate section header or collapsible, per the Recommended option.

**Edit 1 — pre-sort before the render map (line ~768-770):**

The current code:
```tsx
const visibleItems: TimelineRow[] = shouldTruncate
  ? timelineRows.slice(Math.max(0, anchorIdx - 1), anchorIdx + 6)
  : timelineRows;
```

Add a partition-based reorder immediately after (before the `return (` at line 772):
```tsx
const visibleItems: TimelineRow[] = shouldTruncate
  ? timelineRows.slice(Math.max(0, anchorIdx - 1), anchorIdx + 6)
  : timelineRows;

// D-02 (quick-260415-emp): float completed (DONE) rows to the top of the
// per-route timeline while preserving each group's internal chronological
// order. Active and upcoming rows keep their existing order below. The
// partition-preserving sort is stable because Array.filter preserves
// relative order within each input pass.
const orderedVisibleItems: TimelineRow[] = [
  ...visibleItems.filter(r => r.trip?.status === 'completed'),
  ...visibleItems.filter(r => r.trip?.status !== 'completed'),
];
```

**Edit 2 — use the reordered list in the render map (line ~847):**
```tsx
// BEFORE
return visibleItems.map((row) => {

// AFTER
return orderedVisibleItems.map((row) => {
```

**Scroll-anchor verification:** The DOM-level scroll-anchor filter at lines 361-372 (`filter(item => !item.querySelector('.trip-completed-badge'))`) operates on rendered DOM nodes regardless of document order, so it continues to correctly skip DONE rows when finding the current-loop scroll target. No changes needed to that logic.

**`soonestRowKeys` verification:** The soonest-ETA computation loop at line 675 iterates `timelineRows` (not `visibleItems`/`orderedVisibleItems`) and already has `if (row.trip.status === 'completed') continue;` at line 682. Order-independent. No changes needed.

**`anchorIdx` / `currentLoopIndex` verification:** These indices are computed against `timelineRows` (chronological order), not `visibleItems`. The row-slicing that produces `visibleItems` uses those indices directly. We reorder only the FINAL render, so the index math remains correct.

**Do NOT:**
- Add a separator header between DONE and active sections (user rejected Option 2 — sort-only).
- Hide DONE rows older than N minutes (user didn't ask; defer per CONTEXT's Claude's Discretion).
- Reorder `timelineRows` itself (breaks anchorIdx math).
- Change the `.trip-completed-badge` DONE marker (it's the visual cue users rely on).
  </action>
  <verify>
    <automated>
cd frontend && npm run build
    </automated>
    Also (manual, after reload):
    1. Open http://localhost:3000/schedule — select a route with recent completed loops (NORTH or WEST).
    2. Confirm: any row with the DONE badge appears at the TOP of the per-route timeline, ABOVE the active row and upcoming scheduled rows.
    3. If multiple DONE rows exist, confirm they're in chronological order (earliest DONE first, latest DONE last) — filter order preserves this.
    4. Confirm: active + upcoming rows retain their original chronological order below the DONE rows.
    5. Confirm: the current-loop auto-scroll still lands on the ACTIVE row (not a DONE row), even though DONE rows are now visually first.
    6. Confirm: expanding a DONE row still works (click to expand, stops render with Last: times).
  </verify>
  <done>
    - `orderedVisibleItems` partitions completed-first, rest-after, preserving within-group order.
    - Render map iterates `orderedVisibleItems` instead of `visibleItems`.
    - `npm run build` passes.
    - Manual verification: DONE rows at top, active auto-scroll still works, no visual regressions.
    - Commit message: `feat(quick-260415-emp): sort DONE trips to top of schedule timeline per-route`.
  </done>
</task>

</tasks>

<verification>
## Phase-level checks

After all three tasks commit + services reload:

**Backend invariants (curl):**
```bash
# MUST print empty dict {}
curl -s http://localhost:8000/api/trips | python -c "from collections import Counter; import sys, json; trips=json.load(sys.stdin); c=Counter((t['route'], t['departure_time']) for t in trips); print({k:v for k,v in c.items() if v>1})"

# MUST print nothing (all trip_ids have exactly one colon between route and timestamp)
curl -s http://localhost:8000/api/trips | python -c "import sys, json; trips=json.load(sys.stdin); [print(t['trip_id']) for t in trips if ':done' in t['trip_id'] or t['trip_id'].count(':') > 2]"

# Scheduled trips with bound vehicle_id (waiting pill surface) — SHOULD still be >= 0 (non-zero if an idle shuttle is parked at Union)
curl -s http://localhost:8000/api/trips | python -c "import sys, json; trips=json.load(sys.stdin); print(sum(1 for t in trips if t['status']=='scheduled' and t.get('vehicle_id')))"
```

**Frontend checks (http://localhost:3000/schedule):**
1. Select a route with an idle-bound scheduled row (look for the "waiting" pill).
2. Scheduled row: waiting pill visible, NO LIVE badges. 
3. Active row: LIVE badges where expected.
4. DONE rows: sorted to top of per-route list.
5. No duplicate rows per slot — each (route, departure_time) appears exactly once.

**Test suite:**
```bash
./.venv/Scripts/pytest.exe tests/ --ignore=tests/simulation/test_frontend_ux.py
# Expected: all prior tests pass + 2 new from Task 1 (if added) = 94+2 = 96 or similar
```
</verification>

<success_criteria>
1. Every `(route, departure_time)` pair appears at most ONCE in `/api/trips`.
2. Every trip_id in `/api/trips` matches `^[^:]+:[^:]+(:[^:]+)?$` where the optional third colon is only for ISO timezone offset (e.g. `NORTH:2026-04-15T19:30:00+00:00` — note the `+00:00` is part of the ISO timestamp, not a suffix). No `:done` substring anywhere.
3. Future scheduled rows (including idle-bound ones with vehicle_id) display NO LIVE badge on the schedule page.
4. Active rows still display LIVE badges at first stop + secondary stops with live ETAs.
5. Completed (DONE) rows sort to the top of each route's timeline, with active + upcoming rows below.
6. The "waiting" pill still renders on idle-bound scheduled rows (commit 8ab4142 preserved).
7. The defensive la < actual_departure guard still holds (commit fd24946 preserved).
8. `pytest tests/` passes (existing suite + any new trip_id tests).
9. `cd frontend && npm run build` passes.
10. Three atomic commits landed in order: backend trip_id fix → frontend LIVE-gate → frontend DONE-reorder.
</success_criteria>

<output>
After completion, create `.planning/quick/260415-emp-fix-forward-strip-trip-id-suffixes-gate-/260415-emp-SUMMARY.md` following the quick-task SUMMARY template (see prior 260415-drf-SUMMARY.md for shape). Include:
- One-liner describing the three fixes.
- Files Touched table.
- Commits table (3 commits).
- Test Results (pytest + npm build).
- Live Verification (curl output + schedule-page observations).
- Deviations from Plan (if any surfaced during execution).
- Deferred Follow-ups (e.g., whether to add a SCHED badge later; whether to hide old DONE rows).
</output>
