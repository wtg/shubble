---
name: Phase 1 break detection — schedule-gap detector + on_break flag
description: Discussion-phase decisions for quick task 260415-oeb
type: quick-task-context
---

# Quick Task 260415-oeb — Context

**Branch:** `test-server-break-simulation` (detection sits alongside the sim commits from quick-260415-nm1).
**Gathered:** 2026-04-15
**Status:** Ready for planning

<domain>
## Task Boundary

Phase 1 of the backend break-detection work. Covers the single-shuttle-break-with-schedule-gap cases:
- Saturday all day (lunch + dinner each show up as long gaps in the aggregated schedule)
- Sat/Sun/Weekday dinner gap

OUT OF SCOPE (phases 2 & 3): weekday/Sunday LUNCH rotation, which has no schedule gap because other shuttles cover the slot. Those phases will use either driver-assignment gate (production signal) or Hungarian+break-column matching.

Existing Filter 2 (off-route) and Filter 3 (no-movement) in `backend/worker/trips.py` already skip trip emission for shuttles that drive off-route to a break spot or sit stationary — so trip suppression for the sim's simulated-break shuttles is already working. The NEW work here is:

1. **Schedule-gap window detection** — parse today's route schedule once, compute gap windows where no scheduled departure falls for >=35 min. Any shuttle whose current cycle time falls in such a gap is flagged on_break_schedule_gap.
2. **Expose an on_break flag** through `/api/locations` so the frontend can visually distinguish break shuttles from in-service ones.
3. **Frontend greyout** — muted marker style (opacity 0.5 + grey tint) for flagged shuttles; they remain visible on the map but clearly not actively in service.

</domain>

<decisions>
## Implementation Decisions

### Flag location (D-01)
- Add an `on_break: bool` field to each vehicle entry in the `/api/locations` response. Optional follow-up: `break_reason: Optional[str]` (e.g. `"schedule_gap"`, later `"driver_logoff"`, `"rotation"`) — but Phase 1 can ship with just the boolean and `None` reason.
- Flag is derived in `_build_locations_payload` or a helper called from there.

### Gap threshold (D-02)
- **35 minutes.** Conservative; well above the regular 10-minute cadence on weekdays. Captures all intentional lunch/dinner breaks without false-positives on any unusual but intentional short gaps.
- Constant named `SCHEDULE_GAP_THRESHOLD_SEC = 35 * 60` near other `trips.py` threshold constants.

### UI greyout (D-03)
- **Opacity 0.5 + grey tint** on the shuttle marker. Use a CSS filter (e.g. `filter: grayscale(0.8) opacity(0.5)`) or equivalent MapKit overlay styling on the marker when `on_break` is true.
- Shuttle stays visible on the map so operational awareness is preserved.
- No change to the shuttle's position / heading rendering — just the visual style.

### Claude's Discretion
- Whether to cache the per-route gap windows once per day (computed from `aggregated_schedule.json`) or recompute each worker cycle. Gap computation is trivial (<1ms) so per-cycle is fine; a once-at-startup cache with day-rollover invalidation is a minor optimization if desired.
- Frontend implementation specifics — whether to add a `.on-break` className on the marker element or set inline style. Executor picks whichever fits the existing MapKit marker styling pattern.
- How to emit `on_break` through the SSE stream — same shape (field on each vehicle) in the pushed payload as in the REST response. Consistency with the REST contract.

### Deliberately not in scope
- **Per-vehicle break state not from schedule gap** — Weekday/Sunday lunch rotation gets no signal from this detector. That's fine; Phase 2 will add it.
- **Trip-row suppression for gap cases** — already handled by existing Filter 2/3 when shuttles physically off-route or stationary. If a schedule-gap shuttle is somehow still emitting a trip, extending the filter is a small follow-up but not needed for Phase 1 landing.
- **Driver-assignment awareness** — Phase 2.

</decisions>

<specifics>
## Specific Ideas

### Schedule gap parsing
```python
# Once per day (or per worker cycle — cheap):
SCHEDULE_GAP_THRESHOLD_SEC = 35 * 60

def _compute_gap_windows(sched_deps: List[datetime]) -> List[Tuple[datetime, datetime]]:
    """Return (gap_start, gap_end) pairs between consecutive scheduled
    departures whose gap exceeds SCHEDULE_GAP_THRESHOLD_SEC."""
    windows = []
    for prev, nxt in zip(sched_deps, sched_deps[1:]):
        if (nxt - prev).total_seconds() >= SCHEDULE_GAP_THRESHOLD_SEC:
            windows.append((prev, nxt))
    return windows

def _in_schedule_gap(now_utc: datetime, gap_windows: List[Tuple[datetime, datetime]]) -> bool:
    return any(start < now_utc < end for start, end in gap_windows)
```

### Live verification
- Test sim with `DEV_TIME_SHIFT=1 DEV_TARGET_HOUR=6 DEV_TARGET_MINUTE=45` on a Saturday (day=6) — shuttle 001 is in a schedule gap (the 6:40→7:30 period the user mentioned). `curl /api/locations | jq '.[] | .on_break'` → `true` for shuttle 001.
- Same setup on a Weekday evening during dinner gap → `on_break=true`.
- During lunch-rotation window on Weekday → `on_break=false` (schedule is continuous). Phase 2 will cover this.

### Frontend rendering
- Likely touches `frontend/src/locations/MapKitMap.tsx` or a marker-rendering helper. Look for where `VehicleLocationData` is mapped to map markers; add conditional style/class based on `on_break`.

</specifics>

<canonical_refs>
## Canonical References

- `shared/aggregated_schedule.json` — source of gap info
- `backend/fastapi/routes.py` + `backend/fastapi/utils.py` — /api/locations response shape
- `frontend/src/types/vehicleLocation.ts` (or similar) — TypeScript type for vehicle entries
- Prior conversation: break-detection design discussion (schedule-gap, driver-assignment, Hungarian phases)
- Simulation artifacts: `.planning/quick/260415-nm1-simulate-shuttle-break-behavior-in-test-/`

</canonical_refs>
