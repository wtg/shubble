# Phase 2: Trust Foundation - Context

**Gathered:** 2026-04-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Students can see at a glance whether ETA data is live or scheduled, whether the shuttle is early or late, and what missing data means. This phase adds trust signals to the existing schedule page — data source labels, early/late deviation badges, and contextual missing data messages. No new pages, no new API endpoints for frontend consumption (backend may need a `source` field), no countdown changes (Phase 3).

</domain>

<decisions>
## Implementation Decisions

### Data Source Labels
- **D-01:** Inline text badge ("LIVE" / "SCHED") displayed next to each ETA time on the schedule page
- **D-02:** Badge appears on every stop row in the expanded secondary timeline, not just the summary or first stop
- **D-03:** LIVE badge uses existing blue color (#578FCA), SCHED badge uses existing gray (#95a5a6)
- **D-04:** Badge is text-based (not icon-only) for accessibility — color is supplementary, not the sole indicator

### Early/Late Deviation
- **D-05:** Deviation shown as inline text badge after the ETA time (e.g., "+3 min late", "-1 min early")
- **D-06:** Deviation badge only appears on live ETAs — scheduled fallback times never show deviation
- **D-07:** Deviation is computed by comparing live ETA to the scheduled time from `staticETADates` (already computed in Schedule.tsx)

### Missing Data Messages
- **D-08:** Two distinct missing data states replace "--:--": "No shuttle in service" (outside service hours) and "Service starts at X:XX AM" (before first departure)
- **D-09:** During active service loops, stops without live ETA show the scheduled time with a SCHED badge (existing fallback behavior, now explicitly labeled)
- **D-10:** "En route to first stop" state NOT included — scheduled fallback with SCHED badge covers this case adequately

### Deviation Thresholds
- **D-11:** 2-minute dead zone: deviations within +/-2 minutes are treated as on-time (no badge shown)
- **D-12:** 2-tier color scheme: early (>2 min) = blue (#578FCA, existing .eta-early), late (>2 min) = orange (#E67E22, existing .eta-late)
- **D-13:** No "very late" / red tier — keep it simple with 2 tiers for now

### Claude's Discretion
- Exact badge font size, padding, and border-radius styling
- Whether SCHED badge is uppercase or title case
- How "Service starts at X:XX AM" determines the next service start time (from schedule data)
- Whether deviation rounds to nearest minute or truncates

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` — TRUST-01 (source labels), TRUST-02 (deviation display with 2-min dead zone), TRUST-03 (missing data messages), success criterion #4 (text labels not color alone)

### Existing Implementation
- `frontend/src/schedule/Schedule.tsx` — Main schedule component; contains `computeStaticETAs()`, `computeStaticETADates()`, live/scheduled/lastArrival display logic, countdown summary
- `frontend/src/hooks/useStopETAs.ts` — Shared ETA hook; `StopETADetails` type with eta, etaISO, route, vehicleId, lastArrival fields
- `frontend/src/types/vehicleLocation.ts` — `StopETA` and `StopETAMap` types for /api/etas response
- `frontend/src/schedule/styles/Schedule.css` — `.eta-early`, `.eta-late`, `.live-eta`, `.scheduled-fallback`, `.no-eta` CSS classes (early/late classes exist but are never applied)

### Shared Data
- `shared/routes.json` — Route definitions with STOPS and OFFSET values used for static ETA computation
- `shared/aggregated_schedule.json` — Schedule by day/route for determining service hours

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `.eta-early` and `.eta-late` CSS classes: Already defined with correct colors (blue/orange), just need to be applied in JSX
- `computeStaticETADates()` in Schedule.tsx: Already computes scheduled Date objects per stop per loop — deviation = live ETA date minus static date
- `useStopETAs` hook: Already provides `StopETADetails` with `etaISO` (ISO string) for date comparison
- `aggregatedSchedule` data: Already loaded — can determine service hours by checking first/last time entries per day

### Established Patterns
- ETA display uses `<span>` elements with CSS classes (`.live-eta`, `.scheduled-fallback`, `.no-eta`) — badges should follow same pattern
- Schedule.tsx already distinguishes live vs scheduled vs lastArrival in the rendering logic (lines 384-394) — source label logic fits into this existing branch
- 30-second re-render tick already drives countdown updates — deviation badges will update at same cadence

### Integration Points
- Schedule.tsx secondary timeline rendering (lines 340-399): Where LIVE/SCHED badges and deviation badges get added
- Schedule.tsx missing data display (line 392): Where `--:--` gets replaced with contextual messages
- Schedule.css: Where new badge styles get added (adjacent to existing `.eta-early`, `.eta-late`)
- Potentially `/api/etas` response: May need `source` field if backend distinguishes data source, or frontend can infer from presence in `liveETAs` vs `staticETAs`

</code_context>

<specifics>
## Specific Ideas

No specific requirements — open to standard approaches

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-trust-foundation*
*Context gathered: 2026-04-06*
