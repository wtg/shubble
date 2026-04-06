# Phase 2: Trust Foundation - Research

**Researched:** 2026-04-06
**Domain:** Frontend UI — React component modifications, CSS styling, schedule data interpretation
**Confidence:** HIGH

## Summary

Phase 2 is a frontend-only phase that adds trust signals to the existing Schedule.tsx component. All decisions are locked (D-01 through D-13) and the implementation requires no new dependencies, no backend changes, and no new API endpoints. The work is surgically scoped: add LIVE/SCHED badges, early/late deviation text, and contextual missing data messages to the existing secondary timeline rendering in Schedule.tsx.

The existing codebase already contains most of the building blocks: `.eta-early` and `.eta-late` CSS classes with correct colors (but never applied), `computeStaticETADates()` for deviation calculation, `liveETADetails` with ISO timestamps for date comparison, and the live/scheduled/lastArrival branching logic at lines 384-392. The work is primarily wiring these existing pieces together with new badge `<span>` elements and adding service-hours logic from `aggregated_schedule.json`.

**Primary recommendation:** Modify Schedule.tsx's secondary timeline rendering (lines 340-399) to add inline badge spans and deviation calculations using existing data, plus add 3-4 new CSS classes for badge styling. No new files, no new hooks, no new API calls.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Inline text badge ("LIVE" / "SCHED") displayed next to each ETA time on the schedule page
- **D-02:** Badge appears on every stop row in the expanded secondary timeline, not just the summary or first stop
- **D-03:** LIVE badge uses existing blue color (#578FCA), SCHED badge uses existing gray (#95a5a6)
- **D-04:** Badge is text-based (not icon-only) for accessibility -- color is supplementary, not the sole indicator
- **D-05:** Deviation shown as inline text badge after the ETA time (e.g., "+3 min late", "-1 min early")
- **D-06:** Deviation badge only appears on live ETAs -- scheduled fallback times never show deviation
- **D-07:** Deviation is computed by comparing live ETA to the scheduled time from `staticETADates` (already computed in Schedule.tsx)
- **D-08:** Two distinct missing data states replace "--:--": "No shuttle in service" (outside service hours) and "Service starts at X:XX AM" (before first departure)
- **D-09:** During active service loops, stops without live ETA show the scheduled time with a SCHED badge (existing fallback behavior, now explicitly labeled)
- **D-10:** "En route to first stop" state NOT included -- scheduled fallback with SCHED badge covers this case adequately
- **D-11:** 2-minute dead zone: deviations within +/-2 minutes are treated as on-time (no badge shown)
- **D-12:** 2-tier color scheme: early (>2 min) = blue (#578FCA, existing .eta-early), late (>2 min) = orange (#E67E22, existing .eta-late)
- **D-13:** No "very late" / red tier -- keep it simple with 2 tiers for now

### Claude's Discretion
- Exact badge font size, padding, and border-radius styling
- Whether SCHED badge is uppercase or title case
- How "Service starts at X:XX AM" determines the next service start time (from schedule data)
- Whether deviation rounds to nearest minute or truncates

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| TRUST-01 | ETA displays show "Live" or "Scheduled" label indicating data source | Existing live/scheduled branching at Schedule.tsx lines 384-392 provides the decision point; add `<span className="source-badge ...">` in each branch |
| TRUST-02 | Early/late deviation shown when live ETA differs from schedule (with 2-min dead zone) | `computeStaticETADates()` already returns Date objects per stop; `liveETADetails[stop].etaISO` provides live Date; deviation = live - scheduled in minutes |
| TRUST-03 | Missing data states show contextual messages instead of "--:--" | `aggregated_schedule.json` contains all departure times per day/route; first/last entry determines service window; logic replaces the `<span className="no-eta">--:--</span>` at line 391 |
</phase_requirements>

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| React | 19.2.4 | UI framework | Already installed, no changes needed [VERIFIED: frontend/package.json] |
| TypeScript | 5.9.2 | Type safety | Already installed [VERIFIED: frontend/package.json] |

### Supporting
No new libraries needed. This phase uses only existing dependencies.

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Inline `<span>` badges | Component library badges (MUI Chip, etc.) | Overkill; project has zero component library dependencies and decisions specify inline text |
| CSS classes | CSS-in-JS / styled-components | Contradicts existing pattern; project uses plain CSS files |

**Installation:** None required. Zero new dependencies per project decision. [VERIFIED: STATE.md "Zero new runtime dependencies" decision]

## Architecture Patterns

### Recommended Project Structure
No new files needed. Changes are limited to:
```
frontend/src/
  schedule/
    Schedule.tsx          # Modify: add badges, deviation, missing data logic
    styles/Schedule.css   # Modify: add badge styles, apply .eta-early/.eta-late
```

### Pattern 1: Inline Source Badge
**What:** Add a `<span>` with class `source-badge` after each ETA time string in the secondary timeline
**When to use:** Every stop row in the expanded secondary timeline (D-02)
**Example:**
```typescript
// In the secondary timeline rendering (Schedule.tsx ~line 384)
{hasLiveETA ? (
  <>
    <span className="live-eta">{activeETAs[stop]}</span>
    <span className="source-badge source-live">LIVE</span>
    {deviationBadge}  {/* only for live ETAs per D-06 */}
  </>
) : lastArrivalValid ? (
  <>
    <span className="last-arrival">{lastArrivalValid}</span>
    <span className="source-badge source-live">LIVE</span>
  </>
) : scheduledTime ? (
  <>
    <span className="scheduled-fallback">{scheduledTime}</span>
    <span className="source-badge source-sched">SCHED</span>
  </>
) : (
  <span className="no-eta">{missingDataMessage}</span>
)}
```
[VERIFIED: Schedule.tsx lines 384-392 current branching structure]

### Pattern 2: Deviation Calculation
**What:** Compute minutes difference between live ETA and scheduled time, apply dead zone
**When to use:** Only for live ETAs on the current loop (D-06)
**Example:**
```typescript
// Deviation calculation helper (inside Schedule.tsx or extracted inline)
const computeDeviation = (stopKey: string, loopStaticDates: Record<string, Date>): { text: string; className: string } | null => {
  const detail = liveETADetails[stopKey];
  if (!detail?.etaISO) return null;
  
  const scheduledDate = loopStaticDates[stopKey];
  if (!scheduledDate) return null;
  
  const liveDate = new Date(detail.etaISO);
  const diffMinutes = Math.round((liveDate.getTime() - scheduledDate.getTime()) / 60_000);
  
  // D-11: 2-minute dead zone
  if (Math.abs(diffMinutes) <= 2) return null;
  
  if (diffMinutes > 0) {
    return { text: `+${diffMinutes} min late`, className: 'eta-late' };
  } else {
    return { text: `${diffMinutes} min early`, className: 'eta-early' };
  }
};
```
[VERIFIED: computeStaticETADates() at line 180-189, liveETADetails type at useStopETAs.ts]

### Pattern 3: Missing Data Message
**What:** Replace "--:--" with contextual message based on service hours
**When to use:** When no live ETA, no lastArrival, and no scheduled time exist (line 391)
**Example:**
```typescript
// Determine missing data message
const getMissingDataMessage = (routeName: string, dayIndex: number): string => {
  const daySchedule = aggregatedSchedule[dayIndex];
  const routeTimes = daySchedule[routeName as Route];
  
  if (!routeTimes || routeTimes.length === 0) {
    return 'No shuttle in service';  // D-08: no schedule for this day
  }
  
  const now = devNow();
  const firstDeparture = timeToDate(routeTimes[0]);
  const lastDeparture = timeToDate(routeTimes[routeTimes.length - 1]);
  
  if (now < firstDeparture) {
    return `Service starts at ${routeTimes[0]}`;  // D-08: before first departure
  }
  
  if (now > lastDeparture) {
    return 'No shuttle in service';  // After last departure
  }
  
  return 'No shuttle in service';  // During service but no data
};
```
[VERIFIED: aggregated_schedule.json structure — array indexed by day-of-week, containing route->times mapping]

### Anti-Patterns to Avoid
- **Color-only indicators:** D-04 and success criterion #4 explicitly require text labels. Never use color as the sole differentiator.
- **Deviation on scheduled ETAs:** D-06 explicitly states deviation only appears on live ETAs. Showing deviation on SCHED times would be misleading (comparing schedule to itself).
- **Complex state management:** No useState/useEffect needed for badges. These are pure render-time computations from existing data.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Time formatting | Custom AM/PM parser | `Intl.DateTimeFormat` / existing `TIME_FORMAT` | Already used throughout; consistent locale handling |
| Date comparison | String comparison of time strings | `Date.getTime()` arithmetic | Existing pattern with `computeStaticETADates()` returns Date objects |
| Rounding | Custom rounding logic | `Math.round()` | Standard; already used in countdown logic at line 213 |

**Key insight:** Every computation needed already exists in the component. This phase is about surfacing existing data relationships as visual badges, not building new data pipelines.

## Common Pitfalls

### Pitfall 1: Timezone Mismatch in Deviation Calculation
**What goes wrong:** Live ETA ISO string is in UTC, `computeStaticETADates()` uses `new Date()` which creates local time. Deviation could be off by hours.
**Why it happens:** `timeToDate()` (line 91-104) creates dates using `setHours()` on a `new Date()`, which is local time. `liveETADetails.etaISO` is an ISO string that `new Date()` parses as UTC.
**How to avoid:** Both already end up in the same timezone when compared via `.getTime()` because `new Date(isoString)` and `new Date()` with `.setHours()` both resolve to absolute milliseconds. But verify this with the test server: if the backend returns ISO strings without timezone offset, there could be an issue.
**Warning signs:** Deviation values that are consistently off by a fixed amount (e.g., always +300 min).

### Pitfall 2: Missing Data Message During Current Loop
**What goes wrong:** Showing "No shuttle in service" when we're mid-loop but a stop simply doesn't have live data yet.
**Why it happens:** The `--:--` path (line 391) fires when `hasLiveETA` is false AND `scheduledTime` is null. But during the current loop, `scheduledTime` should always exist if the stop is in the route's STOPS array.
**How to avoid:** The `scheduledTime` fallback (line 388-389 with D-09's SCHED badge) should catch most in-service cases. The "No shuttle in service" message only fires when `loopStaticETAs[stop]` is also empty, which only happens for non-current loops or when `computeStaticETAs` returns `{}`.
**Warning signs:** "No shuttle in service" appearing during active service hours.

### Pitfall 3: lastArrival Getting a LIVE Badge Incorrectly
**What goes wrong:** `lastArrival` represents a past event (shuttle already visited this stop), not a live prediction. Labeling it "LIVE" could confuse students.
**Why it happens:** In the existing branching (line 386-387), `lastArrivalValid` is a separate case from `hasLiveETA`.
**How to avoid:** `lastArrival` should get a "LIVE" badge (it IS from live GPS data, not schedule), but should NOT get a deviation badge. It's historical, not predictive.
**Warning signs:** Deviation showing on times that already happened.

### Pitfall 4: Service Hours Edge Case — Last Loop Still Running
**What goes wrong:** After the last scheduled departure, the shuttle is still completing its loop. `getMissingDataMessage()` returns "No shuttle in service" but the shuttle is still active.
**Why it happens:** Last departure time !== last arrival time. The shuttle departs at 8:00 PM but the loop takes 10-15 minutes.
**How to avoid:** During the current loop (even if it's the last one), the secondary timeline is expanded and shows live ETAs or scheduled fallback with SCHED badge. The missing data message only appears in the `--:--` path, which won't fire for the current loop since `computeStaticETAs` will return values. This is a non-issue for the current loop rendering.
**Warning signs:** None expected — existing code structure prevents this naturally.

## Code Examples

### Existing Branching Structure (Schedule.tsx lines 384-392)
```typescript
// CURRENT CODE — what we're modifying
{hasLiveETA ? (
  <span className="live-eta">{activeETAs[stop]}</span>
) : lastArrivalValid ? (
  <span className="last-arrival">{lastArrivalValid}</span>
) : scheduledTime ? (
  <span className="scheduled-fallback">{scheduledTime}</span>
) : (
  <span className="no-eta">--:--</span>
)}
```
[VERIFIED: Schedule.tsx lines 384-392]

### Existing CSS Classes (Schedule.css lines 297-307)
```css
/* EXISTING — defined but .eta-early and .eta-late are never applied */
.eta-early {
  color: #578FCA;
  font-size: 0.8em;
  margin-left: 4px;
}

.eta-late {
  color: #E67E22;
  font-size: 0.8em;
  margin-left: 4px;
}
```
[VERIFIED: Schedule.css lines 297-307]

### New CSS Classes Needed
```css
/* Source badge — shared base */
.source-badge {
  display: inline-block;
  font-size: 0.65em;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  padding: 1px 5px;
  border-radius: 3px;
  margin-left: 6px;
  vertical-align: middle;
}

/* LIVE badge — blue (D-03) */
.source-badge.source-live {
  color: #578FCA;
  background: rgba(87, 143, 202, 0.12);
  border: 1px solid rgba(87, 143, 202, 0.3);
}

/* SCHED badge — gray (D-03) */
.source-badge.source-sched {
  color: #95a5a6;
  background: rgba(149, 165, 166, 0.12);
  border: 1px solid rgba(149, 165, 166, 0.3);
}

/* Missing data message */
.no-service-message {
  color: #95a5a6;
  font-style: italic;
  font-size: 0.85em;
}
```
[ASSUMED — styling details are Claude's discretion per CONTEXT.md]

### Aggregated Schedule Structure
```json
// aggregated_schedule.json — array indexed by day-of-week (0=Sunday)
[
  { "NORTH": ["9:00 AM", "9:20 AM", ...], "WEST": ["9:00 AM", ...] },  // Sunday
  { "NORTH": [...], "WEST": [...] },  // Monday
  ...
]
```
[VERIFIED: aggregated_schedule.json lines 1-50]

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `--:--` for all missing data | Contextual messages | This phase | Students understand WHY data is missing |
| No data source indicator | LIVE/SCHED badges | This phase | Students know if ETA is from GPS or schedule |
| `.eta-early`/`.eta-late` defined but unused | Applied with deviation text | This phase | Early/late status visible at a glance |

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `lastArrival` should get a LIVE badge since it comes from GPS data | Pitfall 3 | Minor — could argue it should have its own "ARRIVED" badge instead |
| A2 | Deviation should use `Math.round()` (nearest minute) | Pattern 2 | Negligible — truncation vs rounding differs by at most 1 minute |
| A3 | Badge styling (font-size, padding, border-radius) | Code Examples | None — explicitly Claude's discretion |
| A4 | The `--:--` case at line 391 only fires when the stop has no scheduled time AND no live data | Pitfall 2 | Could cause wrong missing data messages if assumption is wrong about `computeStaticETAs` coverage |

## Open Questions (RESOLVED)

1. **How should the countdown summary (line 284-291) handle trust signals?**
   - What we know: The summary shows "Next at [stop] in X min (time)" but doesn't indicate live vs scheduled
   - What's unclear: Whether TRUST-01 applies to the summary or only the secondary timeline
   - Recommendation: Add a small LIVE/SCHED indicator to the summary too for consistency, but this is secondary to the timeline work
   - RESOLVED: Summary not modified in this phase (Phase 3 scope). D-02 specifies "every stop row in the expanded secondary timeline" — summary is outside that scope.

2. **Should the first-stop main timeline item (line 328-329) also get a source badge?**
   - What we know: D-02 says "every stop row in the expanded secondary timeline" — the first stop line is part of the main timeline
   - What's unclear: Whether the first stop's ETA display (`- ETA: {activeETAs[firstStop]}`) needs a badge
   - Recommendation: Yes, for consistency — but at minimum the secondary timeline must have badges per D-02
   - RESOLVED: Yes, first-stop gets LIVE badge for consistency. Implemented in Plan 01 Task 2 section C.

## Environment Availability

Step 2.6: SKIPPED (no external dependencies identified). This phase is purely frontend code/CSS changes.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | None installed for frontend (no vitest/jest in package.json) |
| Config file | None |
| Quick run command | Manual visual testing via `npm run dev` |
| Full suite command | N/A |

### Phase Requirements -> Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| TRUST-01 | LIVE/SCHED badges appear on every stop row | manual-only | Visual inspection in browser | N/A |
| TRUST-02 | Deviation badge with 2-min dead zone | manual-only | Visual inspection; verify with test server shuttles running early/late | N/A |
| TRUST-03 | Missing data messages replace "--:--" | manual-only | Check before service hours, after service hours, day with no service | N/A |

**Justification for manual-only:** No frontend test framework exists in this project. The phase involves UI changes to a single component that are best validated visually. Adding vitest + react-testing-library is out of scope for this phase and would be a separate infrastructure task.

### Sampling Rate
- **Per task commit:** Visual verification with test server running (docker-compose --profile test --profile backend up)
- **Per wave merge:** Full walkthrough of all 3 trust signal types
- **Phase gate:** Screenshot evidence of all 3 requirement states

### Wave 0 Gaps
- No frontend test framework — all validation is manual/visual for this phase
- Test server (Phase 1) must be running to test live vs scheduled states

## Security Domain

Not applicable. This phase is frontend-only UI changes with no authentication, user input processing, API changes, or data handling modifications. All data sources are already validated by the existing backend.

## Sources

### Primary (HIGH confidence)
- `frontend/src/schedule/Schedule.tsx` — Full component source, branching logic, existing helpers
- `frontend/src/hooks/useStopETAs.ts` — StopETADetails type and data structure
- `frontend/src/schedule/styles/Schedule.css` — Existing CSS classes including unused .eta-early/.eta-late
- `frontend/src/types/vehicleLocation.ts` — StopETA and StopETAMap types
- `shared/aggregated_schedule.json` — Schedule structure (day-indexed array with route->times)
- `frontend/src/utils/config.ts` — Config including staticETAs flag

### Secondary (MEDIUM confidence)
- `.planning/phases/02-trust-foundation/02-CONTEXT.md` — All locked decisions D-01 through D-13
- `.planning/REQUIREMENTS.md` — TRUST-01, TRUST-02, TRUST-03 requirement definitions
- `.planning/STATE.md` — "Zero new runtime dependencies" constraint

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries, fully verified existing stack
- Architecture: HIGH — all integration points verified in source code
- Pitfalls: HIGH — examined actual code paths and data flow

**Research date:** 2026-04-06
**Valid until:** 2026-05-06 (stable — frontend code changes only, no external API dependencies)
