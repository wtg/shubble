---
phase: 02-trust-foundation
plan: 01
subsystem: ui
tags: [react, css, schedule, trust-signals, accessibility]

# Dependency graph
requires:
  - phase: 01-test-server
    provides: Mock GPS data for live ETA testing
provides:
  - LIVE/SCHED source badges on every schedule stop row
  - Early/late deviation display with 2-min dead zone
  - Contextual missing data messages replacing --:--
affects: [03-eta-polish, 04-handoff]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Inline IIFE for computed JSX values in React render (computeDeviation)"
    - "Source badge pattern: .source-badge.source-{type} CSS class naming"
    - "Missing data message helper using schedule lookup"

key-files:
  created: []
  modified:
    - frontend/src/schedule/styles/Schedule.css
    - frontend/src/schedule/Schedule.tsx

key-decisions:
  - "lastArrival gets LIVE badge but no deviation badge (historical GPS, not predictive)"
  - "2-minute dead zone inclusive (Math.abs <= 2) per design contract D-11"
  - "No red tier for deviations -- only early (blue) and late (orange) per D-13"

patterns-established:
  - "Trust signal badges: text-based LIVE/SCHED with color as supplementary (accessibility)"
  - "Deviation display: +N min late (orange) / -N min early (blue) with dead zone"
  - "Missing data: contextual messages from schedule lookup, not generic placeholders"

requirements-completed: [TRUST-01, TRUST-02, TRUST-03]

# Metrics
duration: 15min
completed: 2026-04-06
---

# Phase 2 Plan 1: Trust Signal Badges Summary

**LIVE/SCHED source badges, early/late deviation display with 2-min dead zone, and contextual missing data messages on the Schedule page**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-06T05:40:00Z
- **Completed:** 2026-04-06T05:55:00Z
- **Tasks:** 3 (2 auto + 1 checkpoint)
- **Files modified:** 2

## Accomplishments
- Every stop row in the secondary timeline shows a blue LIVE or gray SCHED badge indicating data source (TRUST-01)
- Live ETAs deviating from schedule by more than 2 minutes display "+N min late" (orange) or "-N min early" (blue) deviation badges (TRUST-02)
- Missing ETA data shows "No shuttle in service" or "Service starts at X:XX AM" instead of "--:--" (TRUST-03)
- All badges include text labels alongside color for accessibility compliance

## Task Commits

Each task was committed atomically:

1. **Task 1: Add badge and deviation CSS classes to Schedule.css** - `90f76a2` (feat)
2. **Task 2: Add trust signal badges, deviation calculation, and missing data messages to Schedule.tsx** - `0258ba2` (feat)
3. **Task 3: Visual verification of trust signals** - checkpoint (user approved, no code changes)

## Files Created/Modified
- `frontend/src/schedule/styles/Schedule.css` - Added .source-badge, .source-live, .source-sched, .no-service-message classes; updated .eta-early/.eta-late with 12px font-size and font-weight 700
- `frontend/src/schedule/Schedule.tsx` - Added getMissingDataMessage helper, computeDeviation helper with 2-min dead zone, LIVE/SCHED badges on all stop rows, deviation display on live ETAs only

## Decisions Made
- lastArrival times show LIVE badge (they are from GPS data) but no deviation badge (they are historical, not predictive) -- per Pitfall 3 from RESEARCH.md
- Dead zone is inclusive at 2 minutes (Math.abs(diffMinutes) <= 2) -- per design decision D-11
- No red "very late" tier -- only early (blue #578FCA) and late (orange #E67E22) per D-13

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Known Stubs

None - all features are fully wired to existing data sources (liveETADetails, aggregatedSchedule, staticETADates).

## Next Phase Readiness
- Trust signal foundation is complete for schedule page
- Ready for Phase 2 Plan 2 (if any) or Phase 3 ETA polish work
- Deviation thresholds (2-min dead zone) established as baseline -- can be tuned with real data in future phases

---
*Phase: 02-trust-foundation*
*Completed: 2026-04-06*

## Self-Check: PASSED

- [x] frontend/src/schedule/styles/Schedule.css exists
- [x] frontend/src/schedule/Schedule.tsx exists
- [x] .planning/phases/02-trust-foundation/02-01-SUMMARY.md exists
- [x] Commit 90f76a2 found (Task 1)
- [x] Commit 0258ba2 found (Task 2)
