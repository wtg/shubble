---
phase: 02-trust-foundation
verified: 2026-04-06T06:30:00Z
status: human_needed
score: 5/6 must-haves verified
human_verification:
  - test: "Open http://localhost:3000/schedule with the test server running, select today and a route, and expand the current loop secondary timeline. Confirm every stop row shows either a blue LIVE badge or a gray SCHED badge next to the ETA time. Also confirm the first-stop main timeline row shows a LIVE badge when a live ETA is present."
    expected: "Every stop row in the secondary timeline displays a LIVE or SCHED badge. The first-stop ETA line (e.g. '- ETA: 9:45 AM') has a LIVE badge immediately after the time."
    why_human: "Badge rendering requires a live browser session with real or mock GPS data flowing. Static code analysis confirms all four badge branches are in place, but correct rendering depends on component state and CSS application."
  - test: "With the test server running (docker-compose --profile test --profile backend up), wait until a shuttle is more than 2 minutes ahead or behind schedule. Verify a deviation badge appears (e.g. '+3 min late' in orange or '-4 min early' in blue). Then verify that stops within 2 minutes of schedule show NO deviation badge. Also verify that lastArrival rows show a LIVE badge but NO deviation badge."
    expected: "Deviation badge appears only for live ETAs with |diff| > 2 min. Dead zone stops and lastArrival times have no deviation badge."
    why_human: "Deviation depends on live GPS data diverging from the static schedule, which requires the test server's Gaussian noise to produce a >2 min offset. Cannot be triggered from static analysis."
  - test: "Switch the schedule day selector to a day with no service (if one exists), or wait until after the last departure time. Verify the stop cells show 'No shuttle in service' instead of '--:--'. Before the first departure, verify 'Service starts at X:XX AM' appears. Confirm '--:--' is gone from the entire schedule page."
    expected: "No '--:--' anywhere on the page. Contextual messages appear per service state."
    why_human: "Missing data messages only render in the final else branch, which requires a state where isCurrentLoop/isExpanded is false AND no static ETA exists for that stop. Time-dependent browser state required."
  - test: "Inspect all badge elements using browser devtools. Confirm that LIVE and SCHED badges contain the visible text 'LIVE' and 'SCHED' (not just color). Confirm deviation badges show '+N min late' or '-N min early' as text, not just colored dots."
    expected: "All trust signals are text-labeled, not color-only. Passing assistive technology audit."
    why_human: "Accessibility of rendered output (color contrast, screen reader label presence) requires browser inspection."
---

# Phase 2: Trust Foundation Verification Report

**Phase Goal:** Students can see at a glance whether ETA data is live or scheduled, whether the shuttle is early or late, and what missing data means
**Verified:** 2026-04-06T06:30:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | Every stop row in the expanded secondary timeline shows a LIVE or SCHED badge next to the ETA time | ? HUMAN_NEEDED | Code: all 4 branches (hasLiveETA, lastArrival, scheduled, missing) each contain a badge or contextual message. Confirmed via `grep -c "source-badge" Schedule.tsx` = 4. Rendering requires browser. |
| 2 | Live ETAs that deviate from schedule by more than 2 minutes show a deviation badge (+N min late / -N min early) | ? HUMAN_NEEDED | `computeDeviation` exists (line 217), called only inside `hasLiveETA` branch (line 415), uses `Math.abs(diffMinutes) <= 2` dead zone (line 231). Returns `+${diffMinutes} min late` / `${diffMinutes} min early` with correct classNames. Correctness under live data requires browser test. |
| 3 | Stops with no ETA data show contextual messages instead of --:-- | ? HUMAN_NEEDED | `getMissingDataMessage` implemented (lines 193-214). Returns "No shuttle in service" or "Service starts at ${routeTimes[0]}". `--:--` not found anywhere in Schedule.tsx. Correct rendering depends on time-of-day state. |
| 4 | Deviation badges only appear on live ETAs, never on scheduled or lastArrival times | ✓ VERIFIED | `computeDeviation` called exclusively inside the `hasLiveETA` branch (line 414-417). The `lastArrival` branch (lines 419-423) and scheduled branch (lines 424-428) contain no call to `computeDeviation`. |
| 5 | Deviations within +/-2 minutes show no badge (dead zone) | ✓ VERIFIED | Line 231: `if (Math.abs(diffMinutes) <= 2) return null;` — inclusive 2-minute dead zone exactly as specified in D-11. |
| 6 | Color coding always includes text labels for accessibility | ✓ VERIFIED | Source badges render text content "LIVE" / "SCHED" (lines 379, 413, 422, 427). Deviation badges render text "+N min late" / "N min early" (lines 234, 236). No color-only signals found. |

**Score:** 3/6 programmatically verified (truths 4, 5, 6); 3/6 require human browser test (truths 1, 2, 3)

All 4 roadmap success criteria are addressed in the implementation:
- SC-1 (source labels): LIVE/SCHED badges on all branches — code confirmed
- SC-2 (deviation display): computeDeviation with +N/−N text — code confirmed
- SC-3 (contextual messages): getMissingDataMessage replaces --:-- — code confirmed
- SC-4 (text labels for accessibility): All badges are text-labeled — code confirmed

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `frontend/src/schedule/styles/Schedule.css` | Badge CSS classes for LIVE/SCHED source badges, updated .eta-early/.eta-late, and .no-service-message | ✓ VERIFIED | `.source-badge` (line 312), `.source-badge.source-live` (line 324), `.source-badge.source-sched` (line 330), `.no-service-message` (line 337), `.eta-early` with `font-size: 12px; font-weight: 700` (lines 297-302), `.eta-late` with same (lines 304-309). |
| `frontend/src/schedule/Schedule.tsx` | Badge rendering, deviation calculation, missing data messages | ✓ VERIFIED | `getMissingDataMessage` (lines 193-214), `computeDeviation` (lines 217-238), `source-badge` used in JSX (lines 379, 413, 422, 427), `no-service-message` used in JSX (line 430). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| Schedule.tsx secondary timeline rendering | Schedule.css badge classes | className assignments in JSX | ✓ WIRED | `className="source-badge source-live"` (lines 379, 413, 422) and `className="source-badge source-sched"` (line 427). CSS classes `.source-badge.source-live` and `.source-badge.source-sched` confirmed in CSS. |
| Schedule.tsx deviation calculation | liveETADetails[stop].etaISO and staticETADates[stop] | Date arithmetic with 2-min dead zone | ✓ WIRED | `computeDeviation` reads `liveETADetails[stopKey]?.etaISO` (line 221) and `loopStaticDatesForStop[stopKey]` (line 224). `loopStaticDates` is passed as argument (line 415). `Math.abs(diffMinutes) <= 2` dead zone confirmed at line 231. |
| Schedule.tsx missing data | aggregatedSchedule day/route lookup | getMissingDataMessage function | ✓ WIRED | `getMissingDataMessage` reads `aggregatedSchedule[selectedDay]?.[routeName as Route]` (lines 194-195) and returns "No shuttle in service" or template string with `Service starts at` (lines 197-213). Called at line 430 with `safeSelectedRoute`. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| Schedule.tsx (badge rendering) | `liveETADetails` (for hasLiveETA, lastArrival, deviation) | `useStopETAs` hook (external or internal) | Yes — hook fetches from `/api/etas` with polling | ✓ FLOWING |
| Schedule.tsx (SCHED badge branch) | `loopETAs[stop]` / `loopStaticETAs[stop]` | `computeStaticETAs(originalIndex)` using `route.STOPS` offsets | Yes — computed from `routes.json` stop offsets | ✓ FLOWING |
| Schedule.tsx (getMissingDataMessage) | `aggregatedSchedule[selectedDay][routeName]` | `aggregated_schedule.json` (imported at module level) | Yes — static schedule data | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| TypeScript compiles with no errors | `node node_modules/typescript/bin/tsc --noEmit` | Exit code 0, no output | ✓ PASS |
| `--:--` removed from Schedule.tsx | `grep -c "\-\-:\-\-" src/schedule/Schedule.tsx` | 0 matches | ✓ PASS |
| source-badge used 4 times in JSX | `grep -c "source-badge" src/schedule/Schedule.tsx` | 4 | ✓ PASS |
| computeDeviation called only in hasLiveETA branch | Manual code inspection of lines 410-431 | Confirmed: only at line 415, inside `hasLiveETA ?` branch | ✓ PASS |
| Dead zone is inclusive at ±2 min | `grep "Math.abs" src/schedule/Schedule.tsx` | Line 231: `Math.abs(diffMinutes) <= 2` | ✓ PASS |
| Commits exist in git history | `git log --oneline` | `0258ba2` (Schedule.tsx) and `90f76a2` (Schedule.css) confirmed | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| TRUST-01 | 02-01-PLAN.md | ETA displays show "Live" or "Scheduled" label indicating data source | ✓ SATISFIED | `source-badge source-live` (LIVE) and `source-badge source-sched` (SCHED) on all secondary timeline branches. First-stop main timeline also gets LIVE badge (line 379). |
| TRUST-02 | 02-01-PLAN.md | Early/late deviation shown when live ETA differs from schedule (with 2-min dead zone) | ✓ SATISFIED | `computeDeviation` with `Math.abs(diffMinutes) <= 2` dead zone. Returns `+N min late` / `N min early` with `eta-late` / `eta-early` CSS classes. |
| TRUST-03 | 02-01-PLAN.md | Missing data shows contextual messages instead of "--:--" | ✓ SATISFIED | `getMissingDataMessage` returns "No shuttle in service" or "Service starts at X:XX AM". Note: "En route to first stop" was explicitly dropped per design decision D-10 (02-CONTEXT.md line 30) in favor of SCHED badge covering that state. "--:--" not found in Schedule.tsx. |

**Note on TRUST-03 scope reduction:** REQUIREMENTS.md lists "En route to first stop" as one of three example messages for TRUST-03. The design contract (02-CONTEXT.md D-10) explicitly decided this state is covered by the SCHED badge on the scheduled fallback branch. This was a documented design decision, not an oversight.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `frontend/src/schedule/Schedule.tsx` | 414-416 | Inline IIFE `{(() => {...})()}` for deviation render | Info | Idiomatic per SUMMARY patterns-established. Not a stub. Documented pattern. |

No stub patterns found. No TODO/FIXME/placeholder comments in modified files. No hardcoded empty arrays passed to badge-rendering props.

### Human Verification Required

#### 1. LIVE/SCHED Badge Visibility

**Test:** Start the test server (`docker-compose --profile test --profile backend up`), start frontend (`cd frontend && npm run dev`), open http://localhost:3000/schedule, select today and a route, expand the current loop. Inspect every stop row in the secondary timeline.

**Expected:** Every stop row shows either a blue "LIVE" badge or a gray "SCHED" badge immediately after the ETA time. The first-stop departure row shows a "LIVE" badge after the ETA when live data is present.

**Why human:** Badge rendering requires component state populated by live or mock GPS data. Static analysis confirms all code paths include badges, but CSS application and React rendering must be validated in-browser.

#### 2. Deviation Badge with Dead Zone

**Test:** With the test server running, observe the schedule page over several minutes as Gaussian noise causes timing variation. When a live ETA is more than 2 minutes from schedule, verify a deviation badge appears (e.g., "+3 min late" in orange `#E67E22` or "-4 min early" in blue `#578FCA`). Verify that stops within ±2 minutes of schedule show no deviation badge. Also verify that lastArrival rows (showing "Last: X:XX AM") get a LIVE badge but no deviation badge.

**Expected:** Deviation badge appears for |diff| > 2 minutes only on live ETAs. lastArrival times never show deviation badges.

**Why human:** Requires live GPS data with sufficient timing deviation to exceed the 2-minute dead zone. Test server uses Gaussian noise which may take time to produce an observable offset.

#### 3. Contextual Missing Data Messages

**Test:** Select a day with no service (e.g., Sunday if routes don't run), or wait until after the last departure time on today's schedule. Confirm stop cells show "No shuttle in service." Before the first departure time, confirm "Service starts at X:XX AM" appears. Scroll the entire schedule page and confirm "--:--" appears nowhere.

**Expected:** All formerly blank/dash cells now show a contextual message. "--:--" is absent from the page.

**Why human:** Missing data message path requires the component to be in a state where `isCurrentLoop || isExpanded` is true AND `loopETAs[stop]` and `loopStaticETAs[stop]` are both falsy. Reproducing this requires time-of-day conditions or schedule boundary navigation.

#### 4. Accessibility: Text-Only Badge Verification

**Test:** Using browser devtools (or a screen reader), inspect the LIVE and SCHED badge elements. Confirm their text content is "LIVE" and "SCHED" respectively. Inspect deviation badges and confirm they read "+N min late" or "N min early" as text.

**Expected:** All trust signals are conveyed by text, not color alone. Passes basic accessibility requirement.

**Why human:** Screen reader behavior and rendered text content require in-browser inspection.

### Gaps Summary

No code gaps found. All artifacts exist, are substantive, and are wired to live data sources. TypeScript compiles with zero errors. The `--:--` string is fully removed from Schedule.tsx.

The 4 human verification items above are required to confirm correct browser rendering — they do not indicate missing code, only that rendering requires a live browser session with time-dependent state.

---

_Verified: 2026-04-06T06:30:00Z_
_Verifier: Claude (gsd-verifier)_
