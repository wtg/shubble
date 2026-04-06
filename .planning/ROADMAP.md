# Roadmap: Shubble ETA Accuracy & UX

## Overview

This milestone transforms Shubble's ETA display from raw numbers into a trusted information layer. Students will know whether they are seeing a live GPS estimate or a schedule fallback, whether the shuttle is early or late, and what it means when data is missing. The test server simulation is already complete; the remaining work is trust signals, countdown polish, and transition handling on the frontend.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3, 4): Planned milestone work
- Decimal phases (e.g., 2.1): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Test Server Simulation** - Schedule-based shuttle simulation with realistic timing noise (ALREADY COMPLETE)
- [ ] **Phase 2: Trust Foundation** - Data source labels, missing data messages, and early/late deviation display
- [ ] **Phase 3: Countdown & Freshness** - Per-second countdown timer and data freshness indicator
- [ ] **Phase 4: Transition & Polish** - Live-to-schedule handoff and per-stop deviation badges in timeline

## Phase Details

### Phase 1: Test Server Simulation
**Goal**: Developers have a realistic test environment that simulates schedule-based shuttle movement with timing variation
**Depends on**: Nothing (first phase)
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. Shuttles enter campus once at service start, loop on schedule with waits at Student Union, and exit after last loop
  2. Two shuttles per route depart on alternating schedule times
  3. Loop speeds vary with Gaussian noise so ETAs are not perfectly predictable
**Plans**: 0 plans (already implemented)
**Status**: Complete (2026-04-05)

Plans:
- [x] 01-01: Already implemented (schedule-based simulation with upfront action queuing)

### Phase 2: Trust Foundation
**Goal**: Students can see at a glance whether ETA data is live or scheduled, whether the shuttle is early or late, and what missing data means
**Depends on**: Phase 1
**Requirements**: TRUST-01, TRUST-02, TRUST-03
**Success Criteria** (what must be TRUE):
  1. Every ETA value on the schedule page shows a "Live" or "Scheduled" label indicating its data source
  2. When a live ETA deviates from the schedule by more than 2 minutes, the deviation is displayed (e.g., "3 min late") with appropriate color coding
  3. Where ETAs were previously "--:--", a contextual message explains the state ("No shuttle in service", "Service starts at 9:00 AM", "En route to first stop")
  4. Color-coded deviation always includes text labels (not color alone) for accessibility
**Plans**: 1 plan
**UI hint**: yes

Plans:
- [ ] 02-01-PLAN.md -- Trust signal badges, deviation display, and missing data messages

### Phase 3: Countdown & Freshness
**Goal**: Students see a ticking countdown to the next shuttle and know whether the data powering it is fresh
**Depends on**: Phase 2
**Requirements**: DISP-01, TRUST-04
**Success Criteria** (what must be TRUE):
  1. ETA countdown ticks per-second using requestAnimationFrame (not setInterval), rounded to whole minutes, showing "Arriving" when under 1 minute
  2. A freshness indicator is visible when live data is current (pulsing) and warns when data is stale (more than 120 seconds since last update)
  3. Countdown resumes accurately after the browser tab is backgrounded and returned to (no drift or jumps)
**Plans**: TBD
**UI hint**: yes

### Phase 4: Transition & Polish
**Goal**: Students experience smooth transitions when data source changes and can see per-stop early/late status in the timeline
**Depends on**: Phase 2, Phase 3
**Requirements**: DISP-02, DISP-03
**Success Criteria** (what must be TRUE):
  1. When live data appears or disappears mid-session, the ETA display transitions smoothly with a visual animation (no jarring snap between states)
  2. Each stop in the timeline shows a deviation badge ("+2 min late" / "-1 min early") with color thresholds (green for on-time, orange for moderate, red for significant delay)
  3. The data source label updates in sync with the transition so the user always knows what they are seeing
**Plans**: TBD
**UI hint**: yes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 (complete) -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Test Server Simulation | 1/1 | Complete | 2026-04-05 |
| 2. Trust Foundation | 0/1 | Not started | - |
| 3. Countdown & Freshness | 0/? | Not started | - |
| 4. Transition & Polish | 0/? | Not started | - |
