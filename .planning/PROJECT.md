# Shubble — ETA Accuracy & UX Milestone

## What This Is

Shubble is a real-time shuttle tracking app for RPI students. It shows live GPS positions on a map and estimated arrival times on a schedule page. This milestone focuses on making the ETA system trustworthy and polished — students should see whether data is live or scheduled, know if a shuttle is early/late, and never be confused by missing data.

## Core Value

Students trust the ETA numbers. They know if it's a live GPS estimate or a schedule guess, and they can see early/late status at a glance.

## Requirements

### Validated

- ✓ Live GPS tracking on map — existing
- ✓ Per-stop ETA computation (3-layer: LSTM/offset/history) — existing
- ✓ Schedule page with countdown summary and timeline — existing
- ✓ Schedule fallback when live data unavailable — existing
- ✓ Map stop popups with next shuttle time — existing
- ✓ 30-second auto-refresh of ETA data — existing
- ✓ Route polyline display on map — existing
- ✓ Two shuttles per route alternating on schedule (test server) — existing
- ✓ Early/late indicators — deviation from schedule with 2-min dead zone. Validated in Phase 2: Trust Foundation
- ✓ Data source transparency — LIVE/SCHED badges on every stop row. Validated in Phase 2: Trust Foundation
- ✓ Missing data explanation — contextual messages replacing "--:--". Validated in Phase 2: Trust Foundation

### Active
- [ ] Test server timing fix — shuttles wait at Student Union between loops, enter/exit once per service day
- [ ] Schedule-to-live handoff — smooth transition when live data appears or disappears mid-session

### Out of Scope

- Push notifications for delays — adds mobile complexity, not core to this milestone
- Historical analytics dashboard — separate feature, not ETA UX
- Admin interface — no user-facing admin needed
- WebSocket real-time updates — polling at 30s is sufficient for now
- LSTM model retraining — model exists, this milestone is UX not ML

## Context

- **Branch:** `430-use-schedule-matching-to-make-etas-more-accurate`
- **Users:** RPI students checking phones at shuttle stops
- **Backend ETA pipeline:** Worker polls GPS every 5s → computes per-stop ETAs (LSTM for next stop, offsets for subsequent, interpolation for passed) → caches in Redis → served via `/api/etas`
- **Frontend:** React 19 + TypeScript + Vite, Apple MapKit JS for maps
- **Test server:** Mock Samsara API simulating schedule-based shuttle movement with Gaussian noise
- **Existing ETA UX:** Countdown at top of schedule page, blue text for live ETAs, gray for schedule fallback. `eta-early` and `eta-late` CSS classes exist but are never applied.
- **Test server fix already implemented:** Removed DB restore and CSV replay, rewrote schedule-based simulation with upfront action queuing and departure-time waiting

## Constraints

- **Tech stack**: React 19, FastAPI, existing Apple MapKit integration — no framework changes
- **Data freshness**: ETAs refresh every 30s; countdown display must stay in sync
- **Mobile-first**: Most students check on phones — UI must work well on small screens

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Remove CSV replay and DB restore from test server | Confusing, broken, didn't match real behavior | ✓ Good |
| Queue all shuttle actions upfront with depart_at | Simpler than background scheduler threads, no race conditions | — Pending |
| Show data source (live vs schedule) to users | Students need to trust the numbers — transparency builds trust | — Pending |

---
*Last updated: 2026-04-05 after initialization*
