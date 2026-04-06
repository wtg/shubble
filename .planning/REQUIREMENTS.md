# Requirements: Shubble ETA UX Polish

**Defined:** 2026-04-06
**Core Value:** Students trust the ETA numbers -- they know if data is live or scheduled, see early/late status, and never face unexplained missing data.

## v1 Requirements

### Trust Signals

- [ ] **TRUST-01**: ETA displays show "Live" or "Scheduled" label indicating data source
- [ ] **TRUST-02**: Early/late deviation shown when live ETA differs from schedule (with 2-min dead zone to avoid false signals)
- [ ] **TRUST-03**: Missing data states show contextual messages ("No shuttle in service", "Service starts at 9:00 AM", "En route to first stop") instead of "--:--"
- [ ] **TRUST-04**: Freshness indicator pulses when live data is current, warns when stale (>120s since last update)

### Countdown & Display

- [ ] **DISP-01**: Per-second countdown using requestAnimationFrame (not setInterval), rounded to whole minutes, "Arriving" when < 1 min
- [ ] **DISP-02**: Smooth live-to-schedule handoff with visual transition when data source changes mid-session
- [ ] **DISP-03**: Per-stop deviation badges in timeline showing "+2 min late" / "-1 min early" with color coding (green/orange/red)

### Test Server

- [x] **TEST-01**: Shuttles enter campus once at start of service, loop on schedule with waits at Student Union between departures, exit once after last loop
- [x] **TEST-02**: Two shuttles per route with alternating departures from static schedule
- [x] **TEST-03**: Gaussian noise model varies loop speed for realistic timing deviation

## v2 Requirements

### Future Polish

- **POLL-01**: Push notifications for significant delays (>5 min late)
- **POLL-02**: Historical on-time performance display per route
- **POLL-03**: Map stop popups with same trust signals as schedule page

## Out of Scope

| Feature | Reason |
|---------|--------|
| Sub-second countdown precision | Transit ETAs are inherently imprecise; sub-minute creates false confidence |
| Animated shuttle interpolation between GPS polls | Creates inaccurate position display; show data age instead |
| Full HH:MM:SS countdown clock | Overly precise for transit; "X min" format is standard |
| Toast/snackbar notifications | Transient notifications easy to miss; use inline status indicators |
| Percentage-based progress bars | Transit progress is not linear (stops, traffic); time-based countdown only |
| LSTM model retraining | Model exists; this milestone is UX not ML |
| WebSocket real-time updates | 30s polling sufficient for now |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| TRUST-01 | Phase 2 | Pending |
| TRUST-02 | Phase 2 | Pending |
| TRUST-03 | Phase 2 | Pending |
| TRUST-04 | Phase 3 | Pending |
| DISP-01 | Phase 3 | Pending |
| DISP-02 | Phase 4 | Pending |
| DISP-03 | Phase 4 | Pending |
| TEST-01 | Phase 1 | Complete |
| TEST-02 | Phase 1 | Complete |
| TEST-03 | Phase 1 | Complete |

**Coverage:**
- v1 requirements: 10 total
- Mapped to phases: 10
- Unmapped: 0

---
*Requirements defined: 2026-04-06*
*Last updated: 2026-04-05 after roadmap creation*
