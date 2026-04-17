# Phase 2: Trust Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-06
**Phase:** 02-trust-foundation
**Areas discussed:** Data source labels, Early/late deviation, Missing data messages, Deviation thresholds

---

## Data Source Labels

### Q1: How should "Live" vs "Scheduled" be indicated next to each ETA?

| Option | Description | Selected |
|--------|-------------|----------|
| Inline text badge | Small "LIVE" or "SCHED" text badge next to the ETA time | ✓ |
| Icon + color only | Small dot or icon (pulsing green for live, clock for scheduled) with no text | |
| Header-level label | One label at top of timeline section instead of per-ETA labels | |

**User's choice:** Inline text badge
**Notes:** None

### Q2: Should the badge appear on every stop or only summary + first stop?

| Option | Description | Selected |
|--------|-------------|----------|
| Every stop row | Each stop in secondary timeline shows its own LIVE/SCHED badge | ✓ |
| Summary + first stop | Badge only on countdown summary and first stop of each loop | |

**User's choice:** Every stop row
**Notes:** None

---

## Early/Late Deviation

### Q3: How should early/late deviation be displayed?

| Option | Description | Selected |
|--------|-------------|----------|
| Text badge after ETA | "+3 min late" or "-1 min early" inline after time, color-coded | ✓ |
| Separate column | Dedicated deviation column with signed values | |
| Color bar + text | Colored left border on row plus small text deviation | |

**User's choice:** Text badge after ETA
**Notes:** User requested pros/cons analysis before choosing. Key factors: accessibility requirement (TRUST-02 criterion #4 rules out color-only), mobile-first constraint (rules out separate column), existing CSS colors match.

---

## Missing Data Messages

### Q4: Which missing data states should have distinct messages?

| Option | Description | Selected |
|--------|-------------|----------|
| No shuttle in service | Outside service hours entirely | ✓ |
| Service starts at X | Before first departure of the day | ✓ |
| En route to first stop | Shuttle departed but hasn't reached this stop yet | |
| Service ended | After last scheduled loop completed for the day | |

**User's choice:** No shuttle in service, Service starts at X
**Notes:** Multi-select question. "En route" and "Service ended" not selected.

### Q5: For stops with no live ETA during an active loop, what to show?

| Option | Description | Selected |
|--------|-------------|----------|
| Scheduled fallback with SCHED badge | Show scheduled time with gray SCHED badge | ✓ |
| "En route" message | Show "En route" text instead of a time | |
| Both | Scheduled time + "En route" subtitle | |

**User's choice:** Scheduled fallback with SCHED badge
**Notes:** None

---

## Deviation Thresholds

### Q6: What deviation thresholds and colors?

| Option | Description | Selected |
|--------|-------------|----------|
| 2-min dead zone, 2-tier | |±2 min| = on-time, >2 late = orange, >2 early = blue | ✓ |
| 2-min dead zone, 3-tier | Same + >5 min late = red | |
| No dead zone, 2-tier | Any deviation shows, could be noisy | |

**User's choice:** 2-min dead zone, 2-tier
**Notes:** None

---

## Claude's Discretion

- Badge styling (font size, padding, border-radius)
- SCHED badge casing (uppercase vs title case)
- Service start time determination logic
- Deviation rounding (nearest minute vs truncate)

## Deferred Ideas

None — discussion stayed within phase scope
