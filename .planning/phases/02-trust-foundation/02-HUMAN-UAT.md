---
status: partial
phase: 02-trust-foundation
source: [02-VERIFICATION.md]
started: 2026-04-06
updated: 2026-04-06
---

## Current Test

[awaiting human testing]

## Tests

### 1. LIVE/SCHED badge visibility
expected: Blue "LIVE" and gray "SCHED" badges render on all stop rows in the expanded secondary timeline
result: [pending]

### 2. Deviation badge with dead zone
expected: Stops with >2 min deviation show orange "+N min late" or blue "-N min early"; stops within 2 min show no deviation; lastArrival rows show no deviation badge
result: [pending]

### 3. Contextual missing data messages
expected: "No shuttle in service" when outside hours; "Service starts at X:XX AM" when before first departure (requires time-of-day conditions)
result: [pending]

### 4. Accessibility text-only check
expected: Badge text content ("LIVE"/"SCHED") is accessible via screen reader / devtools inspection
result: [pending]

## Summary

total: 4
passed: 0
issues: 0
pending: 4
skipped: 0
blocked: 0

## Gaps
