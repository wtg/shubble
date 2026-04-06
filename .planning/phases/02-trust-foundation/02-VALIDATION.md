---
phase: 2
slug: trust-foundation
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-04-06
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | vitest (frontend) / pytest (backend) |
| **Config file** | `frontend/vite.config.ts` |
| **Quick run command** | `cd frontend && npx vitest run --reporter=verbose` |
| **Full suite command** | `cd frontend && npx vitest run --reporter=verbose` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `cd frontend && npx vitest run --reporter=verbose`
- **After every plan wave:** Run `cd frontend && npx vitest run --reporter=verbose`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | TRUST-01 | — | N/A | manual | Visual: LIVE/SCHED badges visible on every stop row | — | ⬜ pending |
| 02-01-02 | 01 | 1 | TRUST-02 | — | N/A | unit | Deviation calculation: `>2 min = badge, <=2 min = no badge` | ❌ W0 | ⬜ pending |
| 02-01-03 | 01 | 1 | TRUST-03 | — | N/A | manual | Visual: Missing data shows contextual messages | — | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- Existing infrastructure covers all phase requirements. Frontend rendering tests are primarily visual/manual. Deviation calculation logic can be unit tested.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| LIVE/SCHED badge rendering | TRUST-01 | Visual CSS rendering | Open schedule page with test server running; verify each stop shows LIVE or SCHED badge |
| Deviation badge display | TRUST-02 | Visual + timing dependent | Wait for shuttle to deviate >2 min from schedule; verify "+X min late" badge appears |
| Missing data messages | TRUST-03 | Visual + time-of-day dependent | Check schedule outside service hours; verify contextual messages replace "--:--" |
| Accessibility (text + color) | TRUST-02 SC#4 | Manual inspection | Verify deviation badges include text labels, not just color coding |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 10s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
