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
| 02-01-02 | 01 | 1 | TRUST-02 | — | N/A | manual | Visual: Deviation badge appears for >2 min offset, absent for <=2 min; verify with test server | — | ⬜ pending |
| 02-01-03 | 01 | 1 | TRUST-03 | — | N/A | manual | Visual: Missing data shows contextual messages | — | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- No Wave 0 tasks needed. No frontend test framework exists in this project. All phase requirements are validated via manual visual inspection and TypeScript compilation (`npx tsc --noEmit`). Deviation calculation logic is pure render-time computation verified through visual testing with the test server.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| LIVE/SCHED badge rendering | TRUST-01 | No frontend test framework; visual CSS rendering | Open schedule page with test server running; verify each stop shows LIVE or SCHED badge |
| Deviation badge display | TRUST-02 | No frontend test framework; visual + timing dependent | Wait for shuttle to deviate >2 min from schedule; verify "+X min late" badge appears |
| Missing data messages | TRUST-03 | No frontend test framework; visual + time-of-day dependent | Check schedule outside service hours; verify contextual messages replace "--:--" |
| Accessibility (text + color) | TRUST-02 SC#4 | Manual inspection | Verify deviation badges include text labels, not just color coding |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or are documented as manual-only (no frontend test framework)
- [x] Sampling continuity: all tasks verified via visual inspection + TypeScript compilation
- [x] Wave 0 not applicable — no test framework to scaffold
- [x] No watch-mode flags
- [x] Feedback latency < 10s (TypeScript compilation)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending — nyquist_compliant remains false (no automated unit tests possible without frontend test framework; manual verification is the accepted strategy for this phase)
