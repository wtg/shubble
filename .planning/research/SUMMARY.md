# Research Summary: ETA UX Polish

**Domain:** Real-time transit ETA display (frontend)
**Researched:** 2026-04-05
**Overall confidence:** HIGH

## Executive Summary

This milestone adds trust signals to Shubble's ETA display. The backend ETA pipeline already exists (3-layer LSTM/offset/history, polled every 30s, served via `/api/etas`). The frontend already shows ETAs with countdown summaries and timeline views. What is missing is the layer that tells students whether they can trust what they see: is this a live GPS estimate or a schedule guess? Is the shuttle early or late? Is data fresh or stale? What does "--:--" mean?

The research conclusion is emphatic: **zero new runtime dependencies are needed**. The entire feature set -- countdown timers, data freshness indicators, early/late badges, contextual status messages -- is achievable with custom React hooks (~100 lines total), CSS `@keyframes` animations, and browser-native `Intl.RelativeTimeFormat`. The existing codebase already has unused CSS classes (`.eta-early`, `.eta-late`) and the data structures needed to compute deviations (`liveETADetails` vs `computeStaticETADates`).

The key technical insight is using `requestAnimationFrame` instead of `setInterval` for countdown display. Mobile browsers throttle `setInterval` in background tabs, causing countdown drift and jarring jumps when students return to the app. An rAF-based hook using absolute time comparison is the standard solution, is ~20 lines of code, and automatically pauses when the tab is hidden (saving battery on phones).

Transit UX research from Transit App 6.0 and NJ Transit redesign studies informed the feature approach: big clear ETAs first, minimal auxiliary info, explicit data source labels, and always pairing color with text for status indicators (orange alone was misinterpreted as "faster" by 2/3 of test participants).

## Key Findings

**Stack:** Zero new dependencies. Custom hooks (`useCountdown`, `useDataFreshness`) + CSS + `Intl.RelativeTimeFormat` cover everything.

**Architecture:** Four small new components (`FreshnessIndicator`, `DeviationBadge`, `ETADisplay`, `StatusMessage`) as presentational wrappers around two new hooks. One modification to existing `useStopETAs` (add `lastFetchTimestamp`).

**Critical pitfall:** Timer drift on mobile from `setInterval`. Use `requestAnimationFrame` with absolute time comparison instead.

## Implications for Roadmap

Based on research, suggested phase structure:

1. **Trust Foundation** - Data source transparency + early/late badges + missing data messages
   - Addresses: Table stakes features (source labels, deviation display, contextual "--:--" replacement)
   - Avoids: Timer drift pitfall (not building countdown yet), layout shift (reserve fixed space from the start)
   - Rationale: These are the features that directly answer "can I trust this number?" No new hooks needed -- just wiring up existing data and CSS.

2. **Countdown Polish** - rAF countdown hook + hero display + freshness indicator
   - Addresses: Per-second countdown ticking, data freshness visibility, smooth zero-state handling
   - Avoids: False precision (round to minutes), stale data without warning (freshness indicator)
   - Rationale: Requires the new `useCountdown` hook and `useDataFreshness` hook. Build after trust signals are in place so the countdown always appears alongside source context.

3. **Transition Handling** - Live-to-schedule handoff + per-stop deviation badges + color-coded thresholds
   - Addresses: Differentiator features, smooth state transitions
   - Avoids: Over-communicating uncertainty, animating data changes
   - Rationale: Polish layer. Depends on Phase 1 (source labels) and Phase 2 (freshness tracking) being solid.

**Phase ordering rationale:**
- Phase 1 first because it delivers trust with minimal code changes (wiring existing data to existing CSS)
- Phase 2 second because the countdown hook is the largest new code and benefits from Phase 1's source labels being present
- Phase 3 last because transitions and refined badges are polish that builds on both prior phases

**Research flags for phases:**
- Phase 1: Standard patterns, unlikely to need deeper research
- Phase 2: The `useCountdown` hook pattern is well-documented, but integration with the existing `tick` state in Schedule.tsx needs care (Pitfall 7)
- Phase 3: Live-to-schedule handoff needs UX testing -- no standard pattern exists for this specific transition

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Zero-dependency approach verified against npm ecosystem. `Intl.RelativeTimeFormat` confirmed 95%+ browser support via MDN. rAF timer pattern confirmed across multiple authoritative sources |
| Features | HIGH | Table stakes mapped from Transit App 6.0 and NJ Transit research. Anti-features grounded in transit UX studies |
| Architecture | HIGH | Follows existing project patterns (hooks for logic, components for display). File structure matches codebase conventions |
| Pitfalls | HIGH | Timer drift is well-documented. Orange color interpretation backed by UX study. devNow() consistency verified from codebase analysis |

## Gaps to Address

- **Live-to-schedule handoff UX:** No standard transit app pattern found for this specific transition. Will need iterative design during Phase 3
- **Deviation threshold calibration:** The thresholds for "on time" (<1 min), "late" (1-5 min), "very late" (5+ min) are reasonable guesses but should be validated against actual shuttle deviation data from the test server
- **Map stop popup integration:** Research focused on Schedule page. The map stop popups (MapKitCanvas) also show ETAs and will need the same trust signals, but MapKit JS popup customization was not deeply researched

## Sources

- [MDN: Intl.RelativeTimeFormat](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/RelativeTimeFormat)
- [Transit App 6.0 Design](https://blog.transitapp.com/six-o/)
- [NJ Transit UX Case Study](https://medium.com/@aidantoole/nj-transit-app-a-ux-annotation-and-conceptual-redesign-9798eeb7865b)
- [Croct: React Countdown Timer Libraries (2025)](https://blog.croct.com/post/best-react-countdown-timer-libraries)
- [requestAnimationFrame vs setInterval](https://blog.webdevsimplified.com/2021-12/request-animation-frame/)
- [Timer Accuracy in React](https://medium.com/@bsalwiczek/building-timer-in-react-its-not-as-simple-as-you-may-think-80e5f2648f9b)
- [AltexSoft: Mobile UX for Transit](https://www.altexsoft.com/blog/best-mobile-user-experience-design-practices-for-public-transportation-apps/)

---

*Research completed: 2026-04-05*
