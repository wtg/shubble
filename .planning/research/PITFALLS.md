# Domain Pitfalls: ETA UX Polish

**Domain:** Real-time transit ETA display
**Researched:** 2026-04-05

## Critical Pitfalls

Mistakes that cause rewrites or major user trust issues.

### Pitfall 1: Timer Drift on Mobile

**What goes wrong:** Using `setInterval(1000)` for countdown ticking. On mobile browsers, when the tab is backgrounded or the device sleeps, `setInterval` callbacks queue up and fire in bursts when the tab refocuses. The countdown shows stale values for seconds before "catching up."

**Why it happens:** `setInterval` guarantees a minimum delay, not an exact one. Mobile browsers throttle background timers aggressively (Chrome throttles to 1/min for hidden tabs).

**Consequences:** Student opens app, locks phone, unlocks 2 minutes later. Countdown shows "5 min" for a second then jumps to "3 min." Trust destroyed.

**Prevention:** Use `requestAnimationFrame` with absolute time comparison (`targetMs - Date.now()`). rAF naturally pauses when the tab is hidden and resumes with correct values when visible. The `useCountdown` hook in STACK.md implements this.

**Detection:** Test by switching tabs for 30+ seconds and returning. If countdown jumps, the timer is drift-prone.

### Pitfall 2: False Precision in ETA Display

**What goes wrong:** Showing "2 min 47 sec" or exact timestamps like "2:14:33 PM" for shuttle ETAs. Students interpret this as reliable precision and complain when the shuttle arrives at 2:16 instead of 2:14.

**Why it happens:** Developers display the raw data they have. The backend computes ETAs to the second, so it is tempting to show that resolution.

**Consequences:** Students lose trust because the app appears inaccurate, when in reality the underlying data (30s GPS polls, LSTM predictions) simply cannot support second-level accuracy.

**Prevention:** Round to whole minutes. Show "< 1 min" or "Arriving" for the final minute. Never display seconds in ETA values.

**Detection:** If any ETA display includes seconds or the word "seconds", it is too precise.

### Pitfall 3: Ambiguous Data Source

**What goes wrong:** Displaying ETAs without indicating whether they come from live GPS tracking or static schedule fallback. A student sees "2:15 PM" and does not know if the shuttle is actually tracked near their stop or if that is just the scheduled time.

**Why it happens:** The current code already falls back to scheduled times when live data is unavailable (lines 218-240 of Schedule.tsx), but the fallback is visually almost identical to live data (gray italic vs blue is subtle).

**Consequences:** Student waits at a stop because the app shows "2:15 PM", but the shuttle is actually 10 minutes late and the displayed time was a schedule guess. Core trust violation.

**Prevention:** Explicit visual distinction: "Live GPS" label with pulse indicator for tracked ETAs; "Scheduled" label in muted styling for fallback. The distinction must be obvious at a glance, not subtle color differences.

**Detection:** Ask someone unfamiliar with the app: "Is this time based on real tracking or just the schedule?" If they cannot tell, the distinction is insufficient.

## Moderate Pitfalls

### Pitfall 4: Orange = Bad? UX Testing Says Otherwise

**What goes wrong:** Using orange/amber for "late" indicators. NJ Transit UX research found that 2 out of 3 test participants interpreted orange-highlighted routes as "faster" rather than "delayed."

**Prevention:** Pair color with text. Never rely on color alone. The `.eta-late` class uses orange (#E67E22) which is fine, but it must always appear alongside text like "2 min late", not just as a colored dot or background.

### Pitfall 5: Stale Data Without Warning

**What goes wrong:** The 30s polling interval means data can be up to 30s old at display time. If a poll fails silently (network error caught in try/catch), data becomes 60s, 90s, 120s old without the user knowing.

**Prevention:** Track `lastFetchTimestamp` in the `useStopETAs` hook. The `useDataFreshness` hook derives `live`/`stale`/`offline` status. Display a freshness indicator. The existing `fetchETAs` catch block (line 96 of `useStopETAs.ts`) logs to console but does not update any UI state.

### Pitfall 6: Layout Shift When Data State Changes

**What goes wrong:** When switching from "No shuttle in service" (long text) to "3 min" (short text), or when a deviation badge appears/disappears, the timeline items shift position. On mobile, this causes the user to lose their scroll position.

**Prevention:** Reserve fixed space for ETA display and badges. Use `min-width` on ETA containers. Use CSS `visibility: hidden` instead of removing elements when hiding optional badges.

### Pitfall 7: Replacing tick State Without Understanding Its Purpose

**What goes wrong:** The current `Schedule.tsx` has a `tick` state (line 39) with a 30s interval (line 69) that forces re-renders so the countdown summary updates. A developer might remove this when adding `useCountdown`, not realizing other parts of the component also depend on re-renders to update past/future classification of timeline items.

**Prevention:** When integrating `useCountdown`, keep the 30s re-render mechanism for the timeline (past-time classification) but use the countdown hook specifically for the hero display. Document why both exist.

## Minor Pitfalls

### Pitfall 8: Pulse Animation and prefers-reduced-motion

**What goes wrong:** The live indicator pulse animation runs continuously. Users with vestibular disorders or motion sensitivity find this distracting or nauseating.

**Prevention:** Wrap pulse animation in `@media (prefers-reduced-motion: no-preference)`. For reduced-motion users, show a static green dot instead of a pulsing one.

### Pitfall 9: devNow() Inconsistency in New Hooks

**What goes wrong:** New hooks use `Date.now()` directly instead of `devNowMs()`, breaking the dev time simulation used for testing.

**Prevention:** Grep for raw `Date.now()` in any new hook code. All time references must go through `devNow()` / `devNowMs()` from `utils/devTime.ts`.

### Pitfall 10: Countdown Reaches Zero But Data Has Not Refreshed

**What goes wrong:** The countdown reaches "< 1 min" or "Arriving" but the next poll has not happened yet. The shuttle may have already arrived, or the ETA may have shifted. The countdown sits at zero looking broken.

**Prevention:** When countdown reaches 0, switch to "Arriving..." state. On next poll, either update to new ETA or switch to "Arrived" (using `lastArrival` from `StopETADetails`). Do not leave "0 min" on screen.

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| Data source labels | Pitfall 3 (ambiguous source) | Make "Live" vs "Scheduled" distinction bold and obvious, not subtle |
| Early/late badges | Pitfall 4 (orange = bad?) | Always pair color with text label. Never color-only |
| Missing data messages | Pitfall 6 (layout shift) | Reserve fixed space with min-width on ETA containers |
| Countdown hero | Pitfall 1 (timer drift) | Use rAF, not setInterval |
| Countdown hero | Pitfall 10 (reaches zero) | Show "Arriving..." at zero, update on next poll |
| useCountdown integration | Pitfall 7 (removing tick) | Keep 30s tick for timeline classification, use hook for hero only |
| All new hooks | Pitfall 9 (devNow) | Use devNowMs() everywhere, never raw Date.now() |
| Pulse animation | Pitfall 8 (reduced motion) | Respect prefers-reduced-motion media query |

## Sources

- [NJ Transit UX Case Study](https://medium.com/@aidantoole/nj-transit-app-a-ux-annotation-and-conceptual-redesign-9798eeb7865b) -- orange interpretation issue
- [Timer drift in React](https://medium.com/@bsalwiczek/building-timer-in-react-its-not-as-simple-as-you-may-think-80e5f2648f9b) -- setInterval accuracy analysis
- [rAF auto-pause behavior](https://blog.webdevsimplified.com/2021-12/request-animation-frame/) -- background tab throttling
- Existing codebase: `Schedule.tsx` lines 39/69 (tick state), `useStopETAs.ts` line 96 (silent error handling)

---

*Pitfall analysis: 2026-04-05*
