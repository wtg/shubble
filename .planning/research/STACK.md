# Technology Stack: ETA UX Polish

**Project:** Shubble ETA UX Milestone
**Researched:** 2026-04-05

## Recommended Stack

This milestone is **purely frontend UX work** on an existing React 19 + TypeScript + Vite app. The backend ETA pipeline already exists. The recommendation is zero new runtime dependencies -- use the platform, custom hooks, and CSS.

### Core (Already In Place -- No Changes)

| Technology | Version | Purpose | Status |
|------------|---------|---------|--------|
| React | 19.2.4 | UI framework | Existing |
| TypeScript | 5.9.2 | Type safety | Existing |
| Vite | 7.3.1 | Build tool | Existing |
| React Router | 7.13.1 | Client routing | Existing |
| Apple MapKit JS | latest | Map rendering | Existing |
| react-icons | 5.6.0 | Icon library | Existing |

### New: Custom Hooks (Build, Don't Install)

| Hook | Purpose | Why Custom |
|------|---------|------------|
| `useCountdown(targetMs)` | Countdown timer that ticks every second using `requestAnimationFrame` | Existing 30s `setInterval` drifts and updates too slowly. A rAF-based hook is ~20 lines and avoids the `react-timer-hook` dependency (v4.0.5, last published 2024, uses `setInterval` internally) |
| `useDataFreshness(lastFetchMs)` | Returns `"live"` / `"stale"` / `"offline"` status from last successful fetch timestamp | Simple derived state -- compare `Date.now() - lastFetch` against thresholds. No library needed |
| `useRelativeTime(dateMs)` | Returns "2 min", "45 sec" using `Intl.RelativeTimeFormat` | Native browser API with 95%+ support since 2020. No need for `react-timeago` (17KB) or `timeago.js` (2KB) |

### New: CSS-Only Patterns (No Libraries)

| Pattern | Purpose | Approach |
|---------|---------|----------|
| Live pulse indicator | Signal "data is live" next to ETA | CSS `@keyframes` pulse on a small dot. Uses `transform` + `opacity` (compositor-only, no paint jank) |
| Early/late badge | Show schedule deviation | CSS classes `.eta-early` and `.eta-late` already exist in `Schedule.css`. Wire them up with conditional logic |
| Data source label | "Live GPS" vs "Scheduled" transparency | `<span>` with CSS class toggle. No component library needed |
| Skeleton/placeholder states | Replace "--:--" with contextual messages | Conditional rendering with CSS transitions for smooth state changes |
| `prefers-reduced-motion` | Accessibility for animations | Media query to disable/slow pulse animation |

### Supporting Libraries (Already Installed -- Reuse)

| Library | Version | New Use |
|---------|---------|---------|
| react-icons | 5.6.0 | Icons for early/late/live/scheduled indicators (e.g., `FiClock`, `FiWifi`, `FiWifiOff`) |

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Countdown timer | Custom `useCountdown` hook (rAF-based) | `react-timer-hook` v4.0.5 (103k wkly downloads) | Adds dependency for ~20 lines of code. Uses `setInterval` internally which drifts. Last published 2024 |
| Countdown timer | Custom `useCountdown` hook | `react-countdown` (202k wkly downloads) | Over-engineered for this use case (we need "X min" not a full countdown clock). Ships renderer/utility code we would not use |
| Countdown timer | Custom `useCountdown` hook | `react-use-precision-timer` v3 | Good library but overkill. We need a countdown display, not a general-purpose precision timer |
| Relative time | `Intl.RelativeTimeFormat` (native) | `react-timeago` / `timeago.js` | Native API covers needs. 95%+ browser support. Zero bundle cost |
| Status badges | CSS classes + react-icons | Component library (MUI, Chakra) | Project has no component library and does not need one. Adding MUI for badges would create visual inconsistency |
| Animation | CSS `@keyframes` | Framer Motion / React Spring | Pulse dot and fade transitions are trivially CSS. Animation libraries are for complex orchestrated motion |
| Data fetching | Existing `useStopETAs` with polling | React Query / SWR | One polling endpoint does not justify a data-fetching framework. Would create inconsistency with rest of codebase |

## Key Implementation Details

### useCountdown Hook Pattern

```typescript
/**
 * Returns remaining seconds until targetMs, updating ~1/sec via rAF.
 * Returns null when targetMs is in the past or undefined.
 * Uses Date.now() comparison (not tick counting) to avoid drift.
 */
function useCountdown(targetMs: number | undefined): number | null {
  const [remaining, setRemaining] = useState<number | null>(null);

  useEffect(() => {
    if (!targetMs) { setRemaining(null); return; }

    let rafId: number;
    const tick = () => {
      const diff = Math.max(0, Math.round((targetMs - devNowMs()) / 1000));
      setRemaining(prev => prev === diff ? prev : diff);
      if (diff > 0) rafId = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(rafId);
  }, [targetMs]);

  return remaining;
}
```

Why `requestAnimationFrame` over `setInterval`:
- Syncs with browser repaint cycle (no wasted renders when tab is hidden)
- Uses absolute time comparison (`targetMs - Date.now()`) so it never drifts
- Automatically pauses when tab is backgrounded (saves battery on mobile -- critical since most users are students on phones)
- The existing 30s `setInterval` in `Schedule.tsx` remains for ETA data refresh; the rAF hook handles only countdown display

### Data Freshness Thresholds

```typescript
type FreshnessStatus = 'live' | 'stale' | 'offline';

function getDataFreshness(lastFetchMs: number | null): FreshnessStatus {
  if (!lastFetchMs) return 'offline';
  const ageMs = Date.now() - lastFetchMs;
  if (ageMs < 45_000) return 'live';     // Within 1.5x poll interval (30s)
  if (ageMs < 120_000) return 'stale';   // Missed 2-3 polls
  return 'offline';                        // No data for 2+ minutes
}
```

### Early/Late Deviation

```typescript
// deviationMinutes = liveETA - scheduledETA (positive = late, negative = early)
function formatDeviation(deviationMinutes: number): { label: string; className: string } {
  if (Math.abs(deviationMinutes) < 1) return { label: 'On time', className: '' };
  if (deviationMinutes > 0) return {
    label: `${Math.round(deviationMinutes)} min late`,
    className: 'eta-late'
  };
  return {
    label: `${Math.round(Math.abs(deviationMinutes))} min early`,
    className: 'eta-early'
  };
}
```

## What NOT to Install

| Library | Why Not |
|---------|---------|
| `react-timer-hook` | `setInterval`-based, drifts on mobile. Custom rAF hook is better and smaller |
| `react-countdown` | Full countdown clock component -- we need relative minutes, not HH:MM:SS |
| `framer-motion` | 32KB+ for animations that are pure CSS `@keyframes` |
| `@tanstack/react-query` | One polling endpoint does not justify a data-fetching framework |
| `date-fns` / `dayjs` / `luxon` | `Intl.RelativeTimeFormat` and `Intl.DateTimeFormat` (already used) cover all needs |
| `classnames` / `clsx` | Template literals handle conditional classes fine at this scale |
| Any component library | The app has its own CSS design. Adding MUI/Chakra creates visual inconsistency |

## Installation

```bash
# No new packages needed.
# All ETA UX features are built with:
# - Custom React hooks (useCountdown, useDataFreshness, useRelativeTime)
# - CSS @keyframes and existing CSS classes (.eta-early, .eta-late)
# - Intl.RelativeTimeFormat (browser native, 95%+ support)
# - react-icons (already installed)
```

## Confidence Assessment

| Recommendation | Confidence | Rationale |
|----------------|------------|-----------|
| Custom rAF countdown hook | HIGH | Well-documented pattern. rAF + absolute time comparison is standard for accurate browser timers. Multiple sources confirm `setInterval` drift issues |
| `Intl.RelativeTimeFormat` | HIGH | MDN confirms 95%+ browser support since Sept 2020. Codebase already uses `Intl.DateTimeFormat` |
| CSS-only pulse indicator | HIGH | Pure CSS on `transform`/`opacity` stays on compositor. Standard pattern for live indicators |
| No new dependencies | HIGH | Project has 4 runtime deps. Every feature is achievable with platform APIs and ~100 lines of custom code |
| Wire up existing `.eta-early`/`.eta-late` | HIGH | CSS classes already defined. Backend provides scheduled times. Just needs comparison logic |

## Sources

- [MDN: Intl.RelativeTimeFormat](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Intl/RelativeTimeFormat) - Browser API documentation
- [react-timer-hook on npm](https://www.npmjs.com/package/react-timer-hook) - v4.0.5, 103k weekly downloads
- [react-countdown on npm](https://www.npmjs.com/package/react-countdown) - 202k weekly downloads
- [requestAnimationFrame vs setInterval](https://blog.webdevsimplified.com/2021-12/request-animation-frame/) - Why rAF is more accurate
- [CSS Pulse Animation patterns](https://css3shapes.com/how-to-make-a-pulsing-live-indicator/) - Live indicator CSS technique
- [Croct: Best React Countdown Timer Libraries (2025)](https://blog.croct.com/post/best-react-countdown-timer-libraries) - Library comparison
- [Transit App 6.0 Design](https://blog.transitapp.com/six-o/) - Transit ETA UX patterns

---

*Stack analysis: 2026-04-05*
