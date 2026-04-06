# Architecture Patterns: ETA UX Components

**Domain:** Real-time transit ETA display
**Researched:** 2026-04-05

## Recommended Architecture

All changes are frontend-only. The backend ETA pipeline (`/api/etas`) already exists and is unchanged. The architecture adds a thin layer of custom hooks and presentational components on top of the existing data flow.

### Component Boundaries

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `useStopETAs` (existing) | Polls `/api/etas` every 30s, returns `StopETAs` + `StopETADetails` | Schedule, MapKitCanvas |
| `useCountdown` (new hook) | Converts a target timestamp into a ticking seconds-remaining value via rAF | Schedule countdown hero, stop popups |
| `useDataFreshness` (new hook) | Derives `live`/`stale`/`offline` from last successful fetch time | FreshnessIndicator, missing data logic |
| `FreshnessIndicator` (new) | Renders pulse dot + "Live" / "Stale" / "Offline" label | Schedule header, map overlay |
| `DeviationBadge` (new) | Renders "+2 min late" / "On time" / "-1 min early" with color | Timeline stop items |
| `ETADisplay` (new) | Renders an ETA value with source label ("Live GPS" or "Scheduled") | Timeline stop items, stop popups |
| `StatusMessage` (new) | Renders contextual messages when data is missing | Timeline stop items (replaces "--:--") |

### Data Flow

```
/api/etas (30s poll)
    |
    v
useStopETAs hook (existing)
    |
    +---> stopETAs: Record<string, string>        (formatted times)
    +---> stopETADetails: Record<string, {...}>    (ISO strings, route, vehicleId, lastArrival)
    +---> lastFetchTimestamp (new field to add)
              |
              v
         useDataFreshness(lastFetchTs)
              |
              +---> freshnessStatus: 'live' | 'stale' | 'offline'
              |          |
              |          v
              |     FreshnessIndicator component
              |     StatusMessage component (uses freshness to pick message)
              |
         useCountdown(etaDetail.etaISO as ms)
              |
              +---> remainingSeconds: number | null
              |          |
              |          v
              |     Countdown hero ("Next shuttle in 3 min")
              |
         deviation = liveETADate - scheduledETADate
              |
              +---> DeviationBadge component
              +---> ETADisplay component (shows time + source label)
```

### Key Change to Existing Hook

The `useStopETAs` hook needs one addition: tracking `lastFetchTimestamp` so downstream components know data freshness.

```typescript
// In useStopETAs.ts -- add to state
const [lastFetchTs, setLastFetchTs] = useState<number | null>(null);

// In fetchETAs callback -- after successful response
setLastFetchTs(Date.now());

// Return it
return { stopETAs, stopETADetails, lastFetchTs };
```

## Patterns to Follow

### Pattern 1: Derived State via Hooks (Not Components)

**What:** Business logic (countdown math, freshness thresholds, deviation calculation) lives in hooks. Components are purely presentational.

**When:** Always. This is the project's existing pattern (`useStopETAs`, `useClosestStop`).

**Why:** Hooks are testable in isolation, reusable across Schedule and Map views, and keep components small.

```typescript
// Good: hook computes, component renders
const remaining = useCountdown(etaMs);
return <span>{remaining != null ? `${Math.ceil(remaining / 60)} min` : '--'}</span>;

// Bad: component does the math
return <CountdownTimer targetMs={etaMs} format="minutes" onExpire={...} />;
```

### Pattern 2: Additive Styling (Existing CSS Classes)

**What:** Use the CSS classes already defined (`.eta-early`, `.eta-late`, `.live-eta`, `.scheduled-fallback`, `.no-eta`, `.last-arrival`) rather than creating new styling systems.

**When:** For all ETA status display.

**Why:** These classes are already in `Schedule.css` with colors chosen for the existing design. Adding new classes or inline styles would fragment the visual language.

```typescript
// Good: use existing class
<span className={`secondary-timeline-time ${deviation > 0 ? 'eta-late' : 'eta-early'}`}>

// Bad: inline styles or new class names
<span style={{ color: deviation > 0 ? '#E67E22' : '#578FCA' }}>
```

### Pattern 3: Progressive Enhancement for Freshness

**What:** Data freshness degrades gracefully: live (pulse + blue) -> stale (dim + gray, "Data may be outdated") -> offline (hidden indicator, "Using scheduled times").

**When:** Whenever displaying ETA data.

**Why:** Students should never wonder "is this thing working?" The UI should always communicate its data state.

```
Live:    [pulsing green dot] "Live GPS"  +  blue ETA text
Stale:   [static yellow dot] "Updating..." + gray ETA text
Offline: [no dot]            "Scheduled"   + gray italic ETA text (existing .scheduled-fallback)
```

### Pattern 4: devNow() Consistency

**What:** All time calculations use `devNow()` and `devNowMs()` from `utils/devTime.ts` instead of raw `Date.now()`.

**When:** Always. The project already follows this pattern.

**Why:** Allows test server time simulation. The dev offset is computed once at module load, so there is zero runtime cost when `DEV_ENABLED = false`.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Timer State in Multiple Places

**What:** Having countdown logic in both the Schedule component and a countdown hook, or duplicating freshness checks.

**Why bad:** The current `Schedule.tsx` already has a `tick` state with `setInterval(30_000)`. Adding `useCountdown` without removing the tick state would create two competing timer systems.

**Instead:** Replace the `tick` state + 30s interval in Schedule with `useCountdown` for the hero display. Keep the `useStopETAs` 30s polling separate (it fetches data, not ticks display).

### Anti-Pattern 2: Over-Communicating Uncertainty

**What:** Showing confidence percentages, accuracy ranges, or "ETA accuracy: 73%" next to times.

**Why bad:** Students want a simple answer: "when is the shuttle coming?" Transit App's UX research shows that big, clear ETAs with minimal auxiliary info outperforms information-dense displays. Delay badges are useful but confidence scores are noise.

**Instead:** Binary source labels ("Live" vs "Scheduled") and deviation badges ("+2 min late"). Nothing more.

### Anti-Pattern 3: Animating Data Changes

**What:** Using CSS transitions or React Spring to animate ETA numbers when they change (e.g., sliding from "5 min" to "4 min").

**Why bad:** ETA values change every 30 seconds when new poll data arrives. Animating these changes draws attention to the update mechanism rather than the information. It also creates a brief period where the displayed value is between two states and therefore wrong.

**Instead:** Instant updates. The countdown hook ticks smoothly on its own; poll-driven ETA changes should snap to new values.

## Component File Structure

```
frontend/src/
  hooks/
    useStopETAs.ts        (existing -- add lastFetchTs)
    useClosestStop.ts     (existing -- no changes)
    useCountdown.ts       (new)
    useDataFreshness.ts   (new)
  components/
    FreshnessIndicator.tsx (new -- pulse dot + label)
    DeviationBadge.tsx     (new -- early/late/on-time)
    ETADisplay.tsx         (new -- time + source label)
    StatusMessage.tsx      (new -- contextual missing-data messages)
  schedule/
    Schedule.tsx           (existing -- integrate new components)
    styles/Schedule.css    (existing -- add pulse keyframes, minor additions)
  locations/
    MapKitCanvas.tsx       (existing -- integrate FreshnessIndicator in stop popups)
```

## Scalability Considerations

Not applicable to this milestone -- this is a frontend UX layer on an existing system. The backend handles data scaling. The frontend changes add negligible computational cost:

| Concern | Impact |
|---------|--------|
| rAF countdown on mobile | Pauses when tab backgrounded. ~1 state update/sec when visible. Negligible |
| Multiple DeviationBadge renders | Pure calculation, no external calls. O(number_of_stops) which is ~8-12 per route |
| FreshnessIndicator CSS animation | Compositor-only (`transform` + `opacity`). Zero main-thread cost |

## Sources

- Existing codebase: `useStopETAs.ts`, `Schedule.tsx`, `Schedule.css`, `devTime.ts`
- [Transit App 6.0 Design](https://blog.transitapp.com/six-o/) -- minimal auxiliary info around big ETAs
- [requestAnimationFrame patterns](https://css-tricks.com/using-requestanimationframe-with-react-hooks/) -- rAF + React hooks integration

---

*Architecture analysis: 2026-04-05*
