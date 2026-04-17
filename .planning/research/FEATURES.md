# Feature Landscape: ETA UX Polish

**Domain:** Real-time transit ETA display
**Researched:** 2026-04-05

## Table Stakes

Features users expect. Missing = students do not trust the ETA numbers.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Live countdown ("3 min") | Transit apps universally show minutes-until-arrival, not just clock time | Low | Current code shows "X min" in summary but updates only every 30s. Needs per-second ticking via rAF hook |
| Data source transparency | Students need to know if "2:15 PM" is a GPS-derived estimate or a static schedule guess | Low | Add "Live" / "Scheduled" label next to ETAs. Existing `liveETADetails` already distinguishes sources |
| Early/late deviation | Every transit app shows if a vehicle is ahead or behind schedule | Low | CSS classes `.eta-early` and `.eta-late` already exist but are never applied. Wire up deviation calculation: `liveETA - scheduledETA` |
| Graceful missing data | "--:--" with no explanation is confusing. Students need context | Medium | Replace with "No shuttle in service", "En route to first stop", "Service ended" based on time-of-day and data state |
| Data freshness indicator | If ETA data is 2 minutes old, students should know | Low | Pulse dot when live, dim/warning when stale, hidden when offline |

## Differentiators

Features that would set Shubble apart from generic transit apps. Valuable but not required for trust.

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Selected-stop countdown hero | Large, prominent countdown for the student's chosen stop (already partially built) | Low | Existing `next-shuttle-summary` div needs second-level ticking and source label |
| Smooth live-to-schedule handoff | When live data disappears, transition gracefully without jarring UI change | Medium | Animate opacity/color transition. Show "Switched to scheduled times" briefly |
| Per-stop deviation badges in timeline | Each stop in the expanded timeline shows "+2 min late" or "-1 min early" | Low | Deviation data calculable from `liveETADetails.etaISO` vs `computeStaticETADates` |
| Color-coded deviation | Green for early/on-time, orange for moderately late, red for significantly late | Low | Thresholds: on-time (<1 min), early (negative), late (1-5 min orange, 5+ min red) |
| Contextual status messages per stop | "Arrived 2:14 PM" for passed stops, "Next: ~3 min" for upcoming | Medium | Requires combining `lastArrival` data with countdown logic per stop |

## Anti-Features

Features to explicitly NOT build in this milestone.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Sub-second countdown precision | Students do not need "2 min 43 sec" -- transit ETAs are inherently imprecise (30s poll, GPS noise). Sub-minute precision creates false confidence | Round to whole minutes. Show "< 1 min" for final minute |
| Push notifications for delays | Adds mobile complexity (service workers, notification permissions), entirely separate milestone | Show delay status in-app only |
| Animated shuttle interpolation | Interpolating position between 5s GPS polls looks smooth but creates inaccurate position display | Keep discrete marker updates. Show data age instead |
| Full countdown clock (HH:MM:SS) | Overly precise for transit. Makes the display feel like a bomb timer, not a friendly ETA | "X min" format. "Arriving" when < 1 min |
| Toast/snackbar notifications | Transient notifications are easy to miss. Status should be persistent and visible | Inline status indicators attached to relevant UI elements |
| Percentage-based progress bars | Transit progress is not linear -- a shuttle can be 80% distance-wise but only 50% time-wise due to stops | Use time-based countdown only |

## Feature Dependencies

```
Data source transparency --> Early/late deviation
  (need to distinguish live vs scheduled to calculate deviation)

Data freshness indicator --> Graceful missing data
  (freshness status drives which missing-data message to show)

useCountdown hook --> Selected-stop countdown hero
  (hero component uses the hook for per-second ticking)

useCountdown hook --> Per-stop deviation badges
  (badges show live deviation which needs tick-accurate comparison)

Graceful missing data --> Smooth live-to-schedule handoff
  (handoff is a specific case of missing-data handling)
```

## MVP Recommendation

Prioritize (Phase 1 -- trust foundation):
1. **Data source transparency** -- single highest-impact change. Students immediately know what they are looking at
2. **Wire up early/late CSS classes** -- CSS already exists, just needs comparison logic
3. **Replace "--:--" with contextual messages** -- removes the most confusing UX element
4. **Data freshness indicator** -- small CSS pulse dot, high trust signal

Defer to Phase 2 (polish):
- **useCountdown per-second ticking** -- improves experience but not critical for trust
- **Smooth live-to-schedule handoff** -- needs careful state management
- **Per-stop deviation badges** -- nice detail after core trust signals are in place
- **Color-coded deviation thresholds** -- refinement of early/late display

## Sources

- [Transit App 6.0 Design](https://blog.transitapp.com/six-o/) -- big ETA first, auxiliary info in cards
- [NJ Transit UX Case Study](https://medium.com/@aidantoole/nj-transit-app-a-ux-annotation-and-conceptual-redesign-9798eeb7865b) -- delay indicator usability testing (orange misinterpreted as "faster")
- [AltexSoft: Mobile UX for Transit](https://www.altexsoft.com/blog/best-mobile-user-experience-design-practices-for-public-transportation-apps/) -- table stakes for transit apps
- Existing codebase: `Schedule.tsx`, `useStopETAs.ts`, `Schedule.css`

---

*Feature analysis: 2026-04-05*
