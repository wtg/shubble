---
name: Break-detection architecture handoff
type: handoff
branch: test-server-break-simulation
status: Phase 1 shipped; Phase 3 re-architected (not started)
date: 2026-04-15
---

# Break-Detection Architecture Handoff

Context handoff for continuing the break-detection work next session. This document captures (a) what's shipped, (b) the design conversation we worked through, (c) what the next phase looks like after a significant re-architecture.

## Status snapshot

**Branch:** `test-server-break-simulation`

| Phase | Status | Notes |
|---|---|---|
| Phase 1 — schedule-gap detector | ✅ Shipped | quick-260415-oeb. `on_break` flag on `/api/locations` + muted map marker. Catches single-shuttle gaps (Saturday all day, Sat/Sun/Weekday dinner). |
| Phase 2 — driver-assignment gate | ❌ Skipped by user | "We're not asking drivers to log out." Can revisit if Samsara ever exposes it automatically. |
| Phase 3 — lunch-rotation detection | 🚧 Pivoted, not started | Originally planned as Hungarian-over-position-cost; re-architected to **historical-archetype matching** after data analysis. See below. |

## Design conversation summary

### Semantic clarification (important)

The user caught an imprecision in my earlier framing. **Any cost function derived from observed state is behavioral, not predictive.** Only *external categorical signals* (driver logout, dispatcher break plan) would be truly predictive. Everything based on current position / rhythm / patterns is behaviorally-informed. With that correction:

- **Purely reactive behavioral:** real-time gap detection (e.g. "shuttle hasn't visited Union in 40 min → break"). Lags actual break by threshold. Simple.
- **Historically-informed behavioral:** learn patterns offline (archetypes), apply to current observations to flag break earlier and with higher confidence. This is closer to the prior branch's "schedule matching" intent.
- **Purely predictive:** not attainable without external signals we don't have.

### Does Hungarian earn its keep?

Short answer: **at current scale, not really — sort-and-index is equivalent.**

- For 3 shuttles in strict sequential rotation with 1-D morning signatures (e.g. "last departure before lunch"), Hungarian reduces to an O(n log n) sort. Identical output.
- Hungarian matters when: (a) signatures are **multi-dimensional** (shuttles have 6-8 morning departures each → 6-8-dim vectors), AND (b) data is **noisy enough that greedy can cascade errors**. Both become true once we model the full morning pattern, not just a single departure.
- Recommendation: ship simple sort/greedy first, leave a pluggable matching seam so Hungarian is a ~10-line swap if live-data noise justifies it.

## Historical-data validation

Analysis of `ml/cache/shared/locations_raw.csv` (500K pings, 2025-07-31 → 2025-09-23, 49 days, 24 unique vehicle IDs, 6-9 active per day):

### Per-shuttle Union-visit rhythm (Thursday 2025-09-18, representative weekday)

| Shuttle | Visits | Median interval | P95 | Break-length gaps (>40 min) |
|---|---|---|---|---|
| 379809 | 62 | 6m | 15m | 1 (16:21→17:04, 43m) |
| 465689 | 75 | 8m | 18m | 2 (incl. overnight + 17:21→18:08, 47m) |
| 575559 | 53 | 12m | 16m | 1 (15:43→16:25, 42m) |
| 494756 | 67 | 6m | 15m | 0 |
| 768444 | 71 | 13m | 20m | 0 |
| 829898 | 27 | 6m | 15m | 0 |

Four shuttles show clean sequential lunch rotation with ~30-min offsets:
```
575559: break 15:43  ← archetype 1
379809: break 16:21  ← archetype 2 (+38 min)
494756: break 16:47  ← archetype 3 (+26 min)
465689: break 17:21  ← archetype 4 (+34 min)
```

### Aggregate break-start hour distribution (106 candidates, weekdays only)

```
14:00 #########        (9)
15:00 ###############  (15)
16:00 ###########################  (27)   ← peak lunch rotation
17:00 ################  (16)
22:00 ###############################  (31)   ← dinner / end-of-shift
```

Afternoon cluster 14:00-18:00 = lunch rotation. 22:00 cluster = end-of-day (already handled by Phase 1 schedule-gap detector since it's a single-shuttle window with schedule gap).

**Conclusion:** archetypes clearly exist in the data. 3-4 per route on weekdays, 2 per route on Sundays, 1 per route on Saturdays (degenerate — schedule-gap handles it).

## Proposed Phase 3 architecture: historical archetype matching

### Offline (batch, once per day-of-week or on ingest of new data)

Per route per day-of-week:
1. Extract per-shuttle break-start times + morning-departure signatures from historical data
2. Cluster shuttles by break time (k-means or 1-D histogram bucketing; k=1 Sat, k=2 Sun, k=3-4 weekday)
3. For each cluster, compute:
   - **Archetype break time** (cluster center)
   - **Archetype morning signature** (mean vector of morning-departure times of members)
4. Persist archetypes to disk or Redis (TTL: full week)

### Production (real-time, per worker cycle)

For each active shuttle:
1. Observe its morning departure times (pre-lunch-window)
2. Compare morning signature to each archetype's signature (vector distance)
3. Match via **sort** (for single-value sig) or **Hungarian** (for multi-D sig + noise robustness)
4. Predict this shuttle's break time from its matched archetype
5. Flag `on_break = true` when current time >= predicted_break_start - small grace window, OR when actual gap > 1.5× learned median interval (whichever fires first)

### Integration point

Same code path as Phase 1 schedule-gap detector: `backend/fastapi/utils.py`. The `_build_locations_payload` in `routes.py` already computes `on_break` per vehicle — OR the archetype result with the existing schedule-gap result. Single flag on `/api/locations`, no API changes.

## Open design decisions (to lock in next session)

These were left unanswered when context filled up:

1. **Gap threshold for fallback** — historical data suggests 40 min is the natural cutoff (breaks 42-50m, P95 normal 15-20m). Candidates: 30m (faster) / 40m (matches data) / 25m (aggressive).

2. **Signature dimensionality** — single-value (last morning departure time, or phase mod 30) vs multi-value vector (all morning departures). Multi-value is more robust but requires a warm-up.

3. **Matching algorithm** — sort-and-index (1-D sig, simple) vs greedy-best-match (multi-D, moderate) vs Hungarian (multi-D, noise-robust, lead's preference). Per honest comparison: sort ≈ Hungarian at current scale, Hungarian earns its keep as scale grows.

4. **State location for last-Union-visit timestamp** — in-memory dict / Redis key / derived live from vehicle_locations DB. In-memory is simplest; Redis survives restart; DB is most accurate but most load.

5. **Archetype persistence** — how often to rebuild archetypes (daily? weekly? on-demand when drift is detected?). Where to store them (JSON file in `shared/`? Redis? DB table?).

## Next-session checklist

1. **Decide** the 5 open design decisions above.
2. **Build offline archetype extractor** — likely a one-shot script in `ml/` that produces a JSON artifact per day-of-week per route. Reads `ml/cache/shared/locations_raw.csv`, clusters break times, writes archetypes.
3. **Build production matcher** — in `backend/fastapi/utils.py`, add `_match_shuttle_to_archetype(vehicle_id, morning_signature) -> archetype`. Plug into the `on_break` computation alongside the schedule-gap detector.
4. **Wire `on_break = schedule_gap OR archetype_predicted_break`** in `_build_locations_payload`.
5. **Add tests** — archetype extraction on a known day, matcher correctness on known signatures.
6. **Live-verify** with test sim + `DEV_TIME_SHIFT` to simulate a lunch-rotation day at different offsets through the break window.

## Notes on prior work

An earlier implementation of Hungarian-based schedule matching lived in `shared/schedules.py` (deleted on 2026-04-10 in `f81d191`). That code used a **full-day behavioral cost** (`1 − match_fraction on (route, stop, minute)`) and Redis-cached the result for 1 hour. We decided NOT to resurrect the file — correctness priorities don't match (their design was for end-of-day offline matching; ours needs per-cycle real-time). Concepts that inspire Phase 3:

- The idea of building (shuttles × archetypes) cost matrix and running linear_sum_assignment
- The schedule-flattening `[(name, [(time, stop)...])]` shape
- The Redis-cached-assignment pattern

These will be cited in commit messages / file headers when Phase 3 lands, for provenance.

## Pointers

- Phase 1 summary: `.planning/quick/260415-oeb-phase-3-break-detection-schedule-gap-det/260415-oeb-SUMMARY.md` *(typo in original dirname — actually `260415-oeb-phase-1-break-detection-schedule-gap-det`)*
- Simulation summary: `.planning/quick/260415-nm1-simulate-shuttle-break-behavior-in-test-/260415-nm1-SUMMARY.md`
- Historical data file: `ml/cache/shared/locations_raw.csv` (31 MB, 500K rows)
- Dev-time shift: `DEV_TIME_SHIFT=1 DEV_TARGET_HOUR=XX DEV_TARGET_MINUTE=XX` works on both backend (gated on `DEPLOY_MODE=development`) and test-server
- Break simulation live demo: see `260415-nm1-SUMMARY.md` for the DEV env-var combo to spawn the lunch-rotation demo

## Current branch state

```
b5e127f docs(quick-260415-oeb): Phase 1 break detection — schedule-gap + on_break flag
3aeb244 chore(backend): make dev_time _DEV_OFFSET opt-in via DEV_TIME_SHIFT=1
e04a490 feat(quick-260415-oeb): muted shuttle marker when on_break
ad58758 feat(quick-260415-oeb): schedule-gap detector + on_break flag in /api/locations
978141f docs(quick-260415-nm1): Simulate shuttle break behavior in test server
634b125 test(server): make dev_time OFFSET opt-in via DEV_TIME_SHIFT=1
ed48a1e feat(quick-260415-nm1): integrate break rotation + interrupt-aware LOOPING
47a5401 feat(quick-260415-nm1): per-day fleet sizing and rotation-driven ON_BREAK queuing
8de9fbd feat(quick-260415-nm1): upgrade ON_BREAK to drive off-route, sit, return on-route
... (base at d5db478 from 430-per-trip-eta-tracking)
```

`430-per-trip-eta-tracking` has been pushed to origin; PR to create at:
`https://github.com/wtg/shubble/compare/main...430-per-trip-eta-tracking`
(PR body drafted in session, paste from conversation history).
