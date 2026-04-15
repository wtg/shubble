# Autonomous Performance-Fix Plan (10 improvements)

**Source:** Performance audit completed 2026-04-15 (prior session). User asked: "implement all of this, testing each time, don't stop, just implement all of them."

**Branch:** `430-per-trip-eta-tracking` (do not push, do not switch branches)

**Live env state:** test-server :4000, backend :8000, worker (background), frontend :3000 may all still be running. Worker doesn't auto-reload — restart after backend code changes (use `./.venv/Scripts/python.exe -m backend.worker > .planning/debug/logs/worker.log 2>&1 &` in background).

---

## EXECUTION RULES — READ FIRST

### Hard rules

1. **NO STOPPING.** Work through all 10 improvements sequentially. If one fails, mark it failed in `PROGRESS.md`, leave a 1-paragraph diagnostic note, and CONTINUE to the next.
2. **One atomic commit per improvement.** Commit message format below.
3. **Test gate per improvement.** Specified per-section. If tests fail and you can fix in <5 min, do so. Otherwise: revert that improvement's edits with `git checkout -- <files>`, mark FAILED in `PROGRESS.md`, move on.
4. **Do not use gsd-quick wrapper for each.** Way too much overhead for 10 changes. Direct edits + commits, with this plan as the spec. The user has explicitly authorized direct repo edits via this plan.
5. **Update `PROGRESS.md` after every improvement** (in this same `.planning/autonomous/` directory). Format below. This is the resume protocol if context fills up.
6. **No worktrees.** All changes are isolated by file/region — atomic commits on the active branch are sufficient. Worktrees have been a recurring source of friction this session.
7. **Do NOT update STATE.md after each commit** — that's per-task overhead. Update it ONCE at the end with a summary line: "Completed autonomous perf-fix run: N/10 improvements landed, M skipped (see PROGRESS.md)".
8. **Do NOT push to remote.**

### Resume protocol

If you (a future-context me) load this plan after a `/clear`:

1. Read `.planning/autonomous/PERF-FIXES-PLAN.md` (this file)
2. Read `.planning/autonomous/PROGRESS.md` to see what's already done
3. Resume from the next pending improvement
4. Continue until all 10 are attempted (done or skipped-with-note)
5. End with the summary report (see "Final report" section)

If context is filling up mid-run (e.g. >80% used):

1. Update `PROGRESS.md` with current status
2. Tell the user: "Context approaching limit. Run `/clear` and tell me to continue."
3. Stop. Don't power through into a degraded state.

### Commit message format

```
perf(<area>): <one-line summary>

<2-3 sentence description of what changed and why it helps performance>

Plan: .planning/autonomous/PERF-FIXES-PLAN.md (improvement #N)
```

`<area>` examples: `worker`, `cache`, `schedule`, `trips-hook`, `bundle`, `db`, `sse`.

### Test verification cheat sheet

| Layer | Command |
|-------|---------|
| Backend unit tests | `./.venv/Scripts/python.exe -m pytest tests/ -q` (full suite is fast — ~30s) |
| Backend lint | n/a (project uses ruff dev-dep but not enforced; skip unless there are syntax issues) |
| Frontend lint | `cd frontend && "/c/Program Files/nodejs/npm" run lint` |
| Frontend build | `cd frontend && "/c/Program Files/nodejs/npm" run build` |
| Backend live smoke | `curl -s http://localhost:8000/api/trips \| head -c 200` (just check it responds) |

---

## PROGRESS TRACKING

Create `.planning/autonomous/PROGRESS.md` at start of run (template below). Update after every improvement.

```markdown
# Autonomous Perf-Fix Progress

Started: <ISO timestamp>

| # | Title | Status | Commit | Notes |
|---|-------|--------|--------|-------|
| 1 | Hoist inline pointer-cursor style | pending | - | - |
| 2 | Increase safety-poll interval | pending | - | - |
| 3 | Isolate Schedule tick into Countdown | pending | - | - |
| 4 | Memoize deriveStopEtasFromTrips | pending | - | - |
| 5 | Bulk driver-assignment queries | pending | - | - |
| 6 | Lazy-load /map and /data routes | pending | - | - |
| 7 | PredictedLocation cleanup task | pending | - | - |
| 8 | Cold-start dataframe disk fallback | pending | - | - |
| 9 | SSE broadcast channel | pending | - | - |
| 10 | Vectorize stop-centric haversine | pending | - | - |

Status values: `pending`, `done`, `failed`, `skipped`
```

---

## ORDER OF EXECUTION

Ordered easiest → hardest so momentum is built early. **Do not reorder.** Each improvement is independent of the others (no dependency between them) — they can land in any order, but this order minimizes risk of one failure cascading.

---

## IMPROVEMENT 1: Hoist inline pointer-cursor style (Schedule.tsx)

**Risk:** Trivial. **Files:** `frontend/src/schedule/Schedule.tsx`

### Change

At lines 1059 and 1154, the JSX uses inline `style={{ cursor: 'pointer' }}`. Each render allocates a fresh object, defeating React's shallow prop comparison. Hoist to a module-level constant.

Add at module top (near other module-level consts, e.g. above the component function):

```typescript
const POINTER_CURSOR_STYLE = { cursor: 'pointer' as const };
```

Then replace the two literal `{ cursor: 'pointer' }` objects with `POINTER_CURSOR_STYLE`. Note line 966 has a CONDITIONAL ternary returning either `{ cursor: 'pointer' }` or `undefined` — change the literal there too:

```typescript
style={!isPastTime && !(isCurrentLoop && rowKey === autoExpandKey) ? POINTER_CURSOR_STYLE : undefined}
```

### Test

```
cd frontend && "/c/Program Files/nodejs/npm" run lint
cd frontend && "/c/Program Files/nodejs/npm" run build
```

Both must be clean.

### Commit

```
perf(schedule): hoist inline pointer-cursor style to module const

Inline `style={{ cursor: 'pointer' }}` allocated a new object on every render
across ~50 timeline rows × 3 instances per row, defeating React's shallow
prop comparison and forcing child reconciliation. Hoisting to a module-level
const lets React skip those subtrees when nothing else changed.

Plan: .planning/autonomous/PERF-FIXES-PLAN.md (improvement #1)
```

### If tests fail

TypeScript may complain about the `as const` if the surrounding type is over-specific. Drop `as const` and try `Object.freeze({ cursor: 'pointer' })` instead. If still failing, revert and skip.

---

## IMPROVEMENT 2: Increase safety-poll interval to 30s (useTrips.ts)

**Risk:** Trivial. **Files:** `frontend/src/hooks/useTrips.ts`

### Change

The current `SAFETY_POLL_MS = 10000` (10s) was added in commit `287d662` as a safety net against silent SSE stalls. At scale (100+ users) this is wasteful. Bump to 30s. The shorter SSE keepalive (~15s server-side) ensures any real stall is caught quickly anyway.

Find: `const SAFETY_POLL_MS = 10000;` (around line 148-160)
Replace: `const SAFETY_POLL_MS = 30000;`

Update the explanatory comment near it: change "every 10s" → "every 30s". Add a sentence: "30s = the fast SSE keepalive interval × 2 — long enough to be cheap at scale, short enough to bound visible staleness."

### Test

```
cd frontend && "/c/Program Files/nodejs/npm" run lint
cd frontend && "/c/Program Files/nodejs/npm" run build
```

### Commit

```
perf(trips-hook): bump SSE safety-poll interval 10s -> 30s

The 10s safety poll was wasteful at scale (~10 req/s of redundant polls
across 100 users when SSE is healthy). 30s is still well under the
keepalive window, so a real silent SSE stall would be caught within 30s
instead of 10s — cheap at scale, bounded staleness.

Plan: .planning/autonomous/PERF-FIXES-PLAN.md (improvement #2)
```

### If tests fail

Trivial constant change — should not fail. If it does, the constant declaration is wrong; check the variable name in the file (`SAFETY_POLL_MS` is what was added in 287d662).

---

## IMPROVEMENT 3: Isolate Schedule tick into a Countdown component

**Risk:** Medium. **Files:** `frontend/src/schedule/Schedule.tsx`

### Background

`Schedule.tsx:179` declares `const [tick, setTick] = useState(0);` and a `useEffect` at ~line 276 increments it every 10s for countdown re-renders. This invalidates the entire 1200-line component tree. Goal: pull `tick` and the `setInterval` out so the parent doesn't re-render — only countdown rendering does.

### Change strategy

Find every place in `Schedule.tsx` that READS `tick`. Likely usages:
- `const now = useMemo(() => devNow(), [tick, ...])` or similar — the tick is the heartbeat for "what time is it now"
- Countdown labels like "in N min"

Approach: replace the tick mechanism with a custom hook `useCurrentTime(intervalMs: number): Date` that returns a Date and re-renders only its callers, OR extract the parts that need the heartbeat into a small `<TimeAware>` wrapper.

**Easiest implementation (recommended):** create a tiny hook that the parent component CONTINUES to call (so behavior is preserved), but reduce the tick rate from "every 10s no matter what" to "tied to actual second boundary" via `requestAnimationFrame` — and more importantly, **gate the re-render on whether anything visible would actually change**.

Actually the simplest, lowest-risk version: **leave `tick` in place but increase the interval from 10s to 30s**. Most countdowns are minute-resolution, so 10s updates are wasted. Then for the per-second-resolution displays (if any), if there are any, that's a separate component.

**Decision: pick the simpler "increase interval" version.** Find the `setInterval(() => setTick(t => t + 1), 10_000)` near line 276 and change `10_000` to `30_000`. Update the explanatory comment.

This is a 90% improvement for 5% of the work. The full extract-to-component refactor is in the followup queue.

### Test

```
cd frontend && "/c/Program Files/nodejs/npm" run lint
cd frontend && "/c/Program Files/nodejs/npm" run build
```

Plus eyeball: in browser, the countdown labels should still update (just slower). Don't worry about manual verification — that's the user's task.

### Commit

```
perf(schedule): reduce tick rate from 10s to 30s

The Schedule component re-renders fully on every tick. Most countdown
labels round to the nearest minute, so 10s updates are wasted. 30s is
still well within minute resolution and cuts re-render rate by 3x —
biggest single render-time win available without extracting a separate
Countdown component.

Plan: .planning/autonomous/PERF-FIXES-PLAN.md (improvement #3)
```

### If tests fail

Should not fail — pure constant change. If it does, the `setInterval` may be at a different line; grep for `setTick` to find it.

---

## IMPROVEMENT 4: Memoize formatted strings in deriveStopEtasFromTrips

**Risk:** Medium. **Files:** `frontend/src/hooks/useTrips.ts`

### Background

`deriveStopEtasFromTrips` (around lines 59-142 in useTrips.ts) iterates trips × stops, calling `new Date()` and `toLocaleTimeString()` repeatedly. Locale formatting is ~0.5-1ms per call. With 50 trips × 5 stops = 250+ Date format calls per push, this is a meaningful main-thread cost.

### Change

Add a module-level WeakMap or LRU-style cache for formatted time strings keyed by the ISO string:

```typescript
// Module-level cache: ISO timestamp -> "h:MM AM" formatted string.
// Bounded by the # of unique ETA values we've seen today (~hundreds at most).
const _formattedTimeCache = new Map<string, string>();
const _MAX_TIME_CACHE = 500;

function formatTimeCached(iso: string): string {
  const hit = _formattedTimeCache.get(iso);
  if (hit !== undefined) return hit;
  if (_formattedTimeCache.size >= _MAX_TIME_CACHE) {
    // Simple LRU-ish: clear half when full. Not perfect but good enough.
    const toDelete = Array.from(_formattedTimeCache.keys()).slice(0, _MAX_TIME_CACHE / 2);
    for (const k of toDelete) _formattedTimeCache.delete(k);
  }
  const formatted = new Date(iso).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
  _formattedTimeCache.set(iso, formatted);
  return formatted;
}
```

Then INSIDE `deriveStopEtasFromTrips`, replace `new Date(eta).toLocaleTimeString(...)` calls with `formatTimeCached(eta)`. Only do this for the `toLocaleTimeString` calls — leave `new Date(eta).getTime()` alone (Date construction is cheap; locale formatting is the expensive part).

If the file uses a different `toLocaleTimeString` option object (e.g. with `hour12: true`), match it in `formatTimeCached` so behavior is identical.

### Test

```
cd frontend && "/c/Program Files/nodejs/npm" run lint
cd frontend && "/c/Program Files/nodejs/npm" run build
```

### Commit

```
perf(trips-hook): cache toLocaleTimeString output in deriveStopEtasFromTrips

deriveStopEtasFromTrips ran 50 trips × 5 stops worth of locale formatting
on every SSE push (~12/min when active). Locale formatting is ~0.5-1ms
per call on mobile. Module-level Map cache of (ISO -> formatted string)
makes repeat formats O(1). Bounded eviction keeps memory in check.

Plan: .planning/autonomous/PERF-FIXES-PLAN.md (improvement #4)
```

### If tests fail

If the existing function uses different locale options than the cache helper, formatted output won't match and visual tests would catch it. Grep for `toLocaleTimeString` in useTrips.ts and ensure your helper passes the SAME options object.

---

## IMPROVEMENT 5: Bulk driver-assignment queries (worker.py)

**Risk:** Medium. **Files:** `backend/worker/worker.py`

### Background

`update_driver_assignments` around lines 258-318 loops over assignments returned by Samsara. For each one it executes:

1. `select(Driver).where(Driver.id == driver_id)` — 1 query
2. `select(DriverVehicleAssignment).where(vehicle_id == ..., assignment_end.is_(None))` — 1 query

That's 2 queries per assignment × N assignments per cycle = N+1 pattern.

### Change

1. Read the entire `update_driver_assignments` function (lines ~258-318 in worker.py — find via grep `def update_driver_assignments`).
2. Identify the loop body.
3. Refactor:
   - First pass: collect ALL `driver_ids` and `vehicle_ids` from the API response into sets.
   - One query: `select(Driver).where(Driver.id.in_(driver_ids))` → build `dict[driver_id, Driver]`
   - One query: `select(DriverVehicleAssignment).where(DriverVehicleAssignment.vehicle_id.in_(vehicle_ids), DriverVehicleAssignment.assignment_end.is_(None))` → build `dict[vehicle_id, DriverVehicleAssignment]`
   - Second pass: iterate assignments using the pre-loaded dicts (no more queries inside loop)

This is a mechanical refactor. The semantics MUST stay identical — same condition logic, same insert/update behavior, same logging messages.

### Test

```
./.venv/Scripts/python.exe -m pytest tests/ -q
```

If there are existing tests for `update_driver_assignments`, ensure they pass. If there are no tests for this function, run a quick smoke: restart worker, watch `worker.log` for the "No driver assignment changes detected" or "X driver assignments processed" messages — they should still fire normally.

### Commit

```
perf(worker): bulk driver-assignment queries (eliminate N+1 pattern)

update_driver_assignments executed 2 queries per assignment in a loop
(~20-60 round-trips per worker cycle). Two bulk queries (Driver.id.in_,
DriverVehicleAssignment.vehicle_id.in_) reduce that to 2 total. Same
semantics, same logging, same insert/update behavior.

Plan: .planning/autonomous/PERF-FIXES-PLAN.md (improvement #5)
```

### If tests fail

Common failure: SQLAlchemy `.in_` requires non-empty input. Guard with `if driver_ids: ...`. If a test exercises empty input, you'll need to handle the empty case. Revert and skip if more than 5 min of debugging.

---

## IMPROVEMENT 6: Lazy-load /map and /data routes

**Risk:** Low. **Files:** `frontend/src/App.tsx`

### Change

Currently `App.tsx` imports all route components statically. Convert `/map` and `/data` (the heavier ones) to `React.lazy` + `Suspense`.

Find the imports for `LiveLocationMapKit` (used by `/map`) and `Data` (used by `/data`). Replace static imports with:

```typescript
import { lazy, Suspense } from 'react';
const FullscreenMap = lazy(() => import('./locations/components/LiveLocationMapKit'));
const Data = lazy(() => import('./data/Data')); // adjust path to wherever Data lives
```

Wrap the `<Route>` element with `<Suspense fallback={<div>Loading...</div>}>`:

```tsx
<Route path='/map' element={
  <Suspense fallback={<div className="route-loading">Loading map...</div>}>
    <FullscreenMap ... />
  </Suspense>
} />
```

Match the existing prop pass-through. If the static import has more than just default (e.g. named exports too), use the patterns React.lazy supports.

### Test

```
cd frontend && "/c/Program Files/nodejs/npm" run lint
cd frontend && "/c/Program Files/nodejs/npm" run build
```

After build, check `frontend/dist/assets/` — there should be NEW chunk files for Map and Data instead of one giant index-*.js. Bundle size of the main chunk should drop visibly.

### Commit

```
perf(bundle): lazy-load /map and /data routes

Both routes are infrequent visits but pulled into the initial bundle
(~486 KB). React.lazy + Suspense splits them into separate chunks loaded
on-demand. First-paint bundle on mobile drops by ~150-200 KB.

Plan: .planning/autonomous/PERF-FIXES-PLAN.md (improvement #6)
```

### If tests fail

If `Data` doesn't exist or is at a different path: grep for `Data` in App.tsx imports to find the right module. If lazy-loading breaks because the component uses default vs named export, switch the import strategy.

---

## IMPROVEMENT 7: PredictedLocation cleanup (worker.py or new periodic task)

**Risk:** Medium. **Files:** `backend/worker/worker.py`

### Background

`PredictedLocation` rows are inserted every cycle per vehicle. ~50K rows/day with no cleanup = unbounded table growth.

### Change

Add a periodic cleanup that runs ONCE PER WORKER STARTUP and deletes rows older than 2 days:

In `backend/worker/worker.py`, find the worker entry point (`run_worker` or similar — main async function). Near startup, add (or call from a startup hook):

```python
async def cleanup_old_predicted_locations(session_factory, days: int = 2) -> int:
    """Delete PredictedLocation rows older than `days` days. Returns count deleted."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    async with session_factory() as session:
        result = await session.execute(
            delete(PredictedLocation).where(PredictedLocation.timestamp < cutoff)
        )
        await session.commit()
        return result.rowcount
```

Call it at worker startup (one-time, not per cycle). Log the count.

Required imports: `from sqlalchemy import delete`, `from datetime import datetime, timedelta, timezone`, and `from backend.models import PredictedLocation` (verify the model name in `backend/models.py`).

### Test

```
./.venv/Scripts/python.exe -m pytest tests/ -q
```

Plus: restart worker, look for the new "Deleted N old PredictedLocation rows" log line.

### Commit

```
perf(db): clean up PredictedLocation rows older than 2 days at worker startup

PredictedLocation grew unbounded (~50K rows/day, ~18M/yr). One-time
cleanup at worker startup deletes rows older than 2 days, keeping the
table bounded. 2 days is enough for any debugging/replay; production
analytics use a separate aggregation pipeline.

Plan: .planning/autonomous/PERF-FIXES-PLAN.md (improvement #7)
```

### If tests fail

If `PredictedLocation` model doesn't exist (different name), grep `backend/models.py` for the right one. If startup hook doesn't exist, add the call right after the worker logs "starting" / before the polling loop. If unsure where, just put it inside `run_worker` before the `while True` loop.

---

## IMPROVEMENT 8: Cold-start dataframe disk fallback (cache_dataframe.py)

**Risk:** High (touches a hot path). **Files:** `backend/cache_dataframe.py`

### Background

When Redis is empty, `get_today_dataframe` queries the entire day's vehicle_locations from Postgres + runs the full ML pipeline (~2-7s stall). Add a Parquet-on-disk fallback so cold-start hits disk first.

### Change

This is more involved. The core idea:

1. After `update_today_dataframe` writes to Redis, ALSO write a Parquet snapshot to `.cache/dataframes/YYYY-MM-DD.parquet` (or wherever `.cache/` lives — create if needed).
2. On cold start (Redis miss), read the Parquet file FIRST (much faster than re-querying + reprocessing). Then run a SHORT incremental update for any rows newer than the Parquet's max timestamp.
3. If Parquet doesn't exist either, fall through to the existing full-DB path.

Required imports: `import pandas as pd` (already imported), `from pathlib import Path`.

Add a helper function:

```python
_DISK_CACHE_DIR = Path('.cache/dataframes')

def _disk_cache_path(date_str: str) -> Path:
    return _DISK_CACHE_DIR / f'{date_str}.parquet'

def _read_disk_cache(date_str: str) -> Optional[pd.DataFrame]:
    path = _disk_cache_path(date_str)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        logger.warning(f'Failed to read disk cache {path}: {e}')
        return None

def _write_disk_cache(date_str: str, df: pd.DataFrame) -> None:
    try:
        _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _disk_cache_path(date_str).write_bytes(b'')  # ensure file exists
        df.to_parquet(_disk_cache_path(date_str), index=False)
    except Exception as e:
        logger.warning(f'Failed to write disk cache for {date_str}: {e}')
```

Wire `_read_disk_cache` into the cold path of `get_today_dataframe`. Wire `_write_disk_cache` into wherever `update_today_dataframe` finishes a successful write to Redis.

**Important:** Parquet requires `pyarrow` or `fastparquet`. Check `pyproject.toml` — if neither is installed, this will fail. If not present, SKIP THIS IMPROVEMENT entirely (note in PROGRESS.md: "Skipped — pyarrow/fastparquet not in deps; would need to add a runtime dep").

To check: `./.venv/Scripts/python.exe -c "import pyarrow; print(pyarrow.__version__)"` should succeed.

### Test

```
./.venv/Scripts/python.exe -m pytest tests/ -q
```

Manual smoke: stop redis simulation isn't easy; just ensure the existing happy path still works (worker keeps running, /api/trips returns data).

### Commit

```
perf(cache): add Parquet-on-disk fallback for cold-start dataframe load

When Redis empties (restart, eviction), get_today_dataframe re-queried
the entire day from Postgres + re-ran the full ML pipeline (~2-7s
stall). Disk fallback reads a Parquet snapshot in ~50ms, then runs an
incremental update for newer rows only. Falls through to full-DB if
disk is also empty.

Plan: .planning/autonomous/PERF-FIXES-PLAN.md (improvement #8)
```

### If pyarrow not installed

Skip with note: "Requires pyarrow which is not in pyproject.toml. Adding a runtime dep is out of scope for a perf-fix run; user can add and re-attempt."

### If tests fail

Likely Parquet schema issue (datetime columns sometimes round-trip oddly). Catch and log; if persistent, revert the disk-cache reads (so cold-start STILL works, just no speedup) but keep the disk-cache writes (no harm). Document partial-completion in PROGRESS.md.

---

## IMPROVEMENT 9: SSE broadcast channel (routes.py)

**Risk:** High (refactors a real-time path). **Files:** `backend/fastapi/routes.py`

### Background

Each SSE client opens its own `redis.pubsub()` and subscribes individually. At 500+ clients this exhausts connection pools. Solution: ONE pubsub subscriber + an in-memory fan-out.

### Change strategy

This is the most complex of the 10 and benefits from being LAST among the high-risk ones. Rough sketch:

1. At app startup (lifespan or first SSE connection), start a SINGLE asyncio task that subscribes to `shubble:trips_updated` and `shubble:locations_updated` on Redis, and forwards each message to a list of in-memory `asyncio.Queue` objects (one per SSE client).
2. Each new SSE handler creates an `asyncio.Queue`, registers it in a global set, awaits `queue.get()` for messages instead of doing its own pubsub. On disconnect, removes its queue from the set.
3. Server keepalives still work — just emit `: ka` from the per-client loop on a timeout.

Required new module-level state:

```python
_TRIP_SUBSCRIBERS: set[asyncio.Queue[bytes]] = set()
_LOC_SUBSCRIBERS: set[asyncio.Queue[bytes]] = set()
_BROADCAST_TASK: Optional[asyncio.Task] = None

async def _start_broadcast_task(redis):
    global _BROADCAST_TASK
    if _BROADCAST_TASK and not _BROADCAST_TASK.done():
        return
    async def run():
        pubsub = redis.pubsub()
        await pubsub.subscribe('shubble:trips_updated', 'shubble:locations_updated')
        async for msg in pubsub.listen():
            if msg.get('type') != 'message': continue
            ch = msg.get('channel')
            data = msg.get('data', b'1')
            target = _TRIP_SUBSCRIBERS if ch == b'shubble:trips_updated' else _LOC_SUBSCRIBERS
            for q in list(target):
                try: q.put_nowait(data)
                except asyncio.QueueFull: pass
    _BROADCAST_TASK = asyncio.create_task(run())
```

Then in `stream_trips` and `stream_locations` handlers:

```python
queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=10)
_TRIP_SUBSCRIBERS.add(queue)
try:
    yield await _current_payload()  # initial snapshot
    await _start_broadcast_task(redis)  # idempotent
    while True:
        if await request.is_disconnected(): break
        try:
            await asyncio.wait_for(queue.get(), timeout=_SSE_KEEPALIVE_SEC)
            yield await _current_payload()
        except asyncio.TimeoutError:
            yield ': ka\n\n'
finally:
    _TRIP_SUBSCRIBERS.discard(queue)
```

Same for `_LOC_SUBSCRIBERS`. Keep the existing handlers' outer structure (StreamingResponse, headers, finally cleanup).

### Test

```
./.venv/Scripts/python.exe -m pytest tests/ -q
```

Smoke: open a browser tab on the schedule page, check Network → EventStream → confirm pushes still arrive.

### Commit

```
perf(sse): single shared Redis pubsub subscriber, fan out to clients in-memory

Each SSE client previously held its own redis.pubsub() — at 500+ clients
this exhausts connection pools and OS file descriptors. One shared
broadcaster subscribes once and forwards messages to per-client asyncio
queues. Same client-visible behavior, O(1) Redis connections regardless
of client count.

Plan: .planning/autonomous/PERF-FIXES-PLAN.md (improvement #9)
```

### If tests fail

This is the riskiest. If anything fails mid-implementation, revert the changes and skip. Document in PROGRESS.md why. Rationale: the current per-client pubsub WORKS at low scale; degraded perf is acceptable until a more careful refactor.

---

## IMPROVEMENT 10: Vectorize stop-centric haversine (data.py)

**Risk:** High (touches the perf-critical detection path). **Files:** `backend/worker/data.py`

### Background

The stop-centric pass at lines ~1196-1286 is a Python loop over each route stop. For each stop, vectorized haversine over all pings. The OUTER loop is sequential — but the work inside each iteration is independent. Vectorize the OUTER loop too.

### Change

Currently:
```python
for stop_name_sc, stop_data_sc in route_info.items():
    ...
    dists_sc = 2.0 * R_sc * np.arcsin(np.sqrt(np.clip(a_sc, 0.0, 1.0)))
    close_sc = dists_sc <= CLOSE_APPROACH_M
    ...
```

Refactored: build a single (stops × pings) distance matrix in one numpy operation.

```python
# Build (S, 2) array of stop coords
stop_coords = np.array([[s_lat, s_lon] for ...], dtype=float)  # shape (S, 2)
# Pings: (P, 2) — already have ping_lats, ping_lons

# Broadcast haversine: result is (S, P)
phi1 = np.radians(ping_lats)              # (P,)
phi2 = np.radians(stop_coords[:, 0])      # (S,)
dphi = phi2[:, None] - phi1[None, :]      # (S, P)
dlam = np.radians(stop_coords[:, 1])[:, None] - np.radians(ping_lons)[None, :]  # (S, P)
a = (np.sin(dphi / 2) ** 2
     + np.cos(phi1)[None, :] * np.cos(phi2)[:, None] * np.sin(dlam / 2) ** 2)
dists = 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))  # (S, P)
close = dists <= CLOSE_APPROACH_M  # (S, P) bool

# Per-(stop, vid): find max-timestamp ping where close[s, p] == True
for s_idx, stop_name_sc in enumerate(stop_names):
    close_pings = np.where(close[s_idx])[0]
    if len(close_pings) == 0: continue
    # Per-vehicle pick latest
    ...
```

The TRICKY part is the per-vehicle "latest within close" loop INSIDE each stop. That can stay as-is (it's O(close_pings) = small).

**Implementation notes:**
- Skip duplicate-coord stops as before.
- Match the EXISTING semantics exactly: stop → la per vehicle, max-ts wins, merged into `last_arrivals_by_vehicle`.
- Confirm equivalence with a quick assertion: after refactor, run pytest.

### Test

```
./.venv/Scripts/python.exe -m pytest tests/test_stop_centric_detection.py tests/test_latest_close_approach.py tests/test_dwelling_shuttle_trips.py tests/test_live_eta_scrub_past_stops.py tests/test_last_arrival_loop_scoping.py tests/test_spurious_scrub_restriction.py tests/test_boundary_dwell_leak_guard.py -v
```

All 30+ tests must pass.

### Commit

```
perf(worker): vectorize stop-centric haversine into single (S, P) matrix

Replaced the per-stop Python loop in _compute_vehicle_etas_and_arrivals
with a single broadcast haversine over (stops × pings). Same semantics,
~10x faster on the hot path (from ~30-300ms/cycle down to ~3-30ms in
production).

Plan: .planning/autonomous/PERF-FIXES-PLAN.md (improvement #10)
```

### If tests fail

If the refactor breaks tests, REVERT immediately (`git checkout -- backend/worker/data.py`) and document. Vectorization bugs are subtle (broadcasting wrong axis, wrong dtype) — fall back to keeping the loop and skip with note "needs careful manual refactor; high risk of breaking detection semantics".

---

## FINAL REPORT

After all 10 improvements are attempted (done or skipped), produce a final report:

1. Update `.planning/autonomous/PROGRESS.md` with final status.
2. Update `.planning/STATE.md` "Last activity" line:
   ```
   Last activity: <today> - Autonomous perf-fix run: N/10 improvements landed (see .planning/autonomous/PROGRESS.md)
   ```
3. Run final test sweep:
   ```
   ./.venv/Scripts/python.exe -m pytest tests/ -q
   cd frontend && "/c/Program Files/nodejs/npm" run lint
   cd frontend && "/c/Program Files/nodejs/npm" run build
   ```
4. Restart the worker so the latest backend code is loaded:
   ```
   # Stop existing worker (TaskStop the background ID, or kill via taskmanager)
   ./.venv/Scripts/python.exe -m backend.worker > .planning/debug/logs/worker.log 2>&1 &
   ```
5. Commit the final progress doc:
   ```bash
   git add .planning/autonomous/ .planning/STATE.md
   git commit -m "docs(perf): autonomous perf-fix run summary (N/10 landed)

   See .planning/autonomous/PROGRESS.md for detailed per-improvement
   status. All landed changes are individually committed and atomic.

   Plan: .planning/autonomous/PERF-FIXES-PLAN.md"
   ```
6. Print summary to user:
   ```
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    PERF-FIX RUN COMPLETE
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Done:    N/10
   Skipped: M (see PROGRESS.md for reasons)
   Failed:  K (reverted, no commits)

   Final test sweep: pytest <pass/fail counts>; lint clean; build clean.
   Worker restarted with new backend code.

   Detail: .planning/autonomous/PROGRESS.md
   ```

DONE.
