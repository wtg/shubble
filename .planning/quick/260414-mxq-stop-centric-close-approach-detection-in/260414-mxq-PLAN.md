---
phase: 260414-mxq
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - backend/worker/data.py
  - tests/test_stop_centric_detection.py
autonomous: true
requirements:
  - QUICK-260414-mxq

must_haves:
  truths:
    - "Untagged GPS pings within 60 m of a stop's coordinate produce a last_arrival entry for that stop."
    - "HOUSTON_FIELD_HOUSE on NORTH surfaces a non-null `last_arrival` (with `passed_interpolated=False`) on at least one active vid/NORTH trip after a fresh worker cycle on live data."
    - "Duplicate-coordinate stops (e.g. STUDENT_UNION vs STUDENT_UNION_RETURN) are not cross-tagged by the new stop-centric pass — the existing tag-based / `resolve_duplicate_stops` path remains the sole source for those."
    - "Existing tag-based detections still surface (no regression to currently-working stops)."
    - "All four pre-existing regression suites still pass (`test_last_arrival_loop_scoping`, `test_dwelling_shuttle_trips`, `test_live_eta_scrub_past_stops`, `test_latest_close_approach`)."
  artifacts:
    - path: "backend/worker/data.py"
      provides: "Stop-centric close-approach pass merged with the existing tag-based pass in `_compute_vehicle_etas_and_arrivals`"
      contains: "stop-centric"
    - path: "tests/test_stop_centric_detection.py"
      provides: "3 unit tests: HFH drive-by repro, duplicate-coord isolation, tag-based regression"
      min_lines: 100
  key_links:
    - from: "backend/worker/data.py:_compute_vehicle_etas_and_arrivals (stop-centric block)"
      to: "last_arrivals_by_vehicle dict"
      via: "per-(vid,stop) max(timestamp) merge with tag-based result"
      pattern: "max\\(.*tag.*stop_centric"
    - from: "stop-centric pass"
      to: "_ROUTE_REMAP_CACHE (ml/data/stops.py)"
      via: "skip duplicate-coord stop names so we don't cross-tag STUDENT_UNION/_RETURN"
      pattern: "_build_route_remap_cache|_ROUTE_REMAP_CACHE"
---

<objective>
Add a stop-centric close-approach pass to `_compute_vehicle_etas_and_arrivals` so untagged GPS pings within 60 m of a stop coordinate still produce a `last_arrival`. The current tag-based filter (`stops_df = full_df.dropna(subset=['stop_name', ...])`) drops every ping the ML pipeline failed to tag — including all of HOUSTON_FIELD_HOUSE's fast drive-by pings, which never get within the 20 m `add_stops` threshold and whose `add_stops_from_segments` retro-tags get filtered out by the incremental cache (`cache_dataframe.py:379-384`). Result: HFH never displays a real `last_arrival` in the UI, only the interpolated one.

Purpose: Decouple last-arrival detection from the over-permissive ML stop_name tag while preserving the 60 m geometric invariant that already gates everything in this code path.

Output: One new vectorized loop in `data.py` (per route stop, haversine to every ping, keep within-60 m, max-timestamp per (vid, stop)) plus a 3-test guard file.
</objective>

<execution_context>
@$HOME/.claude/get-shit-done/workflows/execute-plan.md
@$HOME/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@backend/worker/data.py
@backend/cache_dataframe.py
@ml/data/stops.py
@shared/routes.json
@tests/test_latest_close_approach.py
@tests/test_last_arrival_loop_scoping.py

<interfaces>
<!-- Extracted from the codebase so the executor doesn't re-explore. -->

From backend/worker/data.py (existing function this plan modifies):
```python
async def _compute_vehicle_etas_and_arrivals(
    vehicle_ids: List[str],
    full_df: pd.DataFrame,
) -> Tuple[Dict[str, dict], Dict[str, Dict[str, str]]]:
    # returns (vehicle_stop_etas, last_arrivals_by_vehicle)
    # last_arrivals_by_vehicle: { vid_str: { stop_name: iso_timestamp_str } }
```

From ml/data/stops.py:
```python
_ROUTE_REMAP_CACHE: dict = {}
# Populated by _build_route_remap_cache():
#   { route_name: [(first_name, last_name, threshold_idx), ...] }
# `first_name` and `last_name` are the two stops sharing one coordinate
# (STUDENT_UNION / STUDENT_UNION_RETURN, etc.).

def _build_route_remap_cache() -> None: ...
```

From shared/stops.py:
```python
class Stops:
    routes_data: dict  # { route_name: { stop_name: { "COORDINATES": [lat, lon], "OFFSET": ..., ... }, "STOPS": [...], "ROUTES": [...] } }
```

From the existing block in `_compute_vehicle_etas_and_arrivals` we are extending (data.py:1159-1238):
- `CLOSE_APPROACH_M = 60.0`
- `tracked_vids_set = {str(v) for v in vehicle_ids}`
- haversine constant: `R = 6371000.0`
- result accumulator: `last_arrivals_by_vehicle: Dict[str, Dict[str, str]] = {}`
- writes: `vehicle_las[str(stop_key)] = ts_dt.isoformat()`
</interfaces>
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add stop-centric close-approach pass + 3 tests, merged with existing tag-based detection</name>
  <files>backend/worker/data.py, tests/test_stop_centric_detection.py</files>
  <behavior>
    Tests in tests/test_stop_centric_detection.py (all 3 must fail before the data.py change and pass after):

    1. `test_hfh_drive_by_with_untagged_pings_surfaces_last_arrival`
       - Build a 3-row DF for vid='vid1', route='NORTH', all `stop_name=None`, with lat/lon offsets giving distances ~29 m / ~40 m / ~10 m from `Stops.routes_data['NORTH']['HOUSTON_FIELD_HOUSE']['COORDINATES']` at timestamps t, t+5s, t+10s.
       - `polyline_idx=0`, `speed_kmh=20.0` (passes idle filter; 0 is valid since HFH doesn't have a duplicate-coord pair so the threshold_idx logic is not exercised).
       - Patch `predict_eta` (`AsyncMock(return_value={'vid1': t + timedelta(minutes=2)})`) like test_latest_close_approach.py does.
       - Call `await _compute_vehicle_etas_and_arrivals(['vid1'], df)`.
       - Assert `'HOUSTON_FIELD_HOUSE' in last_arrivals_by_vehicle['vid1']` AND
         `datetime.fromisoformat(...) == t + timedelta(seconds=10)` (the LATEST within-60 m ping).

    2. `test_duplicate_coord_stops_not_cross_tagged_by_stop_centric_pass`
       - Single ping at NORTH STUDENT_UNION coords (offset by ~5 m), `stop_name='STUDENT_UNION'`, `polyline_idx=0`, vid='vid2'.
       - The duplicate-coord pair on NORTH is (STUDENT_UNION, STUDENT_UNION_RETURN). The stop-centric pass MUST skip both members of that pair (rely on tag-based path for them).
       - After detection, `'STUDENT_UNION' in las['vid2']` (from tag-based path — the row is already tagged) AND `'STUDENT_UNION_RETURN' NOT in las['vid2']` (the stop-centric pass would have cross-tagged it because the coords are identical, so its absence proves the skip works).
       - Use `predict_eta` mock as above.

    3. `test_tag_based_detection_still_fires_when_stop_centric_pass_skips`
       - Construct a row tagged `stop_name='GEORGIAN'` within 20 m of NORTH GEORGIAN coords, vid='vid3'. GEORGIAN is NOT a duplicate-coord stop, so the stop-centric pass also covers it; this test asserts both paths converge on the same answer rather than dropping.
       - Assert `'GEORGIAN' in las['vid3']`.

    All tests use `Stops.routes_data['NORTH'][stop_name]['COORDINATES']` for ground-truth coords and the `_offset_coord(coord, meters_north)` helper pattern from `tests/test_latest_close_approach.py:58-64` (1 deg lat ≈ 111139 m).
  </behavior>
  <action>
    Write tests/test_stop_centric_detection.py FIRST (red phase), then make backend/worker/data.py changes (green phase). Single commit at the end is fine — tests must demonstrably fail without the data.py changes first.

    === backend/worker/data.py ===

    INSERT a new block IMMEDIATELY BEFORE the existing tag-based block at line 1160 (before `if 'stop_name' in full_df.columns:`). The new block writes its results into the SAME `last_arrivals_by_vehicle` dict using max-timestamp merge semantics, then the existing tag-based block runs and applies the same merge. Do NOT modify the existing tag-based block's logic — only change the dict-write line so it merges (max-ts) instead of overwrites.

    Sketch of the new block:

    ```python
    # ---- Stop-centric pass: detect close approaches even for pings the
    # ML stop_name tagger missed. The ML `add_stops` threshold is 20 m
    # (ml/data/stops.py:114), which a 20 mph test shuttle (~42 m per 5 s
    # poll) routinely overshoots — and the `add_stops_from_segments`
    # retro-tag often writes to the EARLIER endpoint of a segment, which
    # the incremental cache (backend/cache_dataframe.py:379-384) then
    # filters out as a context-row update. Net effect: HOUSTON_FIELD_HOUSE
    # and any other fast-drive-by stop never gets a real `last_arrival`.
    #
    # Fix: ignore the stop_name column for this pass entirely. For each
    # route stop, compute haversine distance from EVERY ping in full_df
    # to the stop's coord and keep pings within CLOSE_APPROACH_M (60 m).
    # Per (vehicle_id, stop), take the LATEST timestamp.
    #
    # We MUST skip duplicate-coordinate stop pairs here (e.g.
    # STUDENT_UNION + STUDENT_UNION_RETURN both at 42.730711, -73.676737).
    # A stop-centric pass would tag every ping at Union for BOTH names,
    # which silently breaks trip-completion logic that treats the
    # "_RETURN" detection as the loop end. The tag-based pass below
    # already handles duplicates correctly via `resolve_duplicate_stops`
    # (it uses polyline_idx to decide which name applies). Look up the
    # set of duplicates from the same cache that resolver uses, so a
    # routes.json change automatically propagates here.
    from ml.data.stops import _build_route_remap_cache, _ROUTE_REMAP_CACHE
    _build_route_remap_cache()
    duplicate_stop_names: set = set()
    for _route, entries in _ROUTE_REMAP_CACHE.items():
        for first_name, last_name, _threshold_idx in entries:
            duplicate_stop_names.add(first_name)
            duplicate_stop_names.add(last_name)

    if (
        not full_df.empty
        and 'latitude' in full_df.columns
        and 'longitude' in full_df.columns
        and 'vehicle_id' in full_df.columns
        and 'timestamp' in full_df.columns
    ):
        coord_df = full_df.dropna(subset=['latitude', 'longitude', 'vehicle_id', 'timestamp'])
        if not coord_df.empty:
            tracked_vids_set_sc = {str(v) for v in vehicle_ids}
            mask_tracked = coord_df['vehicle_id'].astype(str).isin(tracked_vids_set_sc)
            coord_df = coord_df[mask_tracked]

        if not coord_df.empty:
            ping_lats = coord_df['latitude'].astype(float).to_numpy()
            ping_lons = coord_df['longitude'].astype(float).to_numpy()
            ping_vids = coord_df['vehicle_id'].astype(str).to_numpy()
            ping_ts = pd.to_datetime(coord_df['timestamp'], utc=True).to_numpy()

            R = 6371000.0
            phi1 = np.radians(ping_lats)
            cos_phi1 = np.cos(phi1)

            # Build the unique set of (stop_name, lat, lon) across all routes,
            # skipping duplicate-coord stops. First occurrence wins (matches
            # the existing tag-based block's `if stop_name not in stop_coord_lookup`).
            seen_stops: set = set()
            for route_info in Stops.routes_data.values():
                if not isinstance(route_info, dict):
                    continue
                for stop_name, stop_data in route_info.items():
                    if not isinstance(stop_data, dict):
                        continue
                    if stop_name in duplicate_stop_names:
                        continue
                    if stop_name in seen_stops:
                        continue
                    coord = stop_data.get('COORDINATES')
                    if not coord:
                        continue
                    seen_stops.add(stop_name)
                    s_lat, s_lon = float(coord[0]), float(coord[1])

                    phi2 = np.radians(s_lat)
                    dphi = np.radians(s_lat - ping_lats)
                    dlam = np.radians(s_lon - ping_lons)
                    a = (
                        np.sin(dphi / 2.0) ** 2
                        + cos_phi1 * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
                    )
                    dists_m = 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
                    close = dists_m <= CLOSE_APPROACH_M
                    if not close.any():
                        continue

                    # Per-vehicle: keep the LATEST within-60m timestamp for this stop.
                    vids_close = ping_vids[close]
                    ts_close = ping_ts[close]
                    # Group by vid, take max ts. Small N (typically <50 close pings
                    # per stop per day), so a Python loop with dict.setdefault is
                    # cheaper than building a temporary DataFrame here.
                    per_vid_max: Dict[str, pd.Timestamp] = {}
                    for vid_v, ts_v in zip(vids_close, ts_close):
                        prev = per_vid_max.get(vid_v)
                        if prev is None or ts_v > prev:
                            per_vid_max[vid_v] = ts_v

                    for vid_v, ts_v in per_vid_max.items():
                        ts_dt = pd.Timestamp(ts_v).to_pydatetime()
                        if ts_dt.tzinfo is None:
                            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                        vehicle_las = last_arrivals_by_vehicle.setdefault(vid_v, {})
                        existing = vehicle_las.get(stop_name)
                        if existing is None:
                            vehicle_las[stop_name] = ts_dt.isoformat()
                        else:
                            # Merge: keep whichever timestamp is newer (this
                            # preserves the existing tag-based result if it
                            # happens to be later, and vice-versa).
                            existing_dt = datetime.fromisoformat(existing)
                            if ts_dt > existing_dt:
                                vehicle_las[stop_name] = ts_dt.isoformat()
    # ---- end stop-centric pass ----
    ```

    THEN modify the existing tag-based block's final write at lines 1237-1238 from a plain assignment to a max-ts merge so the order of the two passes doesn't matter. Replace:

    ```python
    vehicle_las = last_arrivals_by_vehicle.setdefault(vid_str, {})
    vehicle_las[str(stop_key)] = ts_dt.isoformat()
    ```

    with:

    ```python
    vehicle_las = last_arrivals_by_vehicle.setdefault(vid_str, {})
    existing = vehicle_las.get(str(stop_key))
    if existing is None or ts_dt > datetime.fromisoformat(existing):
        vehicle_las[str(stop_key)] = ts_dt.isoformat()
    ```

    === tests/test_stop_centric_detection.py ===

    Mirror the structure of tests/test_latest_close_approach.py:
      - same imports (`AsyncMock`, `patch`, `pd`, `pytest`, `_compute_vehicle_etas_and_arrivals`, `Stops`)
      - same `_mk_row` and `_offset_coord` helpers (copy them verbatim into this new file)
      - 3 `@pytest.mark.asyncio async def` tests as described in <behavior>

    The HFH test creates the 3 untagged rows with explicit `stop_name=None`, calls detection with the predict_eta patch, then asserts the latest within-60 m timestamp wins.

    The duplicate-coord test asserts the stop_centric pass DID NOT write STUDENT_UNION_RETURN — that's the bug it would introduce if we forgot the `duplicate_stop_names` skip. (We construct only ONE row tagged STUDENT_UNION; if stop_centric ran for STUDENT_UNION_RETURN it would also tag this row because the coords are identical.)

    The GEORGIAN test asserts both paths converge.

    Coords reference (verify against shared/routes.json before computing offsets):
      `Stops.routes_data['NORTH']['HOUSTON_FIELD_HOUSE']['COORDINATES']`
      `Stops.routes_data['NORTH']['STUDENT_UNION']['COORDINATES']`
      `Stops.routes_data['NORTH']['GEORGIAN']['COORDINATES']`

    === Constraints (do not violate) ===
    - Do NOT change `CLOSE_APPROACH_M` (60.0) — the geometric invariant.
    - Do NOT modify `tests/test_latest_close_approach.py`, `tests/test_last_arrival_loop_scoping.py`, `tests/test_dwelling_shuttle_trips.py`, or `tests/test_live_eta_scrub_past_stops.py`. They are the regression net.
    - Do NOT add new external dependencies. Use existing numpy/pandas/`Stops.routes_data`/`_ROUTE_REMAP_CACHE` only.
    - Do NOT change the function signature or return type of `_compute_vehicle_etas_and_arrivals`.
    - The new block must run BEFORE the existing tag-based block so the merge semantics are symmetric (either order works, but document the order chosen).
  </action>
  <verify>
    <automated>./.venv/Scripts/python.exe -m pytest tests/test_stop_centric_detection.py tests/test_last_arrival_loop_scoping.py tests/test_dwelling_shuttle_trips.py tests/test_live_eta_scrub_past_stops.py tests/test_latest_close_approach.py -v</automated>
  </verify>
  <done>
    - All 3 new tests pass.
    - All 4 regression suites (17 + 3 + 4 + 3 = 27 existing tests) still pass.
    - `_compute_vehicle_etas_and_arrivals` returns `last_arrivals_by_vehicle` containing HFH for the drive-by repro test (timestamp == latest within-60 m ping).
    - No new entry for STUDENT_UNION_RETURN in the duplicate-coord test.
  </done>
</task>

</tasks>

<verification>
1. Pytest gate (the <verify> command above) — all 30 tests green (3 new + 27 regression).
2. Live verification (post-commit, on running stack):
   a. Flush Redis cache so the worker rebuilds the processed dataframe from scratch:
      `./.venv/Scripts/python.exe -c "import asyncio; from redis import asyncio as aioredis; asyncio.run((lambda: (c:=aioredis.from_url('redis://localhost:32768')).flushdb())())"`
   b. Wait ~4 minutes (one full worker loop cycle).
   c. Inspect `/api/trips` for HFH on NORTH:
      `curl -s http://localhost:8000/api/trips | python -c "import json,sys; [print(f'{t[\"vehicle_id\"][-3:]}/{t[\"route\"]}/{t[\"status\"]:10s} HFH={t[\"stop_etas\"].get(\"HOUSTON_FIELD_HOUSE\",{}).get(\"last_arrival\")} interp={t[\"stop_etas\"].get(\"HOUSTON_FIELD_HOUSE\",{}).get(\"passed_interpolated\")}') for t in json.load(sys.stdin) if t.get('vehicle_id') and t['route']=='NORTH']"`
   d. Expected: at least one vid/NORTH row shows `HFH.last_arrival` non-null AND `interp=False`.
</verification>

<success_criteria>
- All 3 new unit tests in `tests/test_stop_centric_detection.py` pass.
- All 4 listed regression suites pass without modification.
- Live `/api/trips` shows at least one NORTH-route trip with `HOUSTON_FIELD_HOUSE.last_arrival` non-null AND `passed_interpolated=False` after a fresh worker cycle.
- No regression in `STUDENT_UNION` / `STUDENT_UNION_RETURN` trip-completion behavior (covered by duplicate-coord test + existing trip tests).
- Per-cycle worker latency increase from the new pass is bounded (~50 stops × ~5 k pings/day = ~250 k vectorized haversine calcs, <50 ms — no profiling task required, but flag if pytest noticeably slows).
</success_criteria>

<output>
After completion, create `.planning/quick/260414-mxq-stop-centric-close-approach-detection-in/260414-mxq-SUMMARY.md`
</output>
