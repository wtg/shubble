# Autonomous Perf-Fix Progress

Started: 2026-04-15

| # | Title | Status | Commit | Notes |
|---|-------|--------|--------|-------|
| 1 | Hoist inline pointer-cursor style | done | 618cf37 | lint+build clean |
| 2 | Increase safety-poll interval | done | 28c675c | lint+build clean |
| 3 | Isolate Schedule tick into Countdown | done | 5f2a3a9 | simpler variant: 10s->30s per plan |
| 4 | Memoize deriveStopEtasFromTrips | done | e82ed57 | module-level Map cache |
| 5 | Bulk driver-assignment queries | done | f7e1375 | 109 pytest pass; 1 pre-existing unrelated failure |
| 6 | Lazy-load /map and /data routes | done | 97991d3 | /data split OK; /map blocked by LiveLocation static import (noted) |
| 7 | PredictedLocation cleanup task | done | ec53d8f | 89 pytest pass |
| 8 | Cold-start dataframe disk fallback | skipped | - | No pyarrow/fastparquet in deps. Per plan's skip criteria — adding runtime dep out of scope. |
| 9 | SSE broadcast channel | skipped | - | Refactor broke 2 existing test_sse_latency tests (mocks use get_message; broadcaster uses listen). Reverted per plan's revert-on-test-fail rule. Needs test fixture update + module-state teardown before re-attempt. |
| 10 | Vectorize stop-centric haversine | done | 25445dd | 29 detection tests pass (stop_centric + latest_close + dwelling + live_eta_scrub + last_arrival + spurious_scrub + boundary_dwell) |
