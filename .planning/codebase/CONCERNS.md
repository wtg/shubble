# Codebase Concerns

**Analysis Date:** 2026-04-05

## Tech Debt

**Type Safety Issues in Frontend:**
- Issue: TypeScript assertions using `as unknown as` type casting suppress type checking
- Files: `frontend/src/locations/LiveLocation.tsx` (line 16), `frontend/src/schedule/Schedule.tsx` (line 11-12)
- Impact: Potential runtime errors from mismatched data shapes not caught at compile time; brittleness when JSON data structures change
- Fix approach: Create proper TypeScript type definitions from actual JSON schema; use zod or similar for validation at boundaries

**Incomplete Type Definition:**
- Issue: TODO comment indicates unresolved type safety in location data handling
- Files: `frontend/src/locations/LiveLocation.tsx` (line 15)
- Impact: Type checker cannot validate route data structures
- Fix approach: Define complete types for aggregatedSchedule and routeData; consider code generation from backend schema

**Uncached Database Queries for Schedule Data:**
- Issue: `/api/routes` and `/api/schedule` endpoints execute database joins without caching; these are frequently accessed, read-only data
- Files: `backend/fastapi/routes.py` (lines 408-459, 461-508)
- Impact: Unnecessary database load on every schedule view request; potential latency on high-traffic periods
- Fix approach: Add 3600s TTL cache decorator to both endpoints; invalidate only on admin schedule updates

**Custom Cache Implementation Without Distributed Lock Guarantees:**
- Issue: Custom Redis cache (replacing fastapi-cache2) uses `lock_timeout` parameter but implementation may have race conditions
- Files: `backend/cache.py`, `backend/fastapi/routes.py` (cache decorators with `lock_timeout` parameters)
- Impact: Concurrent requests may bypass cache during lock timeout (set to 5.0s in some endpoints), causing thundering herd on cache misses
- Fix approach: Implement proper distributed locking with Redis SETNX + TTL; or reduce lock_timeout to 0.0 if stale data acceptable

**Missing Geofence Data Synchronization:**
- Issue: Geofence events are webhook-only; no periodic sync if webhooks are missed or fail silently
- Files: `backend/fastapi/routes.py` (POST /api/webhook, lines 200-332)
- Impact: Stale geofence state if Samsara webhook delivery fails; vehicles may be tracked when not in service area
- Fix approach: Add periodic geofence sync endpoint; implement webhook delivery verification; add webhook retry logic

---

## Known Bugs

**TypeError in Webhook Processing on Missing Data:**
- Symptoms: Webhook returns 500 error and logs uncaught exception if vehicle data missing from geofence event
- Files: `backend/fastapi/routes.py` (lines 243-283)
- Trigger: Geofence event with empty vehicle data or missing condition details
- Current behavior: Code continues loop (line 269 `continue`) but line 275 assumes vertices exist without safe access
- Workaround: Ensure Samsara sends complete geofence polygon data in all events
- Fix approach: Add defensive checks before accessing nested dictionaries; validate geofence polygon structure before use

**Potential Index Out of Bounds in LiveLocation:**
- Symptoms: Frontend may crash if filteredRouteData is empty but code attempts access
- Files: `frontend/src/schedule/Schedule.tsx` (line 78, accessing `Object.keys(aggregatedSchedule[selectedDay])[0]` without length check)
- Trigger: Schedule data missing for current day (Sunday = 0, etc.)
- Workaround: Manually select a day with schedule data
- Fix approach: Add length check before array access; provide fallback empty state or error message

**Race Condition in Driver Assignment Updates:**
- Symptoms: Multiple concurrent updates may skip assignment changes or create duplicate records
- Files: `backend/worker/worker.py` (lines 168-297)
- Trigger: Two worker cycles updating same vehicle assignment within seconds
- Current state: Uses ON CONFLICT DO UPDATE but DriverVehicleAssignment has no UNIQUE constraint on (driver_id, vehicle_id, assignment_end)
- Impact: Potential duplicate open assignments for same vehicle
- Fix approach: Add UNIQUE constraint on (vehicle_id, assignment_end IS NULL); use upsert with proper conflict handling

**Timezone-Aware Timestamp Serialization:**
- Symptoms: API responses may contain timestamps in different formats (ISO with/without timezone)
- Files: `backend/fastapi/routes.py` (line 94 uses `loc.timestamp.isoformat()` which may exclude timezone in some cases)
- Impact: Frontend may parse times incorrectly; cached data may have inconsistent timestamp format
- Fix approach: Enforce UTC timezone on all model timestamps; use explicit `isoformat(timespec='milliseconds')` format

---

## Security Considerations

**Weak Webhook Signature Validation:**
- Risk: Signature verification optional if `SAMSARA_SECRET` not set; silent fallthrough allows unsigned requests
- Files: `backend/fastapi/routes.py` (lines 207-227)
- Current mitigation: Checks `if secret := settings.samsara_secret_decoded` before validation
- Recommendations: 
  - Make webhook signature verification mandatory in production (check `DEPLOY_MODE`)
  - Log warning when accepting unsigned webhooks
  - Add IP whitelist validation as secondary check

**Geofence Event Processing Doesn't Validate Vehicle Ownership:**
- Risk: Any vehicle ID in webhook can modify database; no authorization check
- Files: `backend/fastapi/routes.py` (lines 279-304)
- Current mitigation: Only processes vehicles already in system (line 286-288 creates on first webhook)
- Recommendations:
  - Validate vehicle ID is in allowed set before processing
  - Log suspicious vehicle IDs
  - Add rate limiting per vehicle_id

**Database Credentials in Application Startup Logs:**
- Risk: DEBUG mode logs may echo DATABASE_URL containing credentials (user:pass@host)
- Files: `backend/fastapi/__init__.py` (line 33, `echo=settings.DEBUG`), `backend/worker/worker.py` (line 305)
- Current mitigation: SQLAlchemy echo should not print full URL, but connection pool logs may
- Recommendations:
  - Mask credentials before logging: use regex to replace `://.*:.*@` with `://***@`
  - Set DEBUG=false in production
  - Audit log output for credential leaks

**No Authentication on API Endpoints:**
- Risk: All endpoints publicly accessible; no user authentication or API keys required
- Files: `backend/fastapi/routes.py` (all GET endpoints, lines 42-567)
- Current mitigation: None; CORS restricts browser access but API is unprotected
- Recommendations:
  - Add API key validation for production endpoints
  - Implement rate limiting per IP/API key
  - Consider Firebase Auth or similar for future user features

---

## Performance Bottlenecks

**Inefficient Route Matching with Full Polyline Iteration:**
- Problem: Finding closest route requires computing haversine distance to every point in every route polyline
- Files: `shared/stops.py` (route matching algorithm)
- Cause: No spatial indexing (R-tree, quad-tree); O(n*m) where n=routes, m=polyline points
- Impact: Latency increases with polyline complexity; could block request thread if many vehicles
- Improvement path:
  - Build KD-tree or R-tree spatial index from polylines at startup
  - Cache route-finding results in Redis per (lat, lon) cell
  - Pre-segment polylines into rough grid; binary search instead of linear scan

**Uncached Vehicle Geofence Status Lookup:**
- Problem: `get_vehicles_in_geofence_query()` called repeatedly without Redis cache reuse
- Files: `backend/utils.py`, `backend/fastapi/utils.py` (line 174), `backend/worker/worker.py` (line 339)
- Cause: Subquery generated fresh each time; aggregates latest geofence event per vehicle from scratch
- Impact: O(n*m) scan across all geofence events; blocks for ~100ms+ on high-volume days
- Improvement path:
  - Cache full query result as bitmap in Redis; invalidate on webhook
  - Use materialized view in PostgreSQL for active vehicles
  - Index on (vehicle_id, event_time) exists but may not be used efficiently by query planner

**ML Dataframe Pipeline Reprocessing on Every Location Update:**
- Problem: `update_today_dataframe()` reprocesses rows through full ML pipeline even when only caching new rows
- Files: `backend/cache_dataframe.py`, `backend/worker/worker.py` (line 351)
- Cause: Window-based approach helps but still runs preprocess/segment/stops on context rows
- Impact: High CPU usage as data accumulates; may exceed 5-second worker cycle time
- Improvement path:
  - Skip already-processed cached rows entirely in additive mode
  - Consider async dataframe operations (polars, dask) for parallelism
  - Profile to confirm bottleneck before optimizing

**N+1 Query Pattern in ETA Endpoint:**
- Problem: `get_latest_etas()` may perform separate query per vehicle if fallback DB aggregation triggered
- Files: `backend/fastapi/utils.py` (line 147 via `get_latest_etas()`)
- Cause: Falls back to DB only if Redis miss; ETA computation is per-vehicle
- Impact: Slow response on cache misses; 100+ vehicles = 100 queries
- Improvement path:
  - Batch all vehicles in single SQL query with JSON aggregation
  - Pre-compute ETAs in worker and cache as single JSON blob
  - Ensure worker-computed Redis cache is always warm

---

## Fragile Areas

**Schedule JSON-to-JavaScript Type Coercion:**
- Files: `frontend/src/schedule/Schedule.tsx` (lines 11-12, 75-78), `frontend/src/locations/LiveLocation.tsx` (lines 6, 16-18)
- Why fragile: Assumes aggregatedSchedule.json structure exactly matches `AggregatedScheduleType` interface; uses unsafe `as unknown as` cast
- Safe modification: 
  - Create JSON schema file (`aggregated_schedule.schema.json`)
  - Use `ajv` library to validate at runtime before type assertion
  - Add tests comparing TypeScript types to actual JSON shapes
- Test coverage: No unit tests for schedule parsing/filtering logic; relies on manual testing

**Worker Prediction Pipeline Integration:**
- Files: `backend/worker/data.py` (lines 31-395), `backend/worker/worker.py` (lines 340-354)
- Why fragile:
  - ML models loaded dynamically from `ml/deploy/` directory; silent failures if model files missing (cached as None)
  - Dataframe column names hardcoded; upstream changes break silently (e.g., 'dist_to_route', 'polyline_idx')
  - No version tracking of model files vs code compatibility
- Safe modification:
  - Add schema validation before model loading
  - Store model metadata (training date, input shape) in model files
  - Verify dataframe columns before use; raise explicit errors if missing
- Test coverage: No tests for ML prediction pipeline integration with database/cache

**Webhook Event Parsing with Deep Nesting:**
- Files: `backend/fastapi/routes.py` (lines 260-276)
- Why fragile: Assumes specific Samsara webhook schema with nested geofenceEntry/geofenceExit; direct index access to `vertices[0]`
- Safe modification:
  - Create Pydantic models for webhook payload validation
  - Use `event_data.get("data", {}).get("conditions", [])` pattern throughout
  - Document exact expected webhook schema
- Test coverage: No tests for webhook parsing; relies on Samsara test server

**Cache Key Generation Based on Function Args:**
- Files: `backend/cache.py` (lines 86-127)
- Why fragile: Generates cache keys by converting args to strings; collisions possible if args have similar string representations
- Safe modification:
  - Add hash-based key generation as fallback for complex args
  - Include function fully-qualified name in key
  - Log cache key misses and collisions for monitoring
- Test coverage: No cache key collision tests

---

## Scaling Limits

**Redis Cache is Single Point of Failure:**
- Current capacity: Single Redis instance; no replication or clustering
- Limit: Cache misses cause load on database; no failover if Redis down
- Impact: If Redis unavailable, all endpoints must hit database directly; 60-100x slower responses
- Scaling path:
  - Add Redis Sentinel for automatic failover
  - Implement cache.py to gracefully degrade without Redis (accept stale data)
  - Consider Redis Cluster for multi-node setup in high-availability deployment

**Database Connection Pool May Exhaust Under Load:**
- Current capacity: Default asyncpg pool (min=10, max=10 by default in SQLAlchemy)
- Limit: 10 concurrent connections; burst requests beyond this queue
- Scaling path:
  - Monitor pool exhaustion; increase pool_size in `create_async_db_engine()`
  - Use PgBouncer or similar connection pooler in front of PostgreSQL
  - Implement circuit breaker pattern if pool exhaustion detected

**PostgreSQL Index on (vehicle_id, timestamp) May Not Scale to Millions of Rows:**
- Current capacity: Vehicle locations grow ~3000 per vehicle per day (6/minute * 480 minutes)
- Limit: B-tree index on 100k+ rows with DESC order may suffer from index bloat
- Scaling path:
  - Implement table partitioning by date (PARTITION BY RANGE)
  - Archive old data to separate tables
  - Rebuild indexes regularly with REINDEX CONCURRENTLY

**Worker Loop Cycles Every 5 Seconds:**
- Current capacity: Handles ~10-20 vehicles per cycle
- Limit: If more than 20 vehicles or cycle takes >5s, next cycle skips (logged as warning)
- Scaling path:
  - Parallelize vehicle location fetching across multiple worker processes
  - Implement async batching for Samsara API calls (concurrent.futures)
  - Monitor cycle time; alert if consistently >4s

---

## Dependencies at Risk

**NumPy/Pandas Version Compatibility:**
- Risk: ML pipeline heavily depends on NumPy/Pandas; no version pinning in requirements
- Impact: Minor version updates could break dtypes, indexing, or aggregation behavior
- Migration plan:
  - Pin major versions in requirements: `numpy>=1.24,<2.0`
  - Add compatibility tests on version update
  - Consider migration to polars (modern, faster alternative)

**Deprecated SQLAlchemy Usage:**
- Risk: Using `selectinload()` and `joinedload()` lazy loading patterns; SQLAlchemy 3.0 will require explicit loaders
- Impact: Code may break on major version upgrade
- Migration plan:
  - Audit all relationship queries; verify explicit loading strategy
  - Add type hints for relationship loading to catch issues early
  - Test on SQLAlchemy 2.1+ (pre-3.0) regularly

**Apple MapKit JS License/Availability:**
- Risk: Maps functionality depends on proprietary Apple service; no fallback if API key invalid
- Impact: Maps blank if key unavailable; no graceful degradation to OSM alternative
- Migration plan:
  - Add fallback to Leaflet + OpenStreetMap tiles
  - Detect MapKit initialization failures; switch to fallback automatically
  - Document both map providers as supported

---

## Missing Critical Features

**No Data Validation on API Responses:**
- Problem: Frontend receives data from `/api/locations`, `/api/etas` but doesn't validate schema
- Blocks: Cannot detect/recover from API contract changes; no type safety at runtime
- Impact: Type mismatch can cause silent failures or crashes in map rendering
- Fix: Implement Zod or similar runtime validation on all API responses

**No Distributed Request Tracing:**
- Problem: Cannot correlate logs across worker → database → API → frontend
- Blocks: Debugging performance issues or error chains across services
- Impact: Hard to identify root cause of slow requests or errors
- Fix: Add OpenTelemetry or similar; implement trace ID propagation

**No Automated Alerting on Worker Failures:**
- Problem: Worker loop catches all exceptions and logs; no alert if cycle fails repeatedly
- Blocks: Cannot detect chronic issues (e.g., API key expired) until users notice
- Impact: Service degradation (no location updates) goes unnoticed for hours
- Fix: Add health check endpoint that fails if worker hasn't updated data in last 15 minutes

**No Rate Limiting on Public Endpoints:**
- Problem: API endpoints have no rate limiting; client can make unlimited requests
- Blocks: Cannot protect against accidental or malicious abuse
- Impact: Slow Samsara API calls could amplify into DDoS attack on database
- Fix: Add middleware for rate limiting (e.g., slowapi, limits by IP)

---

## Test Coverage Gaps

**No Tests for Webhook Processing:**
- What's not tested: Geofence event parsing, vehicle creation, edge cases (missing fields, invalid timestamps)
- Files: `backend/fastapi/routes.py` (lines 200-332)
- Risk: Silent failures in webhook parsing; data loss if edge case triggers exception
- Priority: High (critical data ingestion path)

**No Tests for Frontend Component Integration:**
- What's not tested: Schedule + Map interaction, route selection persistence, ETA updating
- Files: `frontend/src/locations/LiveLocation.tsx`, `frontend/src/schedule/Schedule.tsx`
- Risk: UI bugs only caught during manual testing
- Priority: Medium (UX impact)

**No Tests for ML Pipeline Integration:**
- What's not tested: Dataframe preprocessing, model loading, prediction accuracy
- Files: `backend/worker/data.py`, `backend/cache_dataframe.py`
- Risk: ML predictions silently degrade without detection
- Priority: High (user-visible feature degradation)

**No Tests for Cache Invalidation:**
- What's not tested: Cache coherency across namespaces, soft_clear_namespace behavior, race conditions
- Files: `backend/cache.py`, cache decorators throughout
- Risk: Stale data served to users; cache poisoning on concurrent updates
- Priority: Medium (data correctness)

**No Load/Stress Tests:**
- What's not tested: Performance under 50+ concurrent requests, worker cycle time with 100+ vehicles
- Risk: Unknown failure modes under production load
- Priority: High (before production deployment)

---

## Additional Observations

**Code Duplication in Routes:**
- Database query patterns repeated in multiple endpoints (get locations, get schedule, get routes)
- Consider extracting common patterns to utility functions

**Magic String Constants:**
- Route names like "ENTRY", "EXIT" hardcoded in multiple places
- Consider centralizing in constants file

**Error Messages Logged Without Context:**
- Many log statements lack sufficient context to debug (e.g., "API error: 500" without URL)
- Add function/resource context to error logs

**Documentation Debt:**
- No OpenAPI/Swagger documentation for API endpoints despite FastAPI support
- Consider adding endpoint docstrings with response schemas

---

*Concerns audit: 2026-04-05*
