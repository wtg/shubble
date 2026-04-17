# Testing Patterns

**Analysis Date:** 2026-04-05

## Test Framework

**Runner:**
- **pytest** - Python async test runner
- Config: No `pytest.ini` found; pytest configuration in `pyproject.toml` (project root)
- Async support: `pytest-asyncio` (specified in dev dependencies)

**Assertion Library:**
- Python standard `assert` statements (no pytest assertions needed)
- Mock objects via `unittest.mock` (AsyncMock, MagicMock, patch)

**Run Commands:**
```bash
pytest tests/                           # Run all tests
pytest tests/ -v                        # Verbose output
pytest tests/test_eta_computation.py    # Single file
pytest tests/ -k "test_north_route"     # Filter by name
pytest tests/ --asyncio-mode=auto       # Async mode (required for pytest-asyncio)
```

## Test File Organization

**Location:**
- Backend tests: `tests/` directory at project root (co-located with source, not in backend/)
- Test discovery pattern: `test_*.py` files in `tests/` and `tests/simulation/`
- Frontend tests: Not found (minimal frontend testing)

**Naming:**
- Test functions: `test_<description>` (e.g., `test_north_route_offsets_loaded()`)
- Test classes: `Test<Feature>` (e.g., `TestETAComputationPerformance`)
- Test files: `test_<module>.py` (e.g., `test_eta_computation.py`)

**Structure:**
```
tests/
├── test_eta_computation.py      # ETA math and computation logic
├── test_eta_api.py              # API response shape and aggregation
├── test_velocity.py             # Speed prediction models
└── simulation/
    ├── simulator.py             # Simulation engine (fixtures and data generation)
    ├── test_backend_performance.py  # Backend load/performance tests
    ├── test_data_integrity.py   # Data quality tests
    └── test_frontend_ux.py      # Integration tests
```

## Test Structure

**Suite Organization:**

```python
# Simple function tests
def test_north_route_offsets_loaded():
    """Verify routes.json OFFSET values are accessible."""
    routes = Stops.routes_data
    assert "NORTH" in routes
    north = routes["NORTH"]
    assert north["STUDENT_UNION"]["OFFSET"] == 0

# Async test with fixtures
@pytest.mark.asyncio
async def test_compute_per_stop_etas_single_vehicle_north():
    """One vehicle on NORTH at polyline_idx=2."""
    now = datetime.now(timezone.utc)
    df = make_vehicle_df("v1", "NORTH", 2, timestamp=now)
    
    mock_lstm = AsyncMock(return_value={"v1": stac1_eta})
    with patch("backend.worker.data.predict_eta", mock_lstm):
        result = await compute_per_stop_etas(["v1"], df=df)
    
    assert "STAC_1" in result

# Class-based test suite
class TestETAComputationPerformance:
    """Profile compute_per_stop_etas under simulated load."""
    
    def _make_bulk_dataframe(self, sim: SimulationResult, n_vehicles: int = 10) -> pd.DataFrame:
        """Helper: create test dataframe."""
        rows = []
        # ... populate rows ...
        return pd.DataFrame(rows)
    
    @pytest.mark.asyncio
    async def test_compute_per_stop_etas_under_200ms(self, sim: SimulationResult):
        """ETA computation for 10 vehicles should complete under 200ms."""
        # ... test implementation ...
```

**Patterns:**
- **Setup**: Inline via helper functions (e.g., `make_vehicle_df()`) or class methods (e.g., `_make_bulk_dataframe()`)
- **Teardown**: None observed; tests are stateless with mocked external dependencies
- **Assertion pattern**: Direct `assert` statements with descriptive failure messages

## Mocking

**Framework:** `unittest.mock` (built-in)

**Patterns:**

```python
# Async mock
from unittest.mock import AsyncMock, patch

mock_lstm = AsyncMock(return_value={"v1": stac1_eta})

with patch("backend.worker.data.predict_eta", mock_lstm):
    result = await compute_per_stop_etas(["v1"], df=df)

# MagicMock for database rows
from unittest.mock import MagicMock

row = MagicMock()
row.vehicle_id = "v1"
row.etas = {"COLONIE": {"eta": future, "route": "NORTH"}}

# Mock session factory (async context manager)
mock_session = AsyncMock()
mock_result = MagicMock()
mock_result.scalars.return_value.all.return_value = mock_etas
mock_session.execute = AsyncMock(return_value=mock_result)

mock_factory = MagicMock()
mock_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
mock_factory.return_value.__aexit__ = AsyncMock(return_value=False)
```

**What to Mock:**
- External service calls: API responses (e.g., Samsara API, LSTM model predictions)
- Database queries: Session factory, query results
- Time-dependent code: Use fixed `datetime.now(timezone.utc)` + test-specific timestamps
- Heavy ML models: LSTM, ARIMA models replaced with AsyncMock returning deterministic data

**What NOT to Mock:**
- Route matching logic (`shared/stops.py`) — uses real `routes.json` data
- Schedule data (`shared/schedule.json`) — real data needed for offset calculations
- Helper functions (`make_vehicle_df()`) — create real dataframes for computation testing

## Fixtures and Factories

**Test Data:**

```python
def make_vehicle_df(vehicle_id: str, route: str, polyline_idx: int,
                    lat: float = 42.73, lon: float = -73.68,
                    speed_kmh: float = 20.0,
                    stop_name=None,
                    timestamp=None):
    """Create a minimal dataframe row mimicking the preprocessed worker data."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    rows = []
    # Need at least 10 rows for LSTM (sequence_length)
    for i in range(12):
        ts = timestamp - timedelta(seconds=(11 - i) * 5)
        rows.append({
            "vehicle_id": str(vehicle_id),
            "latitude": lat + i * 0.0001,
            "longitude": lon + i * 0.0001,
            "speed_kmh": speed_kmh,
            "timestamp": ts,
            "route": route,
            "polyline_idx": polyline_idx,
            "stop_name": stop_name if i == 11 else None,
        })
    return pd.DataFrame(rows)

def make_mock_eta_row(vehicle_id: str, etas: dict, timestamp=None):
    """Create a mock ETA database row."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    row = MagicMock()
    row.vehicle_id = vehicle_id
    row.etas = etas
    row.timestamp = timestamp
    return row
```

**Location:**
- Defined inline in test files (e.g., `tests/test_eta_computation.py` lines 14-36)
- Module-level fixtures in `tests/simulation/simulator.py` for simulation data
- Pytest fixtures with `@pytest.fixture(scope="module")` for expensive setup (e.g., simulation runs)

## Coverage

**Requirements:** No coverage thresholds enforced (no `.coveragerc` or pytest plugin configuration found)

**View Coverage:**
```bash
# Assuming pytest-cov is available (not in pyproject.toml)
pytest tests/ --cov=backend --cov-report=html
```

## Test Types

**Unit Tests:**
- **Scope**: Individual functions and methods
- **Approach**: Mock all external dependencies; test logic in isolation
- **Examples**:
  - `test_north_route_offsets_loaded()` — data loading
  - `test_offset_diff_calculation()` — math logic
  - `test_velocity_predictor_protocol_compliance()` — interface conformance
- **Coverage**: All files in `tests/test_*.py` are unit tests

**Integration Tests:**
- **Scope**: Multi-component interactions (worker + database + cache)
- **Approach**: Use simulation engine to generate realistic data; verify end-to-end behavior
- **Examples**:
  - `test_backend_performance.py` — ETA computation under load
  - `test_data_integrity.py` — data consistency through pipeline
  - `test_frontend_ux.py` — API response integration
- **Setup**: `tests/simulation/simulator.py` provides realistic vehicle movement data

**E2E Tests:**
- **Framework**: Not used (no Cypress, Selenium, or Playwright config found)
- **Note**: Frontend is primarily UI; integration tests via API mocking are sufficient

## Common Patterns

**Async Testing:**

```python
@pytest.mark.asyncio
async def test_compute_per_stop_etas_single_vehicle_north():
    """One vehicle on NORTH at polyline_idx=2."""
    now = datetime.now(timezone.utc)
    df = make_vehicle_df("v1", "NORTH", 2, timestamp=now)
    
    stac1_eta = now + timedelta(minutes=2)
    mock_lstm = AsyncMock(return_value={"v1": stac1_eta})
    
    with patch("backend.worker.data.predict_eta", mock_lstm), \
         patch("backend.worker.data.get_today_dataframe", AsyncMock(return_value=df)):
        result = await compute_per_stop_etas(["v1"], df=df)
    
    assert "STAC_1" in result
```

**Pattern:**
- Mark with `@pytest.mark.asyncio` to run async functions
- Mock async functions with `AsyncMock(return_value=...)`
- Use `with patch(...):` context manager for test isolation
- No explicit event loop setup needed (pytest-asyncio handles it)

**Error Testing:**

```python
@pytest.mark.asyncio
async def test_past_etas_filtered_out():
    """ETAs in the past should not appear in results."""
    # Set timestamp far in the past so all ETAs will be in the past
    past = datetime(2020, 1, 1, tzinfo=timezone.utc)
    df = make_vehicle_df("v1", "NORTH", 2, timestamp=past)
    
    past_eta = past + timedelta(minutes=2)
    mock_lstm = AsyncMock(return_value={"v1": past_eta})
    
    with patch("backend.worker.data.predict_eta", mock_lstm):
        result = await compute_per_stop_etas(["v1"], df=df)
    
    # All ETAs should be filtered out since they're in 2020
    assert len(result) == 0
```

**Pattern:**
- Use fixed past/future timestamps to trigger edge cases
- Assert on collection length or absence of keys
- No exception matching needed (test for side effects instead)

**Multi-vehicle Aggregation:**

```python
@pytest.mark.asyncio
async def test_multi_vehicle_aggregation():
    """Two vehicles on different routes — each stop gets earliest ETA."""
    now = datetime.now(timezone.utc)
    df_north = make_vehicle_df("v1", "NORTH", 0, timestamp=now)
    df_west = make_vehicle_df("v2", "WEST", 0, timestamp=now)
    df = pd.concat([df_north, df_west], ignore_index=True)
    
    north_eta = now + timedelta(minutes=2)
    west_eta = now + timedelta(minutes=1)
    mock_lstm = AsyncMock(return_value={"v1": north_eta, "v2": west_eta})
    
    with patch("backend.worker.data.predict_eta", mock_lstm):
        result = await compute_per_stop_etas(["v1", "v2"], df=df)
    
    # Verify aggregation rules
    if "COLONIE" in result:
        assert result["COLONIE"]["vehicle_id"] == "v1"
```

**Pattern:**
- Concatenate test dataframes for multi-vehicle scenarios
- Verify aggregation logic (earliest ETA wins, no mixed orderings)
- Assert on both presence and source of results

## Performance Testing

**Approach:** Use simulation fixtures to generate realistic load

```python
@pytest.mark.asyncio
async def test_compute_per_stop_etas_under_200ms(self, sim: SimulationResult):
    """ETA computation for 10 vehicles should complete under 200ms."""
    n_vehicles = 10
    df = self._make_bulk_dataframe(sim, n_vehicles=n_vehicles)
    vehicle_ids = [f"perf-v{i}" for i in range(n_vehicles)]
    
    start = time.perf_counter()
    result = await compute_per_stop_etas(vehicle_ids, df=df)
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    assert elapsed_ms < 500, (
        f"CRITICAL: compute_per_stop_etas took {elapsed_ms:.0f}ms for "
        f"{n_vehicles} vehicles (threshold: 500ms)"
    )
```

**Pattern:**
- Use `time.perf_counter()` for high-resolution timing
- Include detailed failure messages with actual vs threshold
- Test under realistic data volume (10+ vehicles, 20 GPS points each)

---

*Testing analysis: 2026-04-05*
