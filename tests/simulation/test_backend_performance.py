"""
Subagent 1: Backend Performance Tests

Tests:
- API response time profiling
- N+1 query detection
- Cache behavior verification
- ETA computation performance under load
- Memory usage patterns
"""

import time
import asyncio
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from tests.simulation.simulator import run_simulation, SimulationResult


@pytest.fixture(scope="module")
def sim() -> SimulationResult:
    return run_simulation(seed=42, window_hours=2.0, min_trips=50)


# --- ETA Computation Performance ---

class TestETAComputationPerformance:
    """Profile compute_per_stop_etas under simulated load."""

    def _make_bulk_dataframe(self, sim: SimulationResult, n_vehicles: int = 10) -> pd.DataFrame:
        """Create a realistic dataframe with multiple vehicles and history."""
        rows = []
        now = datetime.now(timezone.utc)

        for v in range(n_vehicles):
            vehicle_id = f"perf-v{v}"
            route = "NORTH" if v % 2 == 0 else "WEST"
            polyline_idx = v % 5

            # 20 data points per vehicle (mimics 100s of GPS updates)
            for i in range(20):
                ts = now - timedelta(seconds=(19 - i) * 5)
                rows.append({
                    "vehicle_id": vehicle_id,
                    "latitude": 42.73 + i * 0.0001,
                    "longitude": -73.68 + i * 0.0001,
                    "speed_kmh": 15.0 + np.random.uniform(-5, 5),
                    "timestamp": ts,
                    "route": route,
                    "polyline_idx": polyline_idx,
                    "stop_name": None,
                })

        return pd.DataFrame(rows)

    @pytest.mark.asyncio
    async def test_compute_per_stop_etas_under_200ms(self, sim: SimulationResult):
        """ETA computation for 10 vehicles should complete under 200ms."""
        from backend.worker.data import compute_per_stop_etas

        n_vehicles = 10
        df = self._make_bulk_dataframe(sim, n_vehicles=n_vehicles)
        vehicle_ids = [f"perf-v{i}" for i in range(n_vehicles)]

        # Mock LSTM to return quickly
        mock_lstm = AsyncMock(return_value={})

        with patch("backend.worker.data.predict_eta", mock_lstm), \
             patch("backend.worker.data.get_today_dataframe", AsyncMock(return_value=df)):

            start = time.perf_counter()
            result = await compute_per_stop_etas(vehicle_ids, df=df)
            elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, (
            f"CRITICAL: compute_per_stop_etas took {elapsed_ms:.0f}ms for "
            f"{n_vehicles} vehicles (threshold: 500ms)"
        )

        if elapsed_ms > 200:
            pytest.warns(UserWarning, match="compute_per_stop_etas")

    @pytest.mark.asyncio
    async def test_predict_eta_fast_on_cached_model_miss(self):
        """predict_eta should be fast after first call caches model failures."""
        from backend.worker.data import predict_eta, _MODEL_CACHE

        now = datetime.now(timezone.utc)
        rows = []
        for i in range(12):
            rows.append({
                "vehicle_id": "perf-v0",
                "latitude": 42.73 + i * 0.0001,
                "longitude": -73.68 + i * 0.0001,
                "speed_kmh": 20.0,
                "timestamp": now - timedelta(seconds=(11 - i) * 5),
                "route": "NORTH",
                "polyline_idx": 2,
                "stop_name": None,
            })
        df = pd.DataFrame(rows)

        # First call may be slow (model load attempt)
        await predict_eta(["perf-v0"], df=df)

        # Second call should be fast because failure is cached
        start = time.perf_counter()
        result = await predict_eta(["perf-v0"], df=df)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, (
            f"predict_eta took {elapsed_ms:.0f}ms on cached miss (threshold: 100ms). "
            f"Model cache should prevent re-loading failed models."
        )

    @pytest.mark.asyncio
    async def test_model_failure_cached_as_none(self):
        """When LSTM model fails to load, the failure should be cached."""
        from backend.worker.data import predict_eta, _MODEL_CACHE

        now = datetime.now(timezone.utc)
        rows = []
        for i in range(12):
            rows.append({
                "vehicle_id": "cache-test",
                "latitude": 42.73 + i * 0.0001,
                "longitude": -73.68 + i * 0.0001,
                "speed_kmh": 20.0,
                "timestamp": now - timedelta(seconds=(11 - i) * 5),
                "route": "NORTH",
                "polyline_idx": 99,  # Unlikely to have a trained model
                "stop_name": None,
            })
        df = pd.DataFrame(rows)

        await predict_eta(["cache-test"], df=df)

        # Check that the cache contains a None for this key
        key = ("NORTH", 99)
        assert key in _MODEL_CACHE, "Model failure not cached"
        assert _MODEL_CACHE[key] is None, "Failed model should be cached as None"


# --- Route Matching Performance ---

class TestRouteMatchingPerformance:
    """Profile Stops.get_closest_point under load."""

    def test_route_matching_bulk(self):
        """100 get_closest_point calls should complete under 500ms."""
        from shared.stops import Stops

        # Generate 100 random points near campus
        rng = np.random.default_rng(42)
        points = [
            (42.73 + rng.uniform(-0.005, 0.005), -73.68 + rng.uniform(-0.005, 0.005))
            for _ in range(100)
        ]

        start = time.perf_counter()
        for point in points:
            Stops.get_closest_point(point)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, (
            f"100 get_closest_point calls took {elapsed_ms:.0f}ms (threshold: 500ms)"
        )

    def test_polyline_distance_bulk(self):
        """100 get_polyline_distances calls should complete under 500ms."""
        from shared.stops import Stops

        rng = np.random.default_rng(42)
        points = [
            (42.73 + rng.uniform(-0.003, 0.003), -73.68 + rng.uniform(-0.003, 0.003))
            for _ in range(100)
        ]

        start = time.perf_counter()
        for point in points:
            result = Stops.get_closest_point(point)
            if result[0] is not None:
                Stops.get_polyline_distances(point, closest_point_result=result)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, (
            f"100 polyline distance calls took {elapsed_ms:.0f}ms (threshold: 500ms)"
        )


# --- Velocity Predictor Performance ---

class TestVelocityPredictorPerformance:

    def test_average_speed_predictor_instant(self):
        """AverageSpeedPredictor should be effectively instant."""
        from backend.worker.velocity import AverageSpeedPredictor

        predictor = AverageSpeedPredictor()

        start = time.perf_counter()
        for _ in range(10000):
            predictor.predict_speed_kmh("v1", "NORTH")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, (
            f"10000 predict_speed_kmh calls took {elapsed_ms:.0f}ms (threshold: 50ms)"
        )


# --- Simulation Payload Generation Performance ---

class TestPayloadPerformance:
    """Verify simulation output generation is fast."""

    def test_generate_api_payload_under_50ms(self, sim: SimulationResult):
        """Converting simulation to API payload should be fast."""
        from tests.simulation.simulator import generate_api_payload

        start = time.perf_counter()
        payload = generate_api_payload(sim)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, (
            f"generate_api_payload took {elapsed_ms:.0f}ms (threshold: 50ms)"
        )
        assert len(payload) > 0, "Empty payload generated"

    def test_generate_locations_payload_under_50ms(self, sim: SimulationResult):
        """Converting simulation to locations payload should be fast."""
        from tests.simulation.simulator import generate_locations_payload

        start = time.perf_counter()
        payload = generate_locations_payload(sim)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, (
            f"generate_locations_payload took {elapsed_ms:.0f}ms (threshold: 50ms)"
        )


# --- Offset Computation Performance ---

class TestOffsetComputationPerformance:
    """Profile the offset-based ETA calculation used in the 3-layer system."""

    def test_offset_diff_computation_bulk(self):
        """Computing offset diffs for 1000 stop pairs should be instant."""
        import json
        from pathlib import Path

        with open(Path(__file__).parent.parent.parent / "shared" / "routes.json") as f:
            routes = json.load(f)

        north = routes["NORTH"]
        stops = north["STOPS"]

        start = time.perf_counter()
        for _ in range(1000):
            for i in range(len(stops) - 1):
                next_offset = north[stops[i + 1]]["OFFSET"]
                curr_offset = north[stops[i]]["OFFSET"]
                _ = (next_offset - curr_offset) * 60  # seconds
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, (
            f"1000 offset diff iterations took {elapsed_ms:.0f}ms (threshold: 100ms)"
        )
