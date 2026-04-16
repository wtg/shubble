"""Tests for Phase 3 break detection: CUSUM + stay-point + fallback.

Covers:
  - CUSUM: adaptive mu, fires at mu+6, reset on Union visit, default mu
  - Stay-point: dwell threshold, off-route gate, on-route suppression
  - predict_on_break: all three signals, active-window gating
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import pytest

from backend.fastapi.break_detection import (
    _compress_dwells,
    _cusum_fires,
    _compute_personal_mu,
    _detect_active_stay_point,
    _min_dist_to_any_route_km,
    _union_visits_utc,
    predict_on_break,
    CUSUM_SLACK_MIN,
    CUSUM_THRESHOLD_MIN,
    CUSUM_DEFAULT_MU_MIN,
    CUSUM_MIN_VISITS,
    FALLBACK_GAP_MIN,
    STAY_POINT_MIN_DWELL_SEC,
)


TZ = ZoneInfo("America/New_York")

UNION_LAT = 42.730711
UNION_LON = -73.676737
BREAK_SPOT_LAT = 42.7265
BREAK_SPOT_LON = -73.672


def _utc(hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime(2025, 9, 22, hh, mm, ss, tzinfo=timezone.utc)


# ---------- CUSUM ----------


class TestCUSUM:
    def test_fires_when_gap_exceeds_floor(self):
        # 4 Union visits at 10-min intervals → mu ≈ 10 min.
        # Adaptive threshold = 10+1+5 = 16, but floor is 20.
        # Elapsed = 21 min > 20 floor → fires.
        visits = [_utc(11, 0), _utc(11, 10), _utc(11, 20), _utc(11, 30)]
        fires, mu, elapsed = _cusum_fires(visits, _utc(11, 51))
        assert fires is True
        assert 9 <= mu <= 11

    def test_does_not_fire_within_floor(self):
        # Same visits, elapsed = 19 min < 20 floor → doesn't fire.
        visits = [_utc(11, 0), _utc(11, 10), _utc(11, 20), _utc(11, 30)]
        fires, mu, elapsed = _cusum_fires(visits, _utc(11, 49))
        assert fires is False

    def test_uses_default_mu_with_few_visits(self):
        # Only 2 visits (below CUSUM_MIN_VISITS=3). Uses default mu=12.
        # Adaptive = 12+1+5 = 18, floor = 20. Elapsed = 21 > 20 → fires.
        visits = [_utc(11, 0), _utc(11, 10)]
        fires, mu, elapsed = _cusum_fires(visits, _utc(11, 31))
        assert fires is True
        assert mu == CUSUM_DEFAULT_MU_MIN

    def test_default_mu_no_fire_within_floor(self):
        # Elapsed = 19 < floor 20 → no fire.
        visits = [_utc(11, 0), _utc(11, 10)]
        fires, _, _ = _cusum_fires(visits, _utc(11, 29))
        assert fires is False

    def test_excludes_break_gaps_from_mu(self):
        # Visits with a 50-min break in the middle (should be excluded from EMA).
        visits = [_utc(10, 0), _utc(10, 12), _utc(11, 2), _utc(11, 14)]
        # Intervals: 12, 50 (break!), 12. Normal intervals: [12, 12]. mu ≈ 12.
        fires, mu, elapsed = _cusum_fires(visits, _utc(11, 14))
        assert 11 <= mu <= 13  # Should be ~12, NOT inflated by the 50-min gap
        assert fires is False  # elapsed = 0

    def test_no_visits_returns_false(self):
        fires, _, _ = _cusum_fires([], _utc(12, 0))
        assert fires is False

    def test_adapts_to_slow_shuttle(self):
        # 15-min loop shuttle. mu ≈ 15. Threshold = 15+1+5 = 21.
        visits = [_utc(10, 0), _utc(10, 15), _utc(10, 30), _utc(10, 45)]
        # At 11:05 (20 min gap) → shouldn't fire yet.
        fires, mu, _ = _cusum_fires(visits, _utc(11, 5))
        assert fires is False
        assert 14 <= mu <= 16
        # At 11:07 (22 min gap) → fires.
        fires, _, _ = _cusum_fires(visits, _utc(11, 7))
        assert fires is True


class TestPersonalMu:
    def test_ema_weights_recent(self):
        # [10, 10, 10, 20] → EMA biased toward 20 but not 20.
        mu = _compute_personal_mu([10.0, 10.0, 10.0, 20.0])
        assert 12 < mu < 18

    def test_excludes_break_gaps(self):
        mu = _compute_personal_mu([10.0, 50.0, 12.0])
        assert 10 <= mu <= 13  # 50 excluded


# ---------- Stay-point ----------


class TestStayPoint:
    def test_fires_after_5min_off_route_dwell(self):
        now = _utc(16, 0)
        pings = [
            (now - timedelta(minutes=i), BREAK_SPOT_LAT, BREAK_SPOT_LON)
            for i in range(6, -1, -1)
        ]
        sp = _detect_active_stay_point(pings, now)
        assert sp is not None
        assert sp[3] >= STAY_POINT_MIN_DWELL_SEC

    def test_does_not_fire_under_threshold(self):
        now = _utc(16, 0)
        pings = [
            (now - timedelta(minutes=1), BREAK_SPOT_LAT, BREAK_SPOT_LON),
            (now, BREAK_SPOT_LAT, BREAK_SPOT_LON),
        ]
        assert _detect_active_stay_point(pings, now) is None

    def test_moving_shuttle_not_detected(self):
        now = _utc(16, 0)
        pings = [
            (now - timedelta(minutes=i), 42.73, -73.68 + i * 0.003)
            for i in range(6, -1, -1)
        ]
        assert _detect_active_stay_point(pings, now) is None


class TestOffRoute:
    def test_union_is_on_route(self):
        assert _min_dist_to_any_route_km(UNION_LAT, UNION_LON) < 0.05

    def test_break_spot_is_off_route(self):
        assert _min_dist_to_any_route_km(BREAK_SPOT_LAT, BREAK_SPOT_LON) > 0.1


# ---------- Integration ----------


class _StubSession:
    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self): return self
    async def __aexit__(self, *a): pass
    async def execute(self, _stmt):
        rows = self._rows
        class _Result:
            def all(self_inner): return rows
        return _Result()


class _StubSessionFactory:
    def __init__(self, rows):
        self._rows = rows
    def __call__(self): return _StubSession(self._rows)


def _ping(vid, ts, lat=UNION_LAT, lon=UNION_LON):
    return (vid, ts, lat, lon)


class TestPredictOnBreak:
    @pytest.mark.asyncio
    async def test_cusum_fires_on_extended_gap(self, monkeypatch):
        # 4 Union visits at 10-min intervals, then 21-min gap.
        # mu~10, adaptive=16, floor=20. 21 > 20 → fires.
        # 15:00-15:30 UTC = 11:00-11:30 ET (inside 8-20 active window).
        fixed = _utc(15, 51)
        monkeypatch.setattr("backend.fastapi.break_detection.dev_now",
                            lambda tz=None: fixed)
        monkeypatch.setattr("backend.fastapi.break_detection.get_campus_start_of_day",
                            lambda: _utc(4, 0))
        rows = [_ping("V1", _utc(15, m)) for m in [0, 10, 20, 30]]
        result = await predict_on_break(["V1"], _StubSessionFactory(rows), TZ)
        assert result["V1"] is True

    @pytest.mark.asyncio
    async def test_cusum_quiet_during_normal_service(self, monkeypatch):
        # 15:40 UTC = 11:40 ET. Elapsed=10min from 15:30. mu~10 → 10 < 16 → no fire.
        fixed = _utc(15, 40)
        monkeypatch.setattr("backend.fastapi.break_detection.dev_now",
                            lambda tz=None: fixed)
        monkeypatch.setattr("backend.fastapi.break_detection.get_campus_start_of_day",
                            lambda: _utc(4, 0))
        rows = [_ping("V1", _utc(15, m)) for m in [0, 10, 20, 30]]
        result = await predict_on_break(["V1"], _StubSessionFactory(rows), TZ)
        assert result["V1"] is False

    @pytest.mark.asyncio
    async def test_stay_point_off_route_fires(self, monkeypatch):
        """Shuttle parked at off-route break spot for 6 min → fires."""
        fixed = _utc(16, 30)
        monkeypatch.setattr("backend.fastapi.break_detection.dev_now",
                            lambda tz=None: fixed)
        monkeypatch.setattr("backend.fastapi.break_detection.get_campus_start_of_day",
                            lambda: _utc(4, 0))
        rows = [_ping("V1", fixed - timedelta(minutes=20))]
        for i in range(6, -1, -1):
            rows.append(_ping("V1", fixed - timedelta(minutes=i),
                              lat=BREAK_SPOT_LAT, lon=BREAK_SPOT_LON))
        result = await predict_on_break(["V1"], _StubSessionFactory(rows), TZ)
        assert result["V1"] is True

    @pytest.mark.asyncio
    async def test_stay_point_on_route_not_union_fires(self, monkeypatch):
        """Shuttle parked at COLONIE (on-route, not Union) for 6 min → fires.
        This is the key improvement: on-route idling is now caught at T+5."""
        fixed = _utc(16, 30)
        monkeypatch.setattr("backend.fastapi.break_detection.dev_now",
                            lambda tz=None: fixed)
        monkeypatch.setattr("backend.fastapi.break_detection.get_campus_start_of_day",
                            lambda: _utc(4, 0))
        COLONIE_LAT, COLONIE_LON = 42.737048, -73.670397
        rows = [_ping("V1", fixed - timedelta(minutes=20))]  # Union visit (in-service)
        for i in range(6, -1, -1):
            rows.append(_ping("V1", fixed - timedelta(minutes=i),
                              lat=COLONIE_LAT, lon=COLONIE_LON))
        result = await predict_on_break(["V1"], _StubSessionFactory(rows), TZ)
        assert result["V1"] is True

    @pytest.mark.asyncio
    async def test_stay_point_at_union_suppressed(self, monkeypatch):
        """Shuttle dwelling at Union (normal between-loop wait) → no fire."""
        fixed = _utc(16, 30)
        monkeypatch.setattr("backend.fastapi.break_detection.dev_now",
                            lambda tz=None: fixed)
        monkeypatch.setattr("backend.fastapi.break_detection.get_campus_start_of_day",
                            lambda: _utc(4, 0))
        rows = [_ping("V1", fixed - timedelta(minutes=i))
                for i in range(6, -1, -1)]
        result = await predict_on_break(["V1"], _StubSessionFactory(rows), TZ)
        assert result["V1"] is False

    @pytest.mark.asyncio
    async def test_fallback_fires_at_40min(self, monkeypatch):
        fixed = _utc(16, 30)
        monkeypatch.setattr("backend.fastapi.break_detection.dev_now",
                            lambda tz=None: fixed)
        monkeypatch.setattr("backend.fastapi.break_detection.get_campus_start_of_day",
                            lambda: _utc(4, 0))
        # One Union visit 45 min ago.
        rows = [_ping("V1", fixed - timedelta(minutes=45))]
        result = await predict_on_break(["V1"], _StubSessionFactory(rows), TZ)
        assert result["V1"] is True

    @pytest.mark.asyncio
    async def test_no_visits_returns_false(self, monkeypatch):
        fixed = _utc(12, 0)
        monkeypatch.setattr("backend.fastapi.break_detection.dev_now",
                            lambda tz=None: fixed)
        monkeypatch.setattr("backend.fastapi.break_detection.get_campus_start_of_day",
                            lambda: _utc(4, 0))
        result = await predict_on_break(["V1"], _StubSessionFactory([]), TZ)
        assert result["V1"] is False

    @pytest.mark.asyncio
    async def test_night_window_suppresses_all_signals(self, monkeypatch):
        # 22:00 local = outside active window. Should not fire even with long gap.
        fixed = _utc(2, 0)  # 22:00 ET
        monkeypatch.setattr("backend.fastapi.break_detection.dev_now",
                            lambda tz=None: fixed)
        monkeypatch.setattr("backend.fastapi.break_detection.get_campus_start_of_day",
                            lambda: datetime(2025, 9, 21, 4, 0, tzinfo=timezone.utc))
        rows = [_ping("V1", fixed - timedelta(hours=2))]
        result = await predict_on_break(["V1"], _StubSessionFactory(rows), TZ)
        assert result["V1"] is False
