"""Tests for Phase 3 break detection: archetype extractor + matcher.

Covers:
  - Extractor: break window, morning signature, dwell compression, clustering
  - Matcher: Hungarian assignment, NaN-masked cost, unmatch on high cost
  - predict_on_break: archetype branch, 40-min fallback, OR combination
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from ml.build_archetypes import (
    _compress_dwells,
    _find_break_start,
    _morning_signature,
    _cluster_center_with_signature,
    BREAK_GAP_MIN,
)
from backend.fastapi.break_archetype import (
    match_signatures,
    _signature_cost,
    _extract_signature,
    _compress_dwells as bak_compress_dwells,
    MAX_MATCH_RMSE,
    SIGNATURE_LEN,
    predict_on_break,
    reset_archetype_cache,
    _ARCHETYPES_CACHE,
)


TZ = ZoneInfo("America/New_York")


# ---------- Extractor helpers ----------


def _ts(hh: int, mm: int, dd: int = 1) -> pd.Timestamp:
    """Synthetic local-tz timestamp for a test day."""
    return pd.Timestamp(year=2025, month=9, day=dd, hour=hh, minute=mm, tz=TZ)


class TestExtractor:
    def test_compress_dwells_collapses_close_pings(self):
        times = [_ts(8, 0), _ts(8, 0, dd=1).replace(second=30), _ts(8, 1)]
        # All within 2min of each other → one visit.
        assert len(_compress_dwells(times)) == 1

    def test_compress_dwells_splits_on_long_gap(self):
        times = [_ts(8, 0), _ts(8, 30), _ts(8, 30).replace(second=45)]
        # 30-min gap separates into two visits; last two collapse.
        assert len(_compress_dwells(times)) == 2

    def test_find_break_start_catches_window_break(self):
        # 11:30 → 12:20 is a 50-min gap starting inside 10:00-14:00 break window.
        # Prior gap 10:30 → 11:30 is also ≥40min — function returns the FIRST
        # qualifying gap's start, which is 10:30.
        visits = [_ts(9, 0), _ts(10, 30), _ts(11, 30), _ts(12, 20)]
        bs = _find_break_start(visits)
        assert bs is not None
        assert bs.hour == 10 and bs.minute == 30

    def test_find_break_start_requires_gap_above_threshold(self):
        # 30-min gap from 11:00 to 11:30 — below BREAK_GAP_MIN=40.
        visits = [_ts(10, 30), _ts(11, 0), _ts(11, 30), _ts(11, 55)]
        assert _find_break_start(visits) is None

    def test_find_break_start_outside_window_ignored(self):
        # 50-min gap at 05:00 local (outside 10:00-14:00).
        visits = [_ts(5, 0), _ts(5, 55), _ts(13, 0)]
        assert _find_break_start(visits) is None

    def test_morning_signature_pads_with_nan(self):
        # Only 2 morning visits; should pad to SIGNATURE_LEN=6 with NaN.
        visits = [_ts(7, 10), _ts(7, 25)]
        sig = _morning_signature(visits)
        assert len(sig) == 6
        assert sig[0] == 10  # 7:10 - 7:00 = 10
        assert sig[1] == 25
        assert all(pd.isna(x) for x in sig[2:])

    def test_cluster_center_produces_sorted_archetypes(self):
        # Two synthetic break clusters at 10:30 and 13:00.
        candidates = [
            {"vehicle_id": "A", "break_start_min": 630, "signature": [0.0] * 6},
            {"vehicle_id": "B", "break_start_min": 635, "signature": [1.0] * 6},
            {"vehicle_id": "C", "break_start_min": 780, "signature": [5.0] * 6},
            {"vehicle_id": "D", "break_start_min": 785, "signature": [6.0] * 6},
        ]
        archetypes = _cluster_center_with_signature(candidates, k=2)
        assert len(archetypes) == 2
        # Ordered by break_start_min ascending.
        assert archetypes[0]["break_start_min"] < archetypes[1]["break_start_min"]
        assert archetypes[0]["member_count"] >= 2
        assert archetypes[1]["member_count"] >= 2


# ---------- Matcher ----------


class TestMatcher:
    def test_signature_cost_uses_rmse_over_observed(self):
        vs = np.array([10.0, 20.0, float("nan"), float("nan"), float("nan"), float("nan")])
        arch = np.array([12.0, 22.0, 30.0, 40.0, 50.0, 60.0])
        # RMSE over the 2 observed components: sqrt((4+4)/2) = 2.0
        assert _signature_cost(vs, arch) == pytest.approx(2.0)

    def test_signature_cost_all_nan_returns_inf(self):
        vs = np.full(6, float("nan"))
        arch = np.zeros(6)
        assert _signature_cost(vs, arch) == float("inf")

    def test_hungarian_assigns_optimal_pairs(self):
        # Two vehicles, two archetypes, clean separation.
        vsigs = {
            "V1": np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            "V2": np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0]),
        }
        archetypes = [
            {"archetype_id": "early", "break_start_min": 660,
             "signature": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]},
            {"archetype_id": "late", "break_start_min": 780,
             "signature": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]},
        ]
        matches = match_signatures(vsigs, archetypes)
        assert matches["V1"]["archetype_id"] == "early"
        assert matches["V2"]["archetype_id"] == "late"

    def test_hungarian_unmatches_when_cost_exceeds_threshold(self):
        # Both vehicles far from the single archetype -> cost > MAX_MATCH_RMSE.
        vsigs = {
            "V1": np.array([0.0] * 6),
            "V2": np.array([500.0] * 6),  # 500 min off — way above 30.
        }
        archetypes = [
            {"archetype_id": "mid", "break_start_min": 700,
             "signature": [200.0] * 6},
        ]
        matches = match_signatures(vsigs, archetypes)
        # V1 is 200 off → above MAX_MATCH_RMSE; V2 is 300 off → above.
        # Both should be unmatched.
        assert matches == {}

    def test_hungarian_handles_more_vehicles_than_archetypes(self):
        # 3 vehicles, 1 archetype. One matches; two are unmatched via pad column.
        vsigs = {
            "best": np.array([5.0] * 6),
            "far1": np.array([100.0] * 6),
            "far2": np.array([200.0] * 6),
        }
        archetypes = [{"archetype_id": "A", "break_start_min": 700, "signature": [5.0] * 6}]
        matches = match_signatures(vsigs, archetypes)
        assert "best" in matches
        assert "far1" not in matches
        assert "far2" not in matches


# ---------- predict_on_break (integration-ish, stubbed) ----------


class _StubSession:
    """Minimal async session stub."""
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


def _union_ping(vid: str, ts_utc: datetime, lat: float = 42.730711, lon: float = -73.676737):
    return (vid, ts_utc, lat, lon)


class TestPredictOnBreak:
    def setup_method(self):
        reset_archetype_cache()

    def teardown_method(self):
        reset_archetype_cache()

    @pytest.mark.asyncio
    async def test_no_archetypes_no_visits_returns_false(self, monkeypatch):
        # No archetype file loaded, no visits — predict should be False.
        _ARCHETYPES_CACHE[0] = []  # force dow=0 to empty archetypes

        # Freeze dev_now to a Monday 12:00 local (16:00 UTC).
        fixed_utc = datetime(2025, 9, 22, 16, 0, tzinfo=timezone.utc)  # Mon 12:00 ET
        monkeypatch.setattr(
            "backend.fastapi.break_archetype.dev_now",
            lambda tz=None: fixed_utc if tz else fixed_utc.replace(tzinfo=None),
        )
        monkeypatch.setattr(
            "backend.fastapi.break_archetype.get_campus_start_of_day",
            lambda: datetime(2025, 9, 22, 4, 0, tzinfo=timezone.utc),  # 00:00 ET
        )

        sf = _StubSessionFactory([])
        result = await predict_on_break(["V1"], sf, TZ)
        assert result == {"V1": False}

    @pytest.mark.asyncio
    async def test_fallback_40min_gap_fires_during_day(self, monkeypatch):
        _ARCHETYPES_CACHE[0] = []  # archetypes absent → only fallback path runs

        # Mon 12:30 local; last Union visit at 11:45 local (45 min ago).
        fixed_utc = datetime(2025, 9, 22, 16, 30, tzinfo=timezone.utc)
        last_union_utc = datetime(2025, 9, 22, 15, 45, tzinfo=timezone.utc)

        monkeypatch.setattr(
            "backend.fastapi.break_archetype.dev_now",
            lambda tz=None: fixed_utc if tz else fixed_utc.replace(tzinfo=None),
        )
        monkeypatch.setattr(
            "backend.fastapi.break_archetype.get_campus_start_of_day",
            lambda: datetime(2025, 9, 22, 4, 0, tzinfo=timezone.utc),
        )

        sf = _StubSessionFactory([_union_ping("V1", last_union_utc)])
        result = await predict_on_break(["V1"], sf, TZ)
        assert result["V1"] is True

    @pytest.mark.asyncio
    async def test_fallback_does_not_fire_at_night(self, monkeypatch):
        _ARCHETYPES_CACHE[0] = []

        # Mon 22:00 local = outside FALLBACK_ACTIVE window (8-20).
        fixed_utc = datetime(2025, 9, 23, 2, 0, tzinfo=timezone.utc)  # Tue 02:00 UTC = Mon 22:00 ET
        last_union_utc = datetime(2025, 9, 23, 1, 0, tzinfo=timezone.utc)  # 1h ago

        monkeypatch.setattr(
            "backend.fastapi.break_archetype.dev_now",
            lambda tz=None: fixed_utc if tz else fixed_utc.replace(tzinfo=None),
        )
        monkeypatch.setattr(
            "backend.fastapi.break_archetype.get_campus_start_of_day",
            lambda: datetime(2025, 9, 22, 4, 0, tzinfo=timezone.utc),
        )

        sf = _StubSessionFactory([_union_ping("V1", last_union_utc)])
        result = await predict_on_break(["V1"], sf, TZ)
        assert result["V1"] is False

    @pytest.mark.asyncio
    async def test_archetype_prediction_fires_near_break_start(self, monkeypatch):
        # Mon archetype predicts break at 12:00 local (720 min). Now = 11:58 local.
        _ARCHETYPES_CACHE[0] = [
            {"archetype_id": "noon", "break_start_min": 720,
             "signature": [5.0, 15.0, 25.0, 35.0, 45.0, 55.0]},
        ]

        fixed_utc = datetime(2025, 9, 22, 15, 58, tzinfo=timezone.utc)  # Mon 11:58 ET

        monkeypatch.setattr(
            "backend.fastapi.break_archetype.dev_now",
            lambda tz=None: fixed_utc if tz else fixed_utc.replace(tzinfo=None),
        )
        monkeypatch.setattr(
            "backend.fastapi.break_archetype.get_campus_start_of_day",
            lambda: datetime(2025, 9, 22, 4, 0, tzinfo=timezone.utc),
        )

        # Give V1 a matching morning signature: visits at 07:05, 07:15,... etc
        # so the extracted signature lands near the archetype's.
        morning_visits = [
            datetime(2025, 9, 22, 11, 5, tzinfo=timezone.utc),   # 07:05 ET
            datetime(2025, 9, 22, 11, 15, tzinfo=timezone.utc),  # 07:15
            datetime(2025, 9, 22, 11, 25, tzinfo=timezone.utc),
            datetime(2025, 9, 22, 11, 35, tzinfo=timezone.utc),
            datetime(2025, 9, 22, 11, 45, tzinfo=timezone.utc),
            datetime(2025, 9, 22, 11, 55, tzinfo=timezone.utc),
        ]
        rows = [_union_ping("V1", t) for t in morning_visits]
        sf = _StubSessionFactory(rows)
        result = await predict_on_break(["V1"], sf, TZ)
        assert result["V1"] is True
