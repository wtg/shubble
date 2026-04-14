"""Tests for 'scrub live ETA behind the shuttle frontier' in build_trip_etas.

Bug 2 of quick task 260414-kg2: live ETA countdowns lingered on stops
the shuttle had physically moved past, because the OFFSET-diff
propagation in compute_per_stop_etas emitted entries in `eta_lookup`
for stops behind the predictor's frontier. Those entries had past
timestamps but still got surfaced as `entry["eta"]`, producing UI
countdowns like "-1:23" on stops the shuttle already drove past with
no detection.

Fix: in `build_trip_etas`, gate the `stop_key in eta_lookup` branch on
the stop's index being at or after the first stop with a strictly-future
predicted eta (the "frontier"). Stops before the frontier with no
detection get eta=None.
"""

from datetime import datetime, timezone, timedelta

from backend.worker.trips import build_trip_etas


STOPS = ["S0", "S1", "S2", "S3", "S4"]
TRIP = {"trip_id": "t", "route": "TEST", "vehicle_id": "v1", "status": "active"}


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def test_stop_before_frontier_has_eta_cleared():
    """A stop whose predictor eta is in the PAST (no detection) must
    appear with eta=None, not the past-iso leaked onto the UI.
    """
    now = datetime(2026, 4, 14, 17, 0, tzinfo=timezone.utc)
    vehicle_stops = [
        ("S2", now - timedelta(seconds=30)),  # past eta (no detection)
        ("S3", now + timedelta(seconds=30)),  # frontier (first future)
        ("S4", now + timedelta(seconds=120)),
    ]
    result = build_trip_etas(
        trip=TRIP,
        vehicle_stops=vehicle_stops,
        last_arrivals={},
        stops_in_route=STOPS,
        now_utc=now,
        loop_cutoff=None,
    )
    # S2 has a past eta in eta_lookup. With the scrub, its entry["eta"]
    # must be None because it's before the frontier (S3).
    assert result["S2"]["eta"] is None, (
        f"Past-eta stop S2 leaked live countdown: {result['S2']}"
    )


def test_stop_at_or_after_frontier_keeps_eta():
    """Stops at or after the frontier retain their future eta."""
    now = datetime(2026, 4, 14, 17, 0, tzinfo=timezone.utc)
    s3_eta = now + timedelta(seconds=30)
    s4_eta = now + timedelta(seconds=120)
    vehicle_stops = [
        ("S2", now - timedelta(seconds=30)),
        ("S3", s3_eta),
        ("S4", s4_eta),
    ]
    result = build_trip_etas(
        trip=TRIP,
        vehicle_stops=vehicle_stops,
        last_arrivals={},
        stops_in_route=STOPS,
        now_utc=now,
        loop_cutoff=None,
    )
    assert result["S3"]["eta"] == s3_eta.isoformat()
    assert result["S4"]["eta"] == s4_eta.isoformat()


def test_frontier_none_preserves_legacy_behavior():
    """If no future ETAs exist (frontier is None) AND eta_lookup is empty,
    stops without a detection end up with eta=None. No crash. This is
    the legacy path for trips that have no live ETAs (e.g. before the
    shuttle starts moving).
    """
    now = datetime(2026, 4, 14, 17, 0, tzinfo=timezone.utc)
    result = build_trip_etas(
        trip=TRIP,
        vehicle_stops=[],  # no ETAs at all
        last_arrivals={},
        stops_in_route=STOPS,
        now_utc=now,
        loop_cutoff=None,
    )
    for s in STOPS:
        assert result[s]["eta"] is None, f"{s} unexpected eta {result[s]}"
        assert result[s]["last_arrival"] is None
        assert result[s]["passed"] is False


def test_detection_branch_still_sets_eta_null():
    """Stops with a real last_arrival must still be handled by the
    existing last_arrivals branch (eta=None, passed=True, last_arrival=iso).
    """
    now = datetime(2026, 4, 14, 17, 0, tzinfo=timezone.utc)
    vehicle_stops = [
        ("S3", now + timedelta(seconds=30)),
        ("S4", now + timedelta(seconds=120)),
    ]
    detected_ts = now - timedelta(seconds=10)
    last_arrivals = {"S2": detected_ts.isoformat()}
    result = build_trip_etas(
        trip=TRIP,
        vehicle_stops=vehicle_stops,
        last_arrivals=last_arrivals,
        stops_in_route=STOPS,
        now_utc=now,
        loop_cutoff=None,
    )
    assert result["S2"]["eta"] is None
    assert result["S2"]["passed"] is True
    assert result["S2"]["last_arrival"] == detected_ts.isoformat()
