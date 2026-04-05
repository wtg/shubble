"""Tests for the velocity prediction abstraction."""

from backend.worker.velocity import (
    AverageSpeedPredictor,
    VelocityPredictor,
    get_velocity_predictor,
)


def test_average_speed_predictor_returns_positive():
    predictor = AverageSpeedPredictor()
    speed = predictor.predict_speed_kmh("vehicle_123", "NORTH")
    assert speed > 0


def test_average_speed_predictor_consistent():
    predictor = AverageSpeedPredictor()
    speed1 = predictor.predict_speed_kmh("v1", "NORTH")
    speed2 = predictor.predict_speed_kmh("v2", "WEST")
    # Average speed predictor returns the same value regardless of vehicle/route
    assert speed1 == speed2


def test_get_velocity_predictor_factory():
    predictor = get_velocity_predictor()
    assert predictor is not None
    assert hasattr(predictor, "predict_speed_kmh")


def test_velocity_predictor_protocol_compliance():
    """Verify AverageSpeedPredictor satisfies VelocityPredictor protocol."""
    predictor = AverageSpeedPredictor()
    # Protocol requires predict_speed_kmh(vehicle_id: str, route: str) -> float
    result = predictor.predict_speed_kmh("any_id", "any_route")
    assert isinstance(result, float)


def test_default_speed_is_reasonable():
    """Campus shuttle should average 15-30 km/h."""
    predictor = AverageSpeedPredictor()
    speed = predictor.predict_speed_kmh("v1", "NORTH")
    assert 10.0 <= speed <= 40.0, f"Default speed {speed} km/h seems unreasonable for campus shuttle"
