"""Swappable velocity prediction interface.

Current implementation: constant average speed.
To swap to LSTM-based prediction, implement VelocityPredictor protocol
and change get_velocity_predictor() to return the new implementation.
"""

from typing import Protocol


class VelocityPredictor(Protocol):
    """Interface for velocity prediction strategies."""

    def predict_speed_kmh(self, vehicle_id: str, route: str) -> float:
        """Return predicted speed in km/h for a vehicle on a route."""
        ...


class AverageSpeedPredictor:
    """Simple constant average speed predictor.

    Campus shuttles average ~20 km/h in urban/campus setting.
    Replace with LSTMSpeedPredictor when ready.
    """

    DEFAULT_SPEED_KMH = 20.0

    def predict_speed_kmh(self, vehicle_id: str, route: str) -> float:
        return self.DEFAULT_SPEED_KMH


_VELOCITY_PREDICTOR_SINGLETON: VelocityPredictor | None = None


def get_velocity_predictor() -> VelocityPredictor:
    """Factory function. Change this single line to swap implementations.

    Singleton so the per-vehicle ETA loop doesn't allocate a new predictor
    every call — free CPU cycles for no behavioral change.
    """
    global _VELOCITY_PREDICTOR_SINGLETON
    if _VELOCITY_PREDICTOR_SINGLETON is None:
        _VELOCITY_PREDICTOR_SINGLETON = AverageSpeedPredictor()
    return _VELOCITY_PREDICTOR_SINGLETON
