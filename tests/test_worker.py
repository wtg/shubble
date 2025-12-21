"""
Unit tests for worker functions
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from server.worker import get_vehicles_in_geofence
from server.models import GeofenceEvent, Vehicle


@pytest.mark.unit
def test_get_vehicles_in_geofence_empty(app, db_session):
    """Test get_vehicles_in_geofence returns empty set when no events"""
    with app.app_context():
        result = get_vehicles_in_geofence()
        assert isinstance(result, set)
        assert len(result) == 0


@pytest.mark.unit
def test_get_vehicles_in_geofence_with_entry(app, db_session):
    """Test get_vehicles_in_geofence returns vehicles with entry events"""
    with app.app_context():
        # Create vehicle first
        vehicle = Vehicle(
            id="vehicle_1",
            name="Test Shuttle"
        )
        db_session.add(vehicle)

        # Create entry event
        event = GeofenceEvent(
            id="event_1",
            vehicle_id="vehicle_1",
            event_type="geofenceEntry",
            event_time=datetime.now(timezone.utc)
        )
        db_session.add(event)
        db_session.commit()

        result = get_vehicles_in_geofence()
        assert "vehicle_1" in result


@pytest.mark.unit
def test_get_vehicles_in_geofence_with_exit(app, db_session):
    """Test get_vehicles_in_geofence excludes vehicles with exit events"""
    with app.app_context():
        # Create vehicle first
        vehicle = Vehicle(
            id="vehicle_1",
            name="Test Shuttle"
        )
        db_session.add(vehicle)

        # Create entry event
        entry_event = GeofenceEvent(
            id="event_1",
            vehicle_id="vehicle_1",
            event_type="geofenceEntry",
            event_time=datetime.now(timezone.utc)
        )
        db_session.add(entry_event)

        # Create later exit event
        exit_event = GeofenceEvent(
            id="event_2",
            vehicle_id="vehicle_1",
            event_type="geofenceExit",
            event_time=datetime.now(timezone.utc)
        )
        db_session.add(exit_event)
        db_session.commit()

        result = get_vehicles_in_geofence()
        assert "vehicle_1" not in result
