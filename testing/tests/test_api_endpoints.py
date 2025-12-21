"""
Integration tests for API endpoints
"""
import pytest
from datetime import datetime, timezone
from backend.models import Vehicle, GeofenceEvent, VehicleLocation


@pytest.mark.integration
def test_get_locations_empty(client, db_session):
    """Test /api/locations returns empty dict when no vehicles in geofence"""
    response = client.get('/api/locations')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)
    assert len(data) == 0


@pytest.mark.integration
def test_get_locations_with_vehicle(client, db_session):
    """Test /api/locations returns vehicle data when vehicle in geofence"""
    # Create a vehicle
    vehicle = Vehicle(
        id="test_vehicle_1",
        name="Test Shuttle",
        license_plate="ABC123",
        vin="1234567890"
    )
    db_session.add(vehicle)

    # Create geofence entry event
    entry_event = GeofenceEvent(
        id="event_1",
        vehicle_id="test_vehicle_1",
        event_type="geofenceEntry",
        event_time=datetime.now(timezone.utc)
    )
    db_session.add(entry_event)

    # Create location
    location = VehicleLocation(
        vehicle_id="test_vehicle_1",
        timestamp=datetime.now(timezone.utc),
        name="Test Shuttle",
        latitude=42.7284,
        longitude=-73.6918,
        heading_degrees=90.0,
        speed_mph=15.0
    )
    db_session.add(location)
    db_session.commit()

    response = client.get('/api/locations')
    assert response.status_code == 200
    data = response.get_json()

    assert "test_vehicle_1" in data
    assert data["test_vehicle_1"]["name"] == "Test Shuttle"
    assert data["test_vehicle_1"]["latitude"] == 42.7284
    assert data["test_vehicle_1"]["longitude"] == -73.6918


@pytest.mark.integration
def test_get_routes(client):
    """Test /api/routes returns route data"""
    response = client.get('/api/routes')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)


@pytest.mark.integration
def test_get_schedule(client):
    """Test /api/schedule returns schedule data"""
    response = client.get('/api/schedule')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)


@pytest.mark.integration
def test_get_aggregated_schedule(client):
    """Test /api/aggregated-schedule returns aggregated schedule"""
    response = client.get('/api/aggregated-schedule')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, list)
    assert len(data) == 7  # 7 days of the week


@pytest.mark.integration
def test_webhook_invalid_json(client):
    """Test webhook rejects invalid JSON"""
    response = client.post(
        '/api/webhook',
        data='invalid json',
        content_type='application/json'
    )
    assert response.status_code == 400


@pytest.mark.integration
def test_webhook_missing_data(client):
    """Test webhook rejects missing required data"""
    response = client.post(
        '/api/webhook',
        json={},
        content_type='application/json'
    )
    assert response.status_code == 400


@pytest.mark.integration
def test_webhook_geofence_entry(client, db_session):
    """Test webhook creates geofence entry event"""
    webhook_data = {
        "eventId": "test_event_1",
        "eventTime": "2024-01-15T12:00:00Z",
        "data": {
            "conditions": [{
                "details": {
                    "geofenceEntry": {
                        "vehicle": {
                            "id": "vehicle_123",
                            "name": "Shuttle 1",
                            "assetType": "vehicle",
                            "licensePlate": "ABC123"
                        },
                        "address": {
                            "name": "Campus",
                            "formattedAddress": "123 Main St",
                            "geofence": {
                                "polygon": {
                                    "vertices": [
                                        {"latitude": 42.7284, "longitude": -73.6918}
                                    ]
                                }
                            }
                        }
                    }
                }
            }]
        }
    }

    response = client.post(
        '/api/webhook',
        json=webhook_data,
        content_type='application/json'
    )
    assert response.status_code == 200

    # Verify vehicle and event were created
    vehicle = Vehicle.query.get("vehicle_123")
    assert vehicle is not None
    assert vehicle.name == "Shuttle 1"

    event = GeofenceEvent.query.get("test_event_1")
    assert event is not None
    assert event.event_type == "geofenceEntry"
    assert event.vehicle_id == "vehicle_123"


@pytest.mark.integration
def test_matched_schedules_cache(client, db_session):
    """Test /api/matched-schedules uses cache"""
    # First request should compute
    response1 = client.get('/api/matched-schedules')
    assert response1.status_code == 200
    data1 = response1.get_json()
    assert data1['status'] == 'success'

    # Second request should use cache
    response2 = client.get('/api/matched-schedules')
    assert response2.status_code == 200
    data2 = response2.get_json()
    assert data2['status'] == 'success'
    assert data2.get('source') == 'cache'


@pytest.mark.integration
def test_matched_schedules_force_recompute(client, db_session):
    """Test /api/matched-schedules force recompute"""
    response = client.get('/api/matched-schedules?force_recompute=true')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'success'
    assert data['source'] == 'recomputed'


@pytest.mark.integration
def test_today_endpoint(client, db_session):
    """Test /api/today returns today's data"""
    # Create a vehicle
    vehicle = Vehicle(
        id="test_vehicle_1",
        name="Test Shuttle"
    )
    db_session.add(vehicle)

    # Create location
    location = VehicleLocation(
        vehicle_id="test_vehicle_1",
        timestamp=datetime.now(timezone.utc),
        name="Test Shuttle",
        latitude=42.7284,
        longitude=-73.6918
    )
    db_session.add(location)
    db_session.commit()

    response = client.get('/api/today')
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)
