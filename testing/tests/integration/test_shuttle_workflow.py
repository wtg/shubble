"""
Integration tests for complete shuttle tracking workflow
"""
import pytest
from datetime import datetime, timezone
from server.models import Vehicle, GeofenceEvent, VehicleLocation


@pytest.mark.integration
@pytest.mark.slow
def test_complete_shuttle_tracking_workflow(client, db_session):
    """
    Test complete workflow:
    1. Vehicle enters geofence (webhook)
    2. Location data is stored
    3. API returns vehicle location
    4. Vehicle exits geofence (webhook)
    5. API no longer returns vehicle
    """
    vehicle_id = "test_shuttle_workflow"

    # Step 1: Send webhook for geofence entry
    entry_webhook = {
        "eventId": "entry_event_1",
        "eventTime": datetime.now(timezone.utc).isoformat(),
        "data": {
            "conditions": [{
                "details": {
                    "geofenceEntry": {
                        "vehicle": {
                            "id": vehicle_id,
                            "name": "Test Shuttle",
                            "assetType": "vehicle",
                            "licensePlate": "TEST123"
                        },
                        "address": {
                            "name": "Campus",
                            "geofence": {
                                "polygon": {
                                    "vertices": [{"latitude": 42.7284, "longitude": -73.6918}]
                                }
                            }
                        }
                    }
                }
            }]
        }
    }

    response = client.post('/api/webhook', json=entry_webhook)
    assert response.status_code == 200

    # Step 2: Add location data
    location = VehicleLocation(
        vehicle_id=vehicle_id,
        timestamp=datetime.now(timezone.utc),
        name="Test Shuttle",
        latitude=42.7284,
        longitude=-73.6918,
        heading_degrees=90.0,
        speed_mph=15.0
    )
    db_session.add(location)
    db_session.commit()

    # Step 3: Verify vehicle appears in /api/locations
    response = client.get('/api/locations')
    assert response.status_code == 200
    data = response.get_json()
    assert vehicle_id in data
    assert data[vehicle_id]["name"] == "Test Shuttle"

    # Step 4: Send webhook for geofence exit
    exit_webhook = {
        "eventId": "exit_event_1",
        "eventTime": datetime.now(timezone.utc).isoformat(),
        "data": {
            "conditions": [{
                "details": {
                    "geofenceExit": {
                        "vehicle": {
                            "id": vehicle_id,
                            "name": "Test Shuttle"
                        },
                        "address": {
                            "name": "Campus",
                            "geofence": {
                                "polygon": {
                                    "vertices": [{"latitude": 42.7284, "longitude": -73.6918}]
                                }
                            }
                        }
                    }
                }
            }]
        }
    }

    response = client.post('/api/webhook', json=exit_webhook)
    assert response.status_code == 200

    # Step 5: Verify vehicle no longer in /api/locations
    # Note: This requires cache invalidation to work properly
    response = client.get('/api/locations')
    assert response.status_code == 200
    data = response.get_json()
    # Vehicle should not appear since latest event is exit
    assert vehicle_id not in data


@pytest.mark.integration
def test_multiple_vehicles_tracking(client, db_session):
    """Test tracking multiple vehicles simultaneously"""
    vehicles = ["vehicle_1", "vehicle_2", "vehicle_3"]

    for i, vehicle_id in enumerate(vehicles):
        # Create vehicle
        vehicle = Vehicle(
            id=vehicle_id,
            name=f"Shuttle {i+1}"
        )
        db_session.add(vehicle)

        # Create entry event
        event = GeofenceEvent(
            id=f"event_{i+1}",
            vehicle_id=vehicle_id,
            event_type="geofenceEntry",
            event_time=datetime.now(timezone.utc)
        )
        db_session.add(event)

        # Create location
        location = VehicleLocation(
            vehicle_id=vehicle_id,
            timestamp=datetime.now(timezone.utc),
            name=f"Shuttle {i+1}",
            latitude=42.7284 + (i * 0.001),
            longitude=-73.6918 + (i * 0.001)
        )
        db_session.add(location)

    db_session.commit()

    # Verify all vehicles appear
    response = client.get('/api/locations')
    assert response.status_code == 200
    data = response.get_json()

    for vehicle_id in vehicles:
        assert vehicle_id in data
