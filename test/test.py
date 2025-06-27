from flask import Flask, request, jsonify
import threading
import time
import requests
from datetime import datetime, timezone

app = Flask(__name__)

# State to simulate "after" tokens and pagination
fake_data = {
    "281474979957434": {
        "name": "Shuttle A",
        "gps": {
            "latitude": 42.730676958536144,
            "longitude": -73.67674616623393,
            "time": "2025-06-16T00:00:00Z",
            "speedMilesPerHour": 25.0,
            "headingDegrees": 90,
            "reverseGeo": {"formattedLocation": "Union"}
        }
    },
    "281474993785467": {
        "name": "Shuttle B",
        "gps": {
            "latitude": 42.730318398121575,
            "longitude": -73.67656636425313,
            "time": "2025-06-16T00:00:01Z",
            "speedMilesPerHour": 15.0,
            "headingDegrees": 180,
            "reverseGeo": {"formattedLocation": "Academy"}
        }
    }
}

@app.route('/fleet/vehicles/stats/feed')
def mock_feed():
    vehicle_ids = request.args.get('vehicleIds', '').split(',')
    after = request.args.get('after')

    print(f'[MOCK API] Received stats request for vehicles {vehicle_ids} after={after}')

    # update timestamps
    new_timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')
    for vehicle_id in fake_data:
        fake_data[vehicle_id]['gps']['time'] = new_timestamp

    data = []
    for vehicle_id in vehicle_ids:
        if vehicle_id in fake_data:
            data.append({
                'id': vehicle_id,
                'name': fake_data[vehicle_id]['name'],
                'gps': [fake_data[vehicle_id]['gps']]
            })

    return jsonify({
        'data': data,
        'pagination': {
            'hasNextPage': False,
            'endCursor': 'fake-token-next'
        }
    })

def send_webhook(vehicle_id, entry=True):
    url = 'http://localhost:8000/api/webhook'
    headers = {'Content-Type': 'application/json'}
    vehicle = {
        'id': vehicle_id,
        'name': fake_data[vehicle_id]['name'],
        'licensePlate': 'FAKE123',
        'vin': '1HGCM82633A004352',
        'assetType': 'vehicle',
        'externalIds': {
            'maintenanceId': '250020'
        },
        'gateway': {
            'model': 'VG34',
            'serial': 'GFRV-43N-VGX'
        }
    }
    address = {
        'id': '123456',
        'name': 'Test Location',
        'formattedAddress': fake_data[vehicle_id]['gps']['reverseGeo']['formattedLocation'],
        'externalIds': {
            'siteId': '54'
        },
    }
    payload = {
        'eventId': str(uuid.uuid4()),
        'eventTime': datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
        'eventType': 'GeofenceEntry' if entry else 'GeofenceExit',
        'orgId': 20936,
        'webhookId': '1411751028848270',
        'data': {
            'vehicle': vehicle,
            'address': address
        }
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        print(f'[WEBHOOK] Sent {payload["eventType"]} for {vehicle_id}: {response.status_code}')
    except Exception as e:
        print(f'[WEBHOOK] Failed: {e}')

def run_mock_api():
    app.run(host='0.0.0.0', port=4000)

def main():
    time.sleep(2)  # wait for main app to start
    send_webhook('281474979957434', True)
    time.sleep(2)
    send_webhook('281474993785467', True)

if __name__ == '__main__':
    threading.Thread(target=main, daemon=True).start()
    run_mock_api()
