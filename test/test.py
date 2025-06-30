from flask import Flask, request, jsonify
import threading
import time
import requests

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
    url = 'http://localhost:3000/api/webhook'
    headers = {'Content-Type': 'application/json'}
    vehicle = {
        'id': vehicle_id,
        'name': fake_data[vehicle_id]['name']
    }
    data = {
        'data': {
            'vehicle': vehicle
        },
        'eventType': 'GeofenceEntry' if entry else 'GeofenceExit',
        'eventTime': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        print(f'[WEBHOOK] Sent {data["eventType"]} for {vehicle_id}: {response.status_code}')
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
