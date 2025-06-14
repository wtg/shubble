import requests
import time

def send_webhook(vehicle, entry):
    url = 'http://localhost:3000/api/webhook'
    headers = {'Content-Type': 'application/json'}
    data = {
        'data': {
            'vehicle': {
                'id': vehicle,
            },
        },
        'eventType': 'GeofenceEntry' if entry else 'GeofenceExit',
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print(f'Successfully sent webhook for vehicle {vehicle}, event: {"GeofenceEntry" if entry else "GeofenceExit"}')
        else:
            print(f'Error sending webhook for vehicle {vehicle}, event: {"GeofenceEntry" if entry else "GeofenceExit"}: {response.status_code} {response.text}')
    except requests.RequestException as e:
        print(f'Error sending webhook for vehicle {vehicle} with entry={entry}: {e}')

def main():
    time.sleep(10)
    # send '281474979957434' through webhook
    send_webhook('281474979957434', True)
    time.sleep(10)
    # send '281474993785467' through webhook
    send_webhook('281474993785467', True)
    time.sleep(10)
    send_webhook('281474979957434', False)
    time.sleep(10)
    send_webhook('281474993785467', False)

if __name__ == '__main__':
    main()

