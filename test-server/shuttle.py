import time
from enum import Enum
import random
from data.stops import Stops
import uuid
from datetime import datetime, timezone
import requests
import logging

logger = logging.getLogger(__name__)


class ShuttleState(Enum):
    WAITING = "waiting"
    ENTERING = "entering"
    LOOPING = "looping"
    ON_BREAK = "on_break"
    EXITING = "exiting"

class Shuttle:
    def __init__(self, shuttle_id: str):
        # larger state
        self.id = shuttle_id
        self.state = ShuttleState.WAITING
        self.next_state = ShuttleState.WAITING

        # shuttle properties
        self.last_updated = time.time()
        self.location = (42.730711, -73.676737)
        self.speed = 0.0002
        self.loop = random.choice(list(Stops.routes_data.keys()))

        self.path = []
        self.path_index = 0
        self.subpath_index = 0
        self.distance_into_segment = 0

    def update_state(self):
        match self.state:
            case ShuttleState.WAITING:
                if self.next_state is None:
                    return
                self.go_to_next_state()
            case _:
                if not self.follow_path():
                    self.go_to_next_state()

        self.last_updated = time.time()

    def go_to_next_state(self):
        self.state = self.next_state
        self.next_state = ShuttleState.WAITING
        match self.state:
            case ShuttleState.ENTERING:
                self.send_webhook(entry=True)
                self.path = self.get_entering_path()
            case ShuttleState.LOOPING:
                self.path = self.get_looping_path()
            case ShuttleState.ON_BREAK:
                self.path = self.get_break_path()
            case ShuttleState.EXITING:
                self.send_webhook(entry=False)
                self.path = self.get_exiting_path()
        self.path_index = 0
        self.distance_into_segment = 0

    def follow_path(self):
        if self.path_index >= len(self.path):
            return False

        travel_distance = self.speed * (time.time() - self.last_updated)

        start = self.path[self.path_index][self.subpath_index]
        end = self.path[self.path_index][self.subpath_index + 1]

        segment_vector = (end[0] - start[0], end[1] - start[1])
        segment_length = (segment_vector[0]**2 + segment_vector[1]**2) ** 0.5 - self.distance_into_segment

        while travel_distance > segment_length:
            travel_distance -= segment_length
            self.subpath_index += 1
            # move to next path segment if needed
            if self.subpath_index >= len(self.path[self.path_index]) - 1:
                self.path_index += 1
                self.subpath_index = 0

            self.distance_into_segment = 0

            if self.path_index >= len(self.path):
                # reached end of path
                self.location = tuple(self.path[-1][-1])
                return False

            start = self.path[self.path_index][self.subpath_index]
            end = self.path[self.path_index][self.subpath_index + 1]
            segment_vector = (end[0] - start[0], end[1] - start[1])
            segment_length = (segment_vector[0]**2 + segment_vector[1]**2) ** 0.5

        self.distance_into_segment += travel_distance

        segment_unit_vector = (segment_vector[0] / segment_length, segment_vector[1] / segment_length)
        self.location = (start[0] + segment_unit_vector[0] * self.distance_into_segment,
                         start[1] + segment_unit_vector[1] * self.distance_into_segment)
        return True

    def send_webhook(self, entry=True):
        url = 'http://localhost:8000/api/webhook'
        headers = {'Content-Type': 'application/json'}

        vehicle = {
            'id': self.id,
            'name': self.id[-3:],
            'licensePlate': f'FAKE{self.id[-3:]}',
            'vin': self.id[-3:],
            'assetType': 'vehicle',
            'externalIds': {
                'maintenanceId': self.id[-3:]
            },
            'gateway': {
                'model': 'VG34',
                'serial': self.id[-3:]
            }
        }

        address = {
            'id': '123456',
            'name': 'Test Location',
            'formattedAddress': 'Test Address',
            'externalIds': {
                'siteId': '54'
            },
            'geofence': {
                'id': 'geofence123',
                'name': 'Test Geofence',
                'polygon': {
                    'vertices': [
                        {
                            'latitude': self.location[0],
                            'longitude': self.location[1]
                        }
                    ]
                }
            }
        }

        geofence_key = 'geofenceEntry' if entry else 'geofenceExit'

        payload = {
            'eventId': str(uuid.uuid4()),
            'eventTime': datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
            'eventType': 'geofenceEntry' if entry else 'geofenceExit',
            'orgId': 20936,
            'webhookId': '1411751028848270',
            'data': {
                'conditions': [
                    {
                        'details': {
                            geofence_key: {
                                'vehicle': vehicle,
                                'address': address
                            }
                        }
                    }
                ]
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                logger.info(f'[WEBHOOK] Sent {payload["eventType"]} for {self.id}: {response.status_code}')
                return True
            else:
                logger.error(f'[WEBHOOK] Failed to send {payload["eventType"]} for {self.id}: {response.status_code}')
                return False
        except Exception as e:
            logger.error(f'[WEBHOOK] Exception sending webhook for {self.id}: {e}')
            return False

    def get_entering_path(self):
        return []

    def get_looping_path(self):
        return Stops.routes_data[self.loop]['ROUTES']

    def get_break_path(self):
        return []

    def get_exiting_path(self):
        return []

    def set_next_state(self, next_state: ShuttleState):
        if next_state not in ShuttleState:
            raise ValueError(f"Invalid shuttle state: {next_state}")
        self.next_state = next_state

    def to_dict(self):
        return {
            "id": self.id,
            "state": self.state.value,
            "next_state": self.next_state.value,
            "location": self.location,
            "last_updated": self.last_updated,
            "speed": self.speed,
            "loop": self.loop,
            "path_index": self.path_index,
            "subpath_index": self.subpath_index,
            "distance_into_segment": self.distance_into_segment
        }
