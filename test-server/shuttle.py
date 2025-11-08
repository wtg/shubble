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
    def __init__(self, shuttle_id: str, latitude: float = 0, longitude: float = 0):
        # larger state
        self.id = shuttle_id
        self.state = ShuttleState.WAITING
        self.next_state = ShuttleState.WAITING

        # shuttle properties
        self.last_updated = time.time()
        self.location = (latitude, longitude)
        self.speed = 0.0002

        self.path = []
        self.path_index = 0
        self.subpath_index = 0
        self.distance_into_segment = 0
        self.current_route = None

    def update_state(self):
        match self.state:
            case ShuttleState.WAITING:
                if self.next_state is None:
                    return
                self.go_to_next_state()
            case _:
                if not self.follow_path():
                    if self.state == ShuttleState.LOOPING:
                        self.set_route(None)
                    self.go_to_next_state()

        self.last_updated = time.time()

    def go_to_next_state(self):
        match self.state:
            case ShuttleState.EXITING:
                self.send_webhook(entry=False)
        match self.next_state:
            case ShuttleState.ENTERING:
                self.send_webhook(entry=True)
                self.path = self.get_entering_path()
            case ShuttleState.LOOPING:
                self.path = self.get_looping_path()
            case ShuttleState.ON_BREAK:
                self.path = self.get_break_path()
            case ShuttleState.EXITING:
                self.path = self.get_exiting_path()
        self.state = self.next_state
        self.next_state = ShuttleState.WAITING
        self.path_index = 0
        self.subpath_index = 0
        self.distance_into_segment = 0

    def follow_path(self):
        """
        Move along the current path based on elapsed time and speed.
        Returns:
            True  - if still moving along the path
            False - if reached the end of the path
        """

        # If we've completed all path segments, stop
        if self.path_index >= len(self.path):
            return False

        # Distance to travel this update
        travel_distance = self.speed * (time.time() - self.last_updated)

        # Get current segment start/end
        start = self.path[self.path_index][self.subpath_index]
        end = self.path[self.path_index][self.subpath_index + 1]

        # Full length of current segment
        segment_vector = (end[0] - start[0], end[1] - start[1])
        segment_length = (segment_vector[0]**2 + segment_vector[1]**2) ** 0.5

        # Remaining distance in current segment
        remaining_distance = segment_length - self.distance_into_segment

        # Consume travel distance, possibly crossing segments
        while travel_distance > remaining_distance:
            # Move to the next segment
            travel_distance -= remaining_distance
            self.subpath_index += 1

            # If we finished a subpath, move to next path in self.path
            if self.subpath_index >= len(self.path[self.path_index]) - 1:
                self.path_index += 1
                self.subpath_index = 0

            self.distance_into_segment = 0

            # If we've reached the end of the path entirely
            if self.path_index >= len(self.path):
                self.location = tuple(self.path[-1][-1])
                return False

            # Update to the new segment
            start = self.path[self.path_index][self.subpath_index]
            end = self.path[self.path_index][self.subpath_index + 1]
            segment_vector = (end[0] - start[0], end[1] - start[1])
            segment_length = (segment_vector[0]**2 + segment_vector[1]**2) ** 0.5
            remaining_distance = segment_length  # new segment, starting fresh

        # Advance into the current segment
        self.distance_into_segment += travel_distance

        # Safety check: shouldn't exceed full segment length
        if self.distance_into_segment > segment_length:
            logger.error("Distance into segment exceeded segment length, logic error in path following")
            logger.error(f"Distance: {self.distance_into_segment}, Segment Length: {segment_length}")

        # Compute current location
        segment_unit_vector = (segment_vector[0] / segment_length, segment_vector[1] / segment_length)
        self.location = (
            start[0] + segment_unit_vector[0] * self.distance_into_segment,
            start[1] + segment_unit_vector[1] * self.distance_into_segment
        )

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
        return random.choice(
            [
                Stops.routes_data['ENTRY1'],
                # Stops.routes_data['ENTRY2'], entry 2 has a bug
            ]
        )['ROUTES']

    def get_looping_path(self):
        return Stops.routes_data[
            self.current_route
        ]['ROUTES']

    def get_break_path(self):
        return []

    def get_exiting_path(self):
        return random.choice(
            [
                Stops.routes_data['EXIT1'],
                Stops.routes_data['EXIT2'],
            ]
        )['ROUTES']

    def set_next_state(self, next_state: ShuttleState):
        if next_state not in ShuttleState:
            raise ValueError(f"Invalid shuttle state: {next_state}")
        self.next_state = next_state

    def set_route(self, route: str):
        if route is not None:
            if route not in Stops.routes_data.keys():
                raise ValueError(f"Invalid shuttle route: {route}")
            elif route not in Stops.active_routes:
                raise ValueError(f"Inactive shuttle route: {route}")
        self.current_route = route

    def to_dict(self):
        return {
            "id": self.id,
            "state": self.state.value,
            "next_state": self.next_state.value,
            "location": self.location,
            "last_updated": self.last_updated,
            "speed": self.speed,
            "path_index": self.path_index,
            "subpath_index": self.subpath_index,
            "distance_into_segment": self.distance_into_segment
        }
