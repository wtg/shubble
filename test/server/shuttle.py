import time
import threading
import random
import uuid
import logging
import requests
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from shared.stops import Stops

logger = logging.getLogger(__name__)


class ShuttleAction(Enum):
    ENTERING = "entering"
    LOOPING = "looping"
    ON_BREAK = "on_break"
    EXITING = "exiting"


@dataclass
class QueuedAction:
    """An action in the shuttle's queue."""
    id: str
    action: ShuttleAction
    status: str = "pending"  # pending, in_progress, completed, failed
    route: Optional[str] = None
    duration: Optional[float] = None  # seconds, for ON_BREAK


class Shuttle:
    """
    Shuttle that processes a queue of ShuttleActions.
    Runs in its own thread, updating location every 5 seconds.
    """

    def __init__(self, shuttle_id: str):
        self.id = shuttle_id
        self.lock = threading.Lock()

        # Action queue
        self._action_queue: list[QueuedAction] = []
        self._action_index: int = 0
        self._current_action: Optional[ShuttleAction] = None

        # Location state
        self._location: tuple[float, float] = (0.0, 0.0)
        self._last_updated: datetime = datetime.now(timezone.utc)
        self._speed: float = 20.0  # mph

        # Path following state
        self._path: list = []
        self._path_index: int = 0
        self._subpath_index: int = 0
        self._distance_into_segment: float = 0.0
        self._current_route: Optional[str] = None

        # Thread control
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        # Action start time (for timed actions)
        self._action_start_time: float = 0.0

    # --- Public Interface ---

    @property
    def action_index(self) -> int:
        with self.lock:
            return self._action_index

    @property
    def location(self) -> tuple[float, float]:
        with self.lock:
            return self._location

    @property
    def state(self) -> Optional[ShuttleAction]:
        with self.lock:
            return self._current_action

    @property
    def speed(self) -> float:
        with self.lock:
            return self._speed

    @property
    def last_updated(self) -> datetime:
        with self.lock:
            return self._last_updated

    @property
    def queue_length(self) -> int:
        with self.lock:
            return len(self._action_queue)

    def get_queue(self) -> list[dict]:
        """Get a copy of the action queue."""
        with self.lock:
            return [
                {
                    "id": a.id,
                    "action": a.action.value,
                    "status": a.status,
                    "route": a.route,
                    "duration": a.duration,
                }
                for a in self._action_queue
            ]

    def push_action(self, action: ShuttleAction, route: Optional[str] = None, duration: Optional[float] = None) -> str:
        """Add an action to the queue. Returns the action ID."""
        if action == ShuttleAction.LOOPING and route is None:
            raise ValueError("LOOPING action requires a route")
        if action == ShuttleAction.LOOPING and route not in Stops.active_routes:
            raise ValueError(f"Invalid or inactive route: {route}")

        action_id = str(uuid.uuid4())
        queued = QueuedAction(id=action_id, action=action, route=route, duration=duration)

        with self.lock:
            self._action_queue.append(queued)
            logger.info(f"Shuttle {self.id}: queued {action.value}" + (f" on {route}" if route else ""))

        return action_id

    def clear_queue(self):
        """Clear all pending actions from the queue."""
        with self.lock:
            # Keep only actions up to current index
            self._action_queue = self._action_queue[:self._action_index + 1] if self._action_index < len(self._action_queue) else []

    def start(self):
        """Start the shuttle's run loop in a background thread."""
        with self.lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info(f"Shuttle {self.id}: started")

    def stop(self):
        """Stop the shuttle's run loop."""
        with self.lock:
            self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            logger.info(f"Shuttle {self.id}: stopped")

    def to_dict(self) -> dict:
        """Convert shuttle state to dictionary."""
        with self.lock:
            next_action = None
            if self._action_index < len(self._action_queue):
                next_action = self._action_queue[self._action_index].action.value

            queue = [
                {
                    "id": a.id,
                    "action": a.action.value,
                    "status": a.status,
                    "route": a.route,
                    "duration": a.duration,
                }
                for a in self._action_queue
            ]

            return {
                "id": self.id,
                "state": self._current_action.value if self._current_action else None,
                "next_state": next_action,
                "location": self._location,
                "last_updated": self._last_updated.isoformat(),
                "speed": self._speed,
                "action_index": self._action_index,
                "queue": queue,
                "current_route": self._current_route,
            }

    # --- Run Loop ---

    def _run_loop(self):
        """Main run loop - updates location every 5 seconds."""
        while True:
            with self.lock:
                if not self._running:
                    break

            self._update_state()

            time.sleep(0.1)  # Update frequently for smooth movement

    def _update_state(self):
        """Update shuttle state based on current action."""
        with self.lock:
            current_action = self._current_action

            if current_action is None:
                self._handle_idle()
            elif current_action == ShuttleAction.ON_BREAK:
                self._handle_on_break()
            elif current_action in (ShuttleAction.ENTERING, ShuttleAction.LOOPING, ShuttleAction.EXITING):
                self._handle_movement()

            self._last_updated = datetime.now(timezone.utc)

    # --- Action Handlers ---

    def _handle_idle(self):
        """Handle idle state - check for next action in queue."""
        if self._action_index < len(self._action_queue):
            next_queued = self._action_queue[self._action_index]
            next_queued.status = "in_progress"
            self._start_action(next_queued)
            self._action_index += 1

    def _handle_on_break(self):
        """Handle ON_BREAK state - wait for duration then return to idle."""
        # Find current action's duration
        if self._action_index > 0:
            current = self._action_queue[self._action_index - 1]
            duration = current.duration or 0

            if time.time() - self._action_start_time >= duration:
                current.status = "completed"
                logger.info(f"Shuttle {self.id}: break finished")
                self._current_action = None

    def _handle_movement(self):
        """Handle movement actions (ENTERING, LOOPING, EXITING)."""
        if not self._follow_path():
            # Path complete
            if self._current_action == ShuttleAction.EXITING:
                self._send_webhook(entry=False)

            # Mark action as completed
            if self._action_index > 0:
                self._action_queue[self._action_index - 1].status = "completed"

            logger.info(f"Shuttle {self.id}: {self._current_action.value} complete")
            self._current_action = None

    def _start_action(self, queued: QueuedAction):
        """Start executing a queued action."""
        action = queued.action
        logger.info(f"Shuttle {self.id}: starting {action.value}" + (f" on {queued.route}" if queued.route else ""))

        self._action_start_time = time.time()

        if action == ShuttleAction.ENTERING:
            self._send_webhook(entry=True)
            self._path = self._get_entering_path()
            self._reset_path_state()
            self._current_action = ShuttleAction.ENTERING

        elif action == ShuttleAction.LOOPING:
            self._current_route = queued.route
            self._path = self._get_looping_path(queued.route)
            self._reset_path_state()
            self._current_action = ShuttleAction.LOOPING

        elif action == ShuttleAction.ON_BREAK:
            self._current_action = ShuttleAction.ON_BREAK

        elif action == ShuttleAction.EXITING:
            self._path = self._get_exiting_path()
            self._reset_path_state()
            self._current_action = ShuttleAction.EXITING

    def _reset_path_state(self):
        """Reset path following state for a new path."""
        self._path_index = 0
        self._subpath_index = 0
        self._distance_into_segment = 0.0

    # --- Path Following ---

    def _follow_path(self) -> bool:
        """
        Move along the current path based on elapsed time and speed.
        Returns True if still moving, False if path complete.
        """
        if self._path_index >= len(self._path):
            return False

        # Distance to travel this update
        speed_mps = self._speed * 0.44704  # mph to m/s
        speed_pseudo = speed_mps / 1000 * 0.02  # Approx to degrees
        elapsed_seconds = (datetime.now(timezone.utc) - self._last_updated).total_seconds()
        travel_distance = speed_pseudo * elapsed_seconds

        # Current segment
        start = self._path[self._path_index][self._subpath_index]
        end = self._path[self._path_index][self._subpath_index + 1]

        segment_vector = (end[0] - start[0], end[1] - start[1])
        segment_length = (segment_vector[0]**2 + segment_vector[1]**2) ** 0.5

        remaining = segment_length - self._distance_into_segment

        # Move through segments
        while travel_distance > remaining:
            travel_distance -= remaining
            self._subpath_index += 1

            if self._subpath_index >= len(self._path[self._path_index]) - 1:
                self._path_index += 1
                self._subpath_index = 0

            self._distance_into_segment = 0.0

            if self._path_index >= len(self._path):
                self._location = tuple(self._path[-1][-1])
                return False

            start = self._path[self._path_index][self._subpath_index]
            end = self._path[self._path_index][self._subpath_index + 1]
            segment_vector = (end[0] - start[0], end[1] - start[1])
            segment_length = (segment_vector[0]**2 + segment_vector[1]**2) ** 0.5
            remaining = segment_length

        # Advance in segment
        self._distance_into_segment += travel_distance

        if segment_length > 0:
            unit = (segment_vector[0] / segment_length, segment_vector[1] / segment_length)
            self._location = (
                start[0] + unit[0] * self._distance_into_segment,
                start[1] + unit[1] * self._distance_into_segment,
            )

        return True

    # --- Path Generation ---

    def _get_entering_path(self) -> list:
        return random.choice([
            Stops.routes_data['ENTRY1'],
        ])['ROUTES']

    def _get_looping_path(self, route: str) -> list:
        return Stops.routes_data[route]['ROUTES']

    def _get_exiting_path(self) -> list:
        return random.choice([
            Stops.routes_data['EXIT1'],
            Stops.routes_data['EXIT2'],
        ])['ROUTES']

    # --- Webhooks ---

    def _send_webhook(self, entry: bool = True):
        """Send geofence entry/exit webhook to backend."""
        url = 'http://localhost:8000/api/webhook'
        headers = {'Content-Type': 'application/json'}

        vehicle = {
            'id': self.id,
            'name': self.id[-3:],
            'licensePlate': f'FAKE{self.id[-3:]}',
            'vin': self.id[-3:],
            'assetType': 'vehicle',
            'externalIds': {'maintenanceId': self.id[-3:]},
            'gateway': {'model': 'VG34', 'serial': self.id[-3:]},
        }

        address = {
            'id': '123456',
            'name': 'Test Location',
            'formattedAddress': 'Test Address',
            'externalIds': {'siteId': '54'},
            'geofence': {
                'id': 'geofence123',
                'name': 'Test Geofence',
                'polygon': {
                    'vertices': [{'latitude': self._location[0], 'longitude': self._location[1]}]
                },
            },
        }

        event_type = 'geofenceEntry' if entry else 'geofenceExit'
        geofence_key = 'geofenceEntry' if entry else 'geofenceExit'

        payload = {
            'eventId': str(uuid.uuid4()),
            'eventTime': datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
            'eventType': event_type,
            'orgId': 20936,
            'webhookId': '1411751028848270',
            'data': {
                'conditions': [{
                    'details': {
                        geofence_key: {
                            'vehicle': vehicle,
                            'address': address,
                        }
                    }
                }]
            },
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=5)
            if response.status_code == 200:
                logger.info(f'Shuttle {self.id}: sent {event_type}')
                return True
            else:
                logger.error(f'Shuttle {self.id}: webhook failed ({response.status_code})')
                return False
        except Exception as e:
            logger.error(f'Shuttle {self.id}: webhook error: {e}')
            return False
