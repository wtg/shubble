import time
import threading
import random
import uuid
import logging
import requests
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone

from .dev_time import dev_now
from typing import Optional

from shared.stops import Stops

logger = logging.getLogger(__name__)

# How long a shuttle dwells at Union between loops in the dev sim.
# Real shuttles pause ~5 min between loops; we use a shorter value here
# so iterative testing doesn't take forever. Must be well below the
# backend's IDLE_THRESHOLD_SEC (1200s) so a normal dwell isn't filtered.
INTER_LOOP_PAUSE_SEC = 60  # 1 minute (real life: ~5 min)

# Fixed off-route parking coordinate all shuttles drive to when taking a break.
# Verified during planning to be >400m from both NORTH and WEST polylines so
# the backend's Filter 2 (off-route) triggers cleanly. Shared across routes
# for simulation simplicity.
BREAK_SPOT: tuple[float, float] = (42.7265, -73.672)

# Small straight-line segment count used when generating the drive-to / drive-back
# paths. Keeping it at 1 is fine because _follow_path interpolates smoothly within
# a single segment and the break "drive" isn't meant to follow a real road.
_BREAK_PATH_SEGMENTS = 1


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
    # When True on a LOOPING action, the shuttle runs exactly ONE loop
    # and then terminates the action (returning to idle) instead of
    # entering the inter-loop dwell and repeating forever. Used by the
    # schedule-strict mode so each scheduled departure produces exactly
    # one run of the route.
    single_loop: bool = False


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
        self._last_updated: datetime = dev_now(timezone.utc)
        self._speed: float = 20.0  # mph

        # Path following state
        self._path: list = []
        self._path_index: int = 0
        self._subpath_index: int = 0
        self._distance_into_segment: float = 0.0
        self._current_route: Optional[str] = None
        # When non-None, the shuttle is dwelling at Union between loops
        # until time.time() reaches this value.
        self._loop_dwell_until: Optional[float] = None

        # Break phase: None | "drive_out" | "waiting" | "drive_back"
        # "drive_out": following straight-line path from current location to BREAK_SPOT
        # "waiting":   stationary at BREAK_SPOT, counting down duration
        # "drive_back": following straight-line path from BREAK_SPOT to first on-route stop
        self._break_phase: Optional[str] = None
        # Wall-clock time.time() stamp for when the "waiting" phase began. Duration
        # is measured from this, NOT from _action_start_time, so drive-out travel
        # time doesn't eat into the stationary break window.
        self._break_wait_start: Optional[float] = None
        # Route used for the drive_back target. Snapshotted at break start so a
        # downstream action that mutates _current_route can't misdirect the return.
        self._break_return_route: Optional[str] = None

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

    def push_action(
        self,
        action: ShuttleAction,
        route: Optional[str] = None,
        duration: Optional[float] = None,
        single_loop: bool = False,
    ) -> str:
        """Add an action to the queue. Returns the action ID."""
        if action == ShuttleAction.LOOPING and route is None:
            raise ValueError("LOOPING action requires a route")
        if action == ShuttleAction.LOOPING and route not in Stops.active_routes:
            raise ValueError(f"Invalid or inactive route: {route}")
        # ON_BREAK may optionally carry a route; if provided it must be active
        # because _handle_on_break uses it to pick the first-stop return target.
        if action == ShuttleAction.ON_BREAK and route is not None and route not in Stops.active_routes:
            raise ValueError(f"Invalid or inactive route for break: {route}")

        action_id = str(uuid.uuid4())
        queued = QueuedAction(
            id=action_id,
            action=action,
            route=route,
            duration=duration,
            single_loop=single_loop,
        )

        with self.lock:
            self._action_queue.append(queued)
            logger.info(
                f"Shuttle {self.id}: queued {action.value}"
                + (f" on {route}" if route else "")
                + (" (single_loop)" if single_loop else "")
            )

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

            self._last_updated = dev_now(timezone.utc)

    # --- Action Handlers ---

    def _handle_idle(self):
        """Handle idle state - check for next action in queue."""
        if self._action_index < len(self._action_queue):
            next_queued = self._action_queue[self._action_index]
            next_queued.status = "in_progress"
            self._start_action(next_queued)
            self._action_index += 1

    def _handle_on_break(self):
        """ON_BREAK state: drive off-route, sit stationary, drive back on-route.

        Phase transitions:
          drive_out  -> waiting    (when drive-out path completes)
          waiting    -> drive_back (when duration elapses)
          drive_back -> idle       (when drive-back path completes; action marked completed)
        """
        if self._action_index <= 0:
            # No queued break on record; can't happen in practice — bail to idle.
            self._current_action = None
            return
        current = self._action_queue[self._action_index - 1]
        duration = current.duration or 0

        if self._break_phase == "drive_out":
            # Follow the straight-line path to BREAK_SPOT.
            if not self._follow_path():
                # Arrived. Switch to waiting; snap location to BREAK_SPOT.
                self._location = BREAK_SPOT
                self._break_phase = "waiting"
                self._break_wait_start = time.time()
                logger.info(
                    f"Shuttle {self.id}: arrived at break spot, "
                    f"waiting {duration:.0f}s"
                )
            return

        if self._break_phase == "waiting":
            # Stationary. GPS pings continue via the outer _last_updated tick.
            wait_started = self._break_wait_start or time.time()
            if time.time() - wait_started >= duration:
                # Duration elapsed. Build drive-back path to first stop of the
                # return route (falls back to Union if route is unknown).
                target = (42.730711, -73.676737)  # Union / su_coords fallback
                if self._break_return_route:
                    try:
                        route_data = Stops.routes_data[self._break_return_route]
                        first_stop_key = route_data['STOPS'][0]
                        coords = route_data[first_stop_key]['COORDINATES']
                        target = (coords[0], coords[1])
                    except (KeyError, IndexError, TypeError):
                        logger.warning(
                            f"Shuttle {self.id}: could not resolve first-stop "
                            f"for route {self._break_return_route}, "
                            f"falling back to Union"
                        )
                self._path = [[[BREAK_SPOT[0], BREAK_SPOT[1]], [target[0], target[1]]]]
                self._reset_path_state()
                self._break_phase = "drive_back"
                logger.info(
                    f"Shuttle {self.id}: break finished, driving back to "
                    f"{self._break_return_route or 'Union'}"
                )
            return

        if self._break_phase == "drive_back":
            if not self._follow_path():
                # Arrived at first stop. Complete the action and return to idle.
                current.status = "completed"
                self._break_phase = None
                self._break_wait_start = None
                self._break_return_route = None
                logger.info(f"Shuttle {self.id}: returned from break, idle")
                self._current_action = None
            return

        # Defensive: if _break_phase got into an unexpected state, finish cleanly.
        logger.warning(
            f"Shuttle {self.id}: unexpected break phase "
            f"{self._break_phase!r}; marking break complete"
        )
        current.status = "completed"
        self._break_phase = None
        self._current_action = None

    def _handle_movement(self):
        """Handle movement actions (ENTERING, LOOPING, EXITING)."""
        # While paused between loops at Union, just hold position
        # until the dwell timer expires, then start the next loop.
        if self._current_action == ShuttleAction.LOOPING and self._loop_dwell_until:
            if time.time() >= self._loop_dwell_until:
                self._loop_dwell_until = None
                self._path = self._get_looping_path(self._current_route)
                self._reset_path_state()
            return

        if not self._follow_path():
            # LOOPING repeats: real shuttles dwell briefly at Union between
            # loops (driver stretch, passenger boarding) then continue.
            # The dev-sim pauses for INTER_LOOP_PAUSE_SEC, much shorter
            # than real life so testing doesn't take 15 minutes per loop.
            # The idle-shuttle filter (>20 min at first stop) is well
            # above this dwell so a normal dwell won't accidentally
            # hide the trip.
            #
            # EXCEPT in schedule-strict mode: when the queued action's
            # single_loop flag is set, one pass around the route is
            # enough. Mark completed and return to idle so the shuttle
            # waits for its NEXT scheduled departure to fire another
            # single_loop action.
            if self._current_action == ShuttleAction.LOOPING and self._current_route:
                current_queued = (
                    self._action_queue[self._action_index - 1]
                    if self._action_index > 0 else None
                )
                if current_queued and current_queued.single_loop:
                    current_queued.status = "completed"
                    logger.info(
                        f"Shuttle {self.id}: single loop complete on "
                        f"{self._current_route}, returning to idle"
                    )
                    self._current_action = None
                    return
                self._loop_dwell_until = time.time() + INTER_LOOP_PAUSE_SEC
                logger.info(
                    f"Shuttle {self.id}: loop complete on {self._current_route}, "
                    f"dwelling at Union for {INTER_LOOP_PAUSE_SEC}s"
                )
                return

            # Path complete (non-LOOPING actions)
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
            # D-01: drive from current location OUT to BREAK_SPOT.
            # Use a single straight segment — _follow_path interpolates smoothly
            # and this matches the backend's expected off-route shape (shuttle
            # leaves polyline at a sharp angle, arrives at a non-stop coord).
            self._break_return_route = queued.route or self._current_route
            self._break_phase = "drive_out"
            self._break_wait_start = None
            start_pt = [self._location[0], self._location[1]]
            end_pt = [BREAK_SPOT[0], BREAK_SPOT[1]]
            self._path = [[start_pt, end_pt]]
            self._reset_path_state()
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
        elapsed_seconds = (dev_now(timezone.utc) - self._last_updated).total_seconds()
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
            'eventTime': dev_now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'),
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
