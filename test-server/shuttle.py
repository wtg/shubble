import time
from enum import Enum
import random

class ShuttleState(Enum):
    WAITING = "waiting"
    ENTERING = "entering"
    LOOPING = "looping"
    ON_BREAK = "on_break"
    EXITING = "exiting"
    EXITED = "exited"

class ShuttleLoop(Enum):
    NORTH = "north"
    WEST = "west"

class Shuttle:
    def __init__(self, shuttle_id: str):
        # larger state
        self.id = shuttle_id
        self.state = ShuttleState.WAITING
        self.next_state = None

        # shuttle properties
        self.last_updated = time.time()
        self.location = (0, 0)
        self.speed = self.speed_range[0] + (self.speed_range[1] - self.speed_range[0]) / 2  # Default speed
        self.loop = random.choice(list(ShuttleLoop))

        # shuttle configuration
        self.speed_range = (0, 10)
        self.break_probability = 0.1  # 10% chance of going on break

    def update_state(self):
        # update every 5 seconds
        pass

    def to_dict(self):
        return {
            "id": self.id,
            "loop": self.loop.value,
            "speed": self.speed,
            "location": self.location,
            "state": self.state.value,
            "last_updated": self.last_updated
        }
