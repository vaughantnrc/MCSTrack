from typing import Final
from enum import IntEnum


class DetectorStatus:

    class Capture(IntEnum):
        STOPPED: Final[int] = 0
        RUNNING: Final[int] = 1
        FAILURE: Final[int] = 2

    class Marker(IntEnum):
        STOPPED: Final[int] = 0
        RUNNING: Final[int] = 1
        FAILURE: Final[int] = 2

    capture_status: Capture
    capture_errors: list[str]

    marker_status: Marker
    marker_errors: list[str]

    def __init__(self):
        self.capture_status = DetectorStatus.Capture.STOPPED
        self.capture_errors = list()

    def in_runnable_state(self):
        return self.capture_status == DetectorStatus.Capture.RUNNING
