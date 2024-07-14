from enum import IntEnum
from typing import Final


class CalibrationResultState(IntEnum):
    RETAIN: Final[int] = 0
    DELETE: Final[int] = -1  # stage for deletion
