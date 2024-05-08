from enum import IntEnum
from typing import Final


class CalibrationImageState(IntEnum):
    IGNORE: Final[int] = 0
    SELECT: Final[int] = 1
    DELETE: Final[int] = -1  # stage for deletion
