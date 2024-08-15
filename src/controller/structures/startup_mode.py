from enum import StrEnum
from typing import Final


class StartupMode(StrEnum):
    DETECTING_ONLY: Final[str] = "detecting_only"
    DETECTING_AND_SOLVING: Final[str] = "detecting_and_solving"
