from enum import StrEnum
from typing import Final


COMPONENT_ROLE_LABEL_DETECTOR: Final[str] = "detector"
COMPONENT_ROLE_LABEL_POSE_SOLVER: Final[str] = "pose_solver"
class ComponentRoleLabel(StrEnum):
    DETECTOR: Final[str] = COMPONENT_ROLE_LABEL_DETECTOR
    POSE_SOLVER: Final[str] = COMPONENT_ROLE_LABEL_POSE_SOLVER


