from src.common import \
    KeyValueSimpleAny, \
    Matrix4x4, \
    Target
from enum import StrEnum
from pydantic import BaseModel, Field, SerializeAsAny


class StartupMode(StrEnum):
    DETECTING_ONLY = "detecting_only"
    DETECTING_AND_SOLVING = "detecting_and_solving"


class MCTComponentConfig(BaseModel):
    label: str = Field()
    ip_address: str = Field()
    port: int = Field()


class DetectorComponentConfig(MCTComponentConfig):
    camera_parameters: list[SerializeAsAny[KeyValueSimpleAny]] | None = Field(default=None)
    marker_parameters: list[SerializeAsAny[KeyValueSimpleAny]] | None = Field(default=None)
    fixed_transform_to_reference: Matrix4x4 | None = Field(default=None)


class PoseSolverConfig(MCTComponentConfig):
    solver_parameters: list[SerializeAsAny[KeyValueSimpleAny]] | None = Field(default=None)
    targets: list[Target] | None = Field(default=None)


class MCTConfiguration(BaseModel):
    startup_mode: StartupMode = Field()
    detectors: list[DetectorComponentConfig] = Field(default_factory=list)
    mixers: list[PoseSolverConfig] = Field(default_factory=list)
