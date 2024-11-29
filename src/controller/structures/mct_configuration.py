from .startup_mode import StartupMode
from src.common.structures import \
    KeyValueSimpleAny, \
    Matrix4x4, \
    TargetBoard, \
    TargetMarker
from pydantic import BaseModel, Field, SerializeAsAny
from typing import Union


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
    targets: list[Union[TargetBoard, TargetMarker]] | None = Field(default=None)


class MCTConfiguration(BaseModel):
    startup_mode: StartupMode = Field()
    detectors: list[DetectorComponentConfig] = Field(default_factory=list)
    pose_solvers: list[PoseSolverConfig] = Field(default_factory=list)
