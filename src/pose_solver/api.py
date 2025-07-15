from src.common import \
    MCTRequest, \
    MCTResponse
from src.common.structures import \
    DetectorFrame, \
    IntrinsicParameters, \
    Matrix4x4, \
    Pose, \
    Target
from pydantic import Field
from typing import Final, Literal


class PoseSolverAddDetectorFrameRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "add_marker_corners"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverAddDetectorFrameRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    detector_label: str = Field()
    detector_frame: DetectorFrame = Field()


class PoseSolverAddTargetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "add_target"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverAddTargetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    target: Target = Field()


class PoseSolverAddTargetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "add_target"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverAddTargetResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    target_id: str = Field()


class PoseSolverGetPosesRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "get_poses"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverGetPosesRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class PoseSolverGetPosesResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "get_poses"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverGetPosesResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    detector_poses: list[Pose]
    target_poses: list[Pose]


class PoseSolverSetExtrinsicRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "set_extrinsic_parameters"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverSetExtrinsicRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    detector_label: str = Field()
    transform_to_reference: Matrix4x4 = Field()


class PoseSolverSetIntrinsicRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "set_intrinsic_parameters"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverSetIntrinsicRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    detector_label: str = Field()
    intrinsic_parameters: IntrinsicParameters = Field()


class PoseSolverSetReferenceRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "set_reference_marker"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverSetReferenceRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    marker_id: int = Field()
    marker_diameter: float = Field()


class PoseSolverSetTargetsRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "set_targets"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverSetTargetsRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    targets: list[Target] = Field()


class PoseSolverStartRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "start_pose_solver"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverStartRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class PoseSolverStopRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "stop_pose_solver"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverStopRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)
