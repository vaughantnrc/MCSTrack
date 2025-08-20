from src.common import \
    DetectorFrame, \
    ExtrinsicCalibration, \
    ExtrinsicCalibrator, \
    IntrinsicParameters, \
    Matrix4x4, \
    MCTRequest, \
    MCTResponse, \
    Pose, \
    Target
from pydantic import Field
from typing import Final, Literal


class ExtrinsicCalibrationCalculateRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_calculate"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationCalculateRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class ExtrinsicCalibrationCalculateResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_calculate"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationCalculateResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    result_identifier: str = Field()
    extrinsic_calibration: ExtrinsicCalibration = Field()


class ExtrinsicCalibrationDeleteStagedRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_delete_staged"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationDeleteStagedRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class ExtrinsicCalibrationImageAddRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_image_add"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationImageAddRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    image_base64: str = Field()
    detector_label: str = Field()
    timestamp_utc_iso8601: str = Field()


class ExtrinsicCalibrationImageAddResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_image_add"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationImageAddResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    image_identifier: str = Field()


class ExtrinsicCalibrationImageGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_image_get"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationImageGetRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    image_identifier: str = Field()


class ExtrinsicCalibrationImageGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_image_get"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationImageGetResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    image_base64: str = Field()


class ExtrinsicCalibrationImageMetadataListRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_image_metadata_list"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationImageMetadataListRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class ExtrinsicCalibrationImageMetadataListResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_image_metadata_list"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationImageMetadataListResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    metadata_list: list[ExtrinsicCalibrator.ImageMetadata] = Field(default_factory=list)


class ExtrinsicCalibrationImageMetadataUpdateRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_image_metadata_update"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationImageMetadataUpdateRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    image_identifier: str = Field()
    image_state: ExtrinsicCalibrator.ImageState = Field()
    image_label: str | None = Field(default=None)


class ExtrinsicCalibrationResultGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_result_get"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationResultGetRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    result_identifier: str = Field()


class ExtrinsicCalibrationResultGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_result_get"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationResultGetResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    extrinsic_calibration: ExtrinsicCalibration = Field()


class ExtrinsicCalibrationResultGetActiveRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_result_active_get"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationResultGetActiveRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class ExtrinsicCalibrationResultGetActiveResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_result_active_get"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationResultGetActiveResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    extrinsic_calibration: ExtrinsicCalibration = Field()


class ExtrinsicCalibrationResultMetadataListRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_result_metadata_list"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationResultMetadataListRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class ExtrinsicCalibrationResultMetadataListResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_result_metadata_list"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationResultMetadataListResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    metadata_list: list[ExtrinsicCalibrator.ResultMetadata] = Field(default_factory=list)


class ExtrinsicCalibrationResultMetadataUpdateRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_extrinsic_calibration_result_metadata_update"

    @staticmethod
    def type_identifier() -> str:
        return ExtrinsicCalibrationResultMetadataUpdateRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    result_identifier: str = Field()
    result_state: ExtrinsicCalibrator.ResultState = Field()
    result_label: str | None = Field(default=None)


class PoseSolverAddDetectorFrameRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_pose_solver_add_marker_corners"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverAddDetectorFrameRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    detector_label: str = Field()
    detector_frame: DetectorFrame = Field()


class PoseSolverAddTargetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_pose_solver_add_target"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverAddTargetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    target: Target = Field()


class PoseSolverAddTargetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "mixer_pose_solver_add_target"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverAddTargetResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    target_id: str = Field()


class PoseSolverGetPosesRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_pose_solver_get_poses"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverGetPosesRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class PoseSolverGetPosesResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "mixer_pose_solver_get_poses"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverGetPosesResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    detector_poses: list[Pose]
    target_poses: list[Pose]


class PoseSolverSetExtrinsicRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_pose_solver_set_extrinsic_parameters"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverSetExtrinsicRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    detector_label: str = Field()
    transform_to_reference: Matrix4x4 = Field()


class PoseSolverSetReferenceRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_pose_solver_set_reference_marker"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverSetReferenceRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    marker_id: int = Field()
    marker_diameter: float = Field()


class PoseSolverSetTargetsRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_pose_solver_set_targets"

    @staticmethod
    def type_identifier() -> str:
        return PoseSolverSetTargetsRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    targets: list[Target] = Field()


class MixerStartRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_start"

    @staticmethod
    def type_identifier() -> str:
        return MixerStartRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class MixerStopRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_stop"

    @staticmethod
    def type_identifier() -> str:
        return MixerStopRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class MixerUpdateIntrinsicParametersRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "mixer_update_intrinsic_parameters"

    @staticmethod
    def type_identifier() -> str:
        return MixerUpdateIntrinsicParametersRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    detector_label: str = Field()
    intrinsic_parameters: IntrinsicParameters = Field()
