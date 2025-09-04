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


class ExtrinsicCalibrationCalculateRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_calculate"

    parsable_type: str = Field(default=type_identifier())


class ExtrinsicCalibrationCalculateResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_calculate"

    parsable_type: str = Field(default=type_identifier())

    result_identifier: str = Field()
    extrinsic_calibration: ExtrinsicCalibration = Field()


class ExtrinsicCalibrationDeleteStagedRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_delete_staged"

    parsable_type: str = Field(default=type_identifier())


class ExtrinsicCalibrationImageAddRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_image_add"

    parsable_type: str = Field(default=type_identifier())

    image_base64: str = Field()
    detector_label: str = Field()
    timestamp_utc_iso8601: str = Field()


class ExtrinsicCalibrationImageAddResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_image_add"

    parsable_type: str = Field(default=type_identifier())

    image_identifier: str = Field()


class ExtrinsicCalibrationImageGetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_image_get"

    parsable_type: str = Field(default=type_identifier())

    image_identifier: str = Field()


class ExtrinsicCalibrationImageGetResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_image_get"

    parsable_type: str = Field(default=type_identifier())

    image_base64: str = Field()


class ExtrinsicCalibrationImageMetadataListRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_image_metadata_list"

    parsable_type: str = Field(default=type_identifier())


class ExtrinsicCalibrationImageMetadataListResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_image_metadata_list"

    parsable_type: str = Field(default=type_identifier())

    metadata_list: list[ExtrinsicCalibrator.ImageMetadata] = Field(default_factory=list)


class ExtrinsicCalibrationImageMetadataUpdateRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_image_metadata_update"

    parsable_type: str = Field(default=type_identifier())

    image_identifier: str = Field()
    image_state: ExtrinsicCalibrator.ImageState = Field()
    image_label: str | None = Field(default=None)


class ExtrinsicCalibrationResultGetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_result_get"

    parsable_type: str = Field(default=type_identifier())

    result_identifier: str = Field()


class ExtrinsicCalibrationResultGetResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_result_get"

    parsable_type: str = Field(default=type_identifier())

    extrinsic_calibration: ExtrinsicCalibration = Field()


class ExtrinsicCalibrationResultGetActiveRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_result_active_get"

    parsable_type: str = Field(default=type_identifier())


class ExtrinsicCalibrationResultGetActiveResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_result_active_get"

    parsable_type: str = Field(default=type_identifier())

    extrinsic_calibration: ExtrinsicCalibration = Field()


class ExtrinsicCalibrationResultMetadataListRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_result_metadata_list"

    parsable_type: str = Field(default=type_identifier())


class ExtrinsicCalibrationResultMetadataListResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_result_metadata_list"

    parsable_type: str = Field(default=type_identifier())

    metadata_list: list[ExtrinsicCalibrator.ResultMetadata] = Field(default_factory=list)


class ExtrinsicCalibrationResultMetadataUpdateRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_extrinsic_calibration_result_metadata_update"

    parsable_type: str = Field(default=type_identifier())

    result_identifier: str = Field()
    result_state: ExtrinsicCalibrator.ResultState = Field()
    result_label: str | None = Field(default=None)


class PoseSolverAddDetectorFrameRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_pose_solver_add_marker_corners"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())

    detector_label: str = Field()
    detector_frame: DetectorFrame = Field()


class PoseSolverAddTargetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_pose_solver_add_target"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())

    target: Target = Field()


class PoseSolverAddTargetResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_pose_solver_add_target"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())

    target_id: str = Field()


class PoseSolverGetPosesRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_pose_solver_get_poses"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())


class PoseSolverGetPosesResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_pose_solver_get_poses"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())

    detector_poses: list[Pose]
    target_poses: list[Pose]


class PoseSolverSetExtrinsicRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_pose_solver_set_extrinsic_parameters"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())

    detector_label: str = Field()
    transform_to_reference: Matrix4x4 = Field()


class PoseSolverSetReferenceRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_pose_solver_set_reference_marker"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())

    marker_id: int = Field()
    marker_diameter: float = Field()


class PoseSolverSetTargetsRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_pose_solver_set_targets"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())

    targets: list[Target] = Field()


class MixerStartRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_start"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())


class MixerStopRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_stop"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())


class MixerUpdateIntrinsicParametersRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "mixer_update_intrinsic_parameters"

    parsable_type: str = Field(default=type_identifier())

    detector_label: str = Field()
    intrinsic_parameters: IntrinsicParameters = Field()
