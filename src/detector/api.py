from src.common import \
    DetectorFrame, \
    ImageFormat, \
    ImageResolution, \
    IntrinsicCalibration, \
    IntrinsicCalibrator, \
    KeyValueMetaAny, \
    KeyValueSimpleAny, \
    MCTRequest, \
    MCTResponse
from pydantic import Field, SerializeAsAny


class AnnotatorParametersGetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_annotator_parameters_get"

    parsable_type: str = Field(default=type_identifier())


class AnnotatorParametersGetResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_annotator_parameters_get"

    parsable_type: str = Field(default=type_identifier())

    parameters: list[SerializeAsAny[KeyValueMetaAny]] = Field()


class AnnotatorParametersSetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_annotator_parameters_set"

    parsable_type: str = Field(default=type_identifier())

    parameters: list[SerializeAsAny[KeyValueSimpleAny]] = Field()


class CameraImageGetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_camera_image_get"

    parsable_type: str = Field(default=type_identifier())

    format: ImageFormat = Field()
    requested_resolution: ImageResolution | None = Field(default=None)


class CameraImageGetResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_camera_image_get"

    parsable_type: str = Field(default=type_identifier())

    format: ImageFormat = Field()
    image_base64: str = Field()
    original_resolution: ImageResolution = Field()


class CameraParametersGetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_camera_parameters_get"

    parsable_type: str = Field(default=type_identifier())


class CameraParametersGetResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_camera_parameters_get"

    parsable_type: str = Field(default=type_identifier())

    parameters: list[SerializeAsAny[KeyValueMetaAny]] = Field()


class CameraParametersSetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_camera_parameters_set"

    parsable_type: str = Field(default=type_identifier())

    parameters: list[SerializeAsAny[KeyValueSimpleAny]] = Field()


class CameraParametersSetResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_camera_parameters_set"

    parsable_type: str = Field(default=type_identifier())

    resolution: ImageResolution = Field()  # Sometimes parameter changes may result in changes of resolution


class CameraResolutionGetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_camera_resolution_get"

    parsable_type: str = Field(default=type_identifier())


class CameraResolutionGetResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_camera_resolution_get"

    parsable_type: str = Field(default=type_identifier())

    resolution: ImageResolution = Field()


class DetectorFrameGetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_frame_get"

    parsable_type: str = Field(default=type_identifier())

    include_detected: bool = Field(default=True)
    include_rejected: bool = Field(default=True)


class DetectorFrameGetResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_frame_get"

    parsable_type: str = Field(default=type_identifier())

    frame: DetectorFrame = Field()


class DetectorStartRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_start"

    parsable_type: str = Field(default=type_identifier())


class DetectorStopRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_stop"

    parsable_type: str = Field(default=type_identifier())


class IntrinsicCalibrationCalculateRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_calculate"

    parsable_type: str = Field(default=type_identifier())

    image_resolution: ImageResolution = Field()


class IntrinsicCalibrationCalculateResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_calculate"

    parsable_type: str = Field(default=type_identifier())

    result_identifier: str = Field()
    intrinsic_calibration: IntrinsicCalibration = Field()


class IntrinsicCalibrationDeleteStagedRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_delete_staged"

    parsable_type: str = Field(default=type_identifier())


class IntrinsicCalibrationImageAddRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_image_add"

    parsable_type: str = Field(default=type_identifier())


class IntrinsicCalibrationImageAddResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_image_add"

    parsable_type: str = Field(default=type_identifier())

    image_identifier: str = Field()


class IntrinsicCalibrationImageGetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_image_get"

    parsable_type: str = Field(default=type_identifier())

    image_identifier: str = Field()


class IntrinsicCalibrationImageGetResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_image_get"

    parsable_type: str = Field(default=type_identifier())

    image_base64: str = Field()


class IntrinsicCalibrationImageMetadataListRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_image_metadata_list"

    parsable_type: str = Field(default=type_identifier())

    image_resolution: ImageResolution = Field()


class IntrinsicCalibrationImageMetadataListResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_image_metadata_list"

    parsable_type: str = Field(default=type_identifier())

    metadata_list: list[IntrinsicCalibrator.ImageMetadata] = Field(default_factory=list)


class IntrinsicCalibrationImageMetadataUpdateRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_image_metadata_update"

    parsable_type: str = Field(default=type_identifier())

    image_identifier: str = Field()
    image_state: IntrinsicCalibrator.ImageState = Field()
    image_label: str | None = Field(default=None)


class IntrinsicCalibrationResolutionListRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_detector_resolutions_list"

    parsable_type: str = Field(default=type_identifier())


class IntrinsicCalibrationResolutionListResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_detector_resolutions_list"

    parsable_type: str = Field(default=type_identifier())

    resolutions: list[ImageResolution] = Field()


class IntrinsicCalibrationResultGetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_result_get"

    parsable_type: str = Field(default=type_identifier())

    result_identifier: str = Field()


class IntrinsicCalibrationResultGetResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_result_get"

    parsable_type: str = Field(default=type_identifier())

    intrinsic_calibration: IntrinsicCalibration = Field()


class IntrinsicCalibrationResultGetActiveRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_result_active_get"

    parsable_type: str = Field(default=type_identifier())


class IntrinsicCalibrationResultGetActiveResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_result_active_get"

    parsable_type: str = Field(default=type_identifier())

    intrinsic_calibration: IntrinsicCalibration = Field()


class IntrinsicCalibrationResultMetadataListRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_result_metadata_list"

    parsable_type: str = Field(default=type_identifier())

    image_resolution: ImageResolution = Field()


class IntrinsicCalibrationResultMetadataListResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_result_metadata_list"

    parsable_type: str = Field(default=type_identifier())

    metadata_list: list[IntrinsicCalibrator.ResultMetadata] = Field(default_factory=list)


class IntrinsicCalibrationResultMetadataUpdateRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "detector_intrinsic_calibration_result_metadata_update"

    parsable_type: str = Field(default=type_identifier())

    result_identifier: str = Field()
    result_state: IntrinsicCalibrator.ResultState = Field()
    result_label: str | None = Field(default=None)
