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
from typing import Final


class AnnotatorParametersGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_annotator_parameters_get"

    @staticmethod
    def type_identifier() -> str:
        return AnnotatorParametersGetRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class AnnotatorParametersGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_annotator_parameters_get"

    @staticmethod
    def type_identifier() -> str:
        return AnnotatorParametersGetResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    parameters: list[SerializeAsAny[KeyValueMetaAny]] = Field()


class AnnotatorParametersSetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_annotator_parameters_set"

    @staticmethod
    def type_identifier() -> str:
        return AnnotatorParametersSetRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    parameters: list[SerializeAsAny[KeyValueSimpleAny]] = Field()


class CameraImageGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_image_get"

    @staticmethod
    def type_identifier() -> str:
        return CameraImageGetRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    format: ImageFormat = Field()
    requested_resolution: ImageResolution | None = Field(default=None)


class CameraImageGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_image_get"

    @staticmethod
    def type_identifier() -> str:
        return CameraImageGetResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    format: ImageFormat = Field()
    image_base64: str = Field()
    original_resolution: ImageResolution = Field()


class CameraParametersGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_parameters_get"

    @staticmethod
    def type_identifier() -> str:
        return CameraParametersGetRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class CameraParametersGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_parameters_get"

    @staticmethod
    def type_identifier() -> str:
        return CameraParametersGetResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    parameters: list[SerializeAsAny[KeyValueMetaAny]] = Field()


class CameraParametersSetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_parameters_set"

    @staticmethod
    def type_identifier() -> str:
        return CameraParametersSetRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    parameters: list[SerializeAsAny[KeyValueSimpleAny]] = Field()


class CameraParametersSetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_parameters_set"

    @staticmethod
    def type_identifier() -> str:
        return CameraParametersSetResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    resolution: ImageResolution = Field()  # Sometimes parameter changes may result in changes of resolution


class CameraResolutionGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_resolution_get"

    @staticmethod
    def type_identifier() -> str:
        return CameraResolutionGetRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class CameraResolutionGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_resolution_get"

    @staticmethod
    def type_identifier() -> str:
        return CameraResolutionGetResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    resolution: ImageResolution = Field()


class DetectorFrameGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_frame_get"

    @staticmethod
    def type_identifier() -> str:
        return DetectorFrameGetRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    include_detected: bool = Field(default=True)
    include_rejected: bool = Field(default=True)


class DetectorFrameGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_frame_get"

    @staticmethod
    def type_identifier() -> str:
        return DetectorFrameGetResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    frame: DetectorFrame = Field()


class DetectorStartRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_start"

    @staticmethod
    def type_identifier() -> str:
        return DetectorStartRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class DetectorStopRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_stop"

    @staticmethod
    def type_identifier() -> str:
        return DetectorStopRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class IntrinsicCalibrationCalculateRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_calculate"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationCalculateRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    image_resolution: ImageResolution = Field()


class IntrinsicCalibrationCalculateResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_calculate"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationCalculateResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    result_identifier: str = Field()
    intrinsic_calibration: IntrinsicCalibration = Field()


class IntrinsicCalibrationDeleteStagedRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_delete_staged"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationDeleteStagedRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class IntrinsicCalibrationImageAddRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_image_add"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationImageAddRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class IntrinsicCalibrationImageAddResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_image_add"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationImageAddResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    image_identifier: str = Field()


class IntrinsicCalibrationImageGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_image_get"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationImageGetRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    image_identifier: str = Field()


class IntrinsicCalibrationImageGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_image_get"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationImageGetResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    image_base64: str = Field()


class IntrinsicCalibrationImageMetadataListRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_image_metadata_list"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationImageMetadataListRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    image_resolution: ImageResolution = Field()


class IntrinsicCalibrationImageMetadataListResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_image_metadata_list"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationImageMetadataListResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    metadata_list: list[IntrinsicCalibrator.ImageMetadata] = Field(default_factory=list)


class IntrinsicCalibrationImageMetadataUpdateRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_image_metadata_update"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationImageMetadataUpdateRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    image_identifier: str = Field()
    image_state: IntrinsicCalibrator.ImageState = Field()
    image_label: str | None = Field(default=None)


class IntrinsicCalibrationResolutionListRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_detector_resolutions_list"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationResolutionListRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class IntrinsicCalibrationResolutionListResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_detector_resolutions_list"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationResolutionListResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    resolutions: list[ImageResolution] = Field()


class IntrinsicCalibrationResultGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_result_get"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationResultGetRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    result_identifier: str = Field()


class IntrinsicCalibrationResultGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_result_get"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationResultGetResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    intrinsic_calibration: IntrinsicCalibration = Field()


class IntrinsicCalibrationResultGetActiveRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_result_active_get"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationResultGetActiveRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)


class IntrinsicCalibrationResultGetActiveResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_result_active_get"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationResultGetActiveResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    intrinsic_calibration: IntrinsicCalibration = Field()


class IntrinsicCalibrationResultMetadataListRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_result_metadata_list"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationResultMetadataListRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    image_resolution: ImageResolution = Field()


class IntrinsicCalibrationResultMetadataListResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_result_metadata_list"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationResultMetadataListResponse._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    metadata_list: list[IntrinsicCalibrator.ResultMetadata] = Field(default_factory=list)


class IntrinsicCalibrationResultMetadataUpdateRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_intrinsic_calibration_result_metadata_update"

    @staticmethod
    def type_identifier() -> str:
        return IntrinsicCalibrationResultMetadataUpdateRequest._TYPE_IDENTIFIER

    parsable_type: str = Field(default=_TYPE_IDENTIFIER)

    result_identifier: str = Field()
    result_state: IntrinsicCalibrator.ResultState = Field()
    result_label: str | None = Field(default=None)
