from src.common import \
    MCTRequest, \
    MCTResponse
from src.common.structures import \
    CaptureFormat, \
    DetectorFrame, \
    IntrinsicCalibration, \
    ImageResolution, \
    KeyValueMetaAny, \
    KeyValueSimpleAny
from .structures import \
    CalibrationImageMetadata, \
    CalibrationImageState, \
    CalibrationResultMetadata, \
    CalibrationResultState
from pydantic import Field, SerializeAsAny
from typing import Final, Literal


class CalibrationCalculateRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_calculate"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationCalculateRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    image_resolution: ImageResolution = Field()


class CalibrationCalculateResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_calculate"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationCalculateResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    result_identifier: str = Field()
    intrinsic_calibration: IntrinsicCalibration = Field()


class CalibrationDeleteStagedRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_delete_staged"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationDeleteStagedRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class CalibrationImageAddRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_image_add"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationImageAddRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class CalibrationImageAddResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_image_add"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationImageAddResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    image_identifier: str = Field()


class CalibrationImageGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_image_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationImageGetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    image_identifier: str = Field()


class CalibrationImageGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_image_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationImageGetResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    image_base64: str = Field()


class CalibrationImageMetadataListRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_image_metadata_list"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationImageMetadataListRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    image_resolution: ImageResolution = Field()


class CalibrationImageMetadataListResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_image_metadata_list"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationImageMetadataListResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    metadata_list: list[CalibrationImageMetadata] = Field(default_factory=list)


class CalibrationImageMetadataUpdateRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_image_metadata_update"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationImageMetadataUpdateRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    image_identifier: str = Field()
    image_state: CalibrationImageState = Field()
    image_label: str | None = Field(default=None)


class CalibrationResolutionListRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_detector_resolutions_list"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationResolutionListRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class CalibrationResolutionListResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_detector_resolutions_list"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationResolutionListResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    resolutions: list[ImageResolution] = Field()


class CalibrationResultGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_result_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationResultGetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    result_identifier: str = Field()


class CalibrationResultGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_result_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationResultGetResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    intrinsic_calibration: IntrinsicCalibration = Field()


class CalibrationResultGetActiveRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_result_active_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationResultGetActiveRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class CalibrationResultGetActiveResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_result_active_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationResultGetActiveResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    intrinsic_calibration: IntrinsicCalibration | None = Field()


class CalibrationResultMetadataListRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_result_metadata_list"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationResultMetadataListRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    image_resolution: ImageResolution = Field()


class CalibrationResultMetadataListResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_result_metadata_list"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationResultMetadataListResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    metadata_list: list[CalibrationResultMetadata] = Field(default_factory=list)


class CalibrationResultMetadataUpdateRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_calibration_result_metadata_update"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CalibrationResultMetadataUpdateRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    result_identifier: str = Field()
    result_state: CalibrationResultState = Field()
    result_label: str | None = Field(default=None)


class CameraImageGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_image_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CameraImageGetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    format: CaptureFormat = Field()
    requested_resolution: ImageResolution | None = Field(default=None)


class CameraImageGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_image_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CameraImageGetResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    format: CaptureFormat = Field()
    image_base64: str = Field()


class CameraParametersGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_parameters_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CameraParametersGetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class CameraParametersGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_parameters_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CameraParametersGetResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    parameters: list[SerializeAsAny[KeyValueMetaAny]] = Field()


class CameraParametersSetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_parameters_set"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CameraParametersSetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    parameters: list[SerializeAsAny[KeyValueSimpleAny]] = Field()


class CameraParametersSetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_parameters_set"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CameraParametersSetResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    resolution: ImageResolution = Field()  # Sometimes parameter changes may result in changes of resolution


class CameraResolutionGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_resolution_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CameraResolutionGetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class CameraResolutionGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_camera_resolution_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return CameraResolutionGetResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    resolution: ImageResolution = Field()


class DetectorFrameGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_frame_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return DetectorFrameGetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    include_detected: bool = Field(default=True)
    include_rejected: bool = Field(default=True)


class DetectorFrameGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_frame_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return DetectorFrameGetResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    frame: DetectorFrame = Field()


class DetectorStartRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_start"

    @staticmethod
    def parsable_type_identifier() -> str:
        return DetectorStartRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class DetectorStopRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_stop"

    @staticmethod
    def parsable_type_identifier() -> str:
        return DetectorStopRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class ImageRecorderClearRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_image_recorder_clear"

    @staticmethod
    def parsable_type_identifier() -> str:
        return ImageRecorderClearRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class ImageRecorderGetStateRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_image_recorder_get_state"

    @staticmethod
    def parsable_type_identifier() -> str:
        return ImageRecorderGetStateRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class ImageRecorderGetStateResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_image_recorder_get_state"

    @staticmethod
    def parsable_type_identifier() -> str:
        return ImageRecorderGetStateResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    remaining_time_seconds: float = Field()
    image_count: int = Field()


class ImageRecorderRetrieveRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_image_recorder_retrieve"

    @staticmethod
    def parsable_type_identifier() -> str:
        return ImageRecorderRetrieveRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class ImageRecorderRetrieveResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_image_recorder_retrieve"

    @staticmethod
    def parsable_type_identifier() -> str:
        return ImageRecorderRetrieveResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    archive_base64: str = Field()


class ImageRecorderStartRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_image_recorder_start"

    @staticmethod
    def parsable_type_identifier() -> str:
        return ImageRecorderStartRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    duration_seconds: float = Field(default=30.0)


class ImageRecorderStopRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_image_recorder_stop"

    @staticmethod
    def parsable_type_identifier() -> str:
        return ImageRecorderStopRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class MarkerParametersGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_marker_parameters_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return MarkerParametersGetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class MarkerParametersGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "detector_marker_parameters_get"

    @staticmethod
    def parsable_type_identifier() -> str:
        return MarkerParametersGetResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    parameters: list[SerializeAsAny[KeyValueMetaAny]] = Field()


class MarkerParametersSetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "detector_marker_parameters_set"

    @staticmethod
    def parsable_type_identifier() -> str:
        return MarkerParametersSetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    parameters: list[SerializeAsAny[KeyValueSimpleAny]] = Field()
