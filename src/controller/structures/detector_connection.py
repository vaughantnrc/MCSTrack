from .mct_component_address import MCTComponentAddress
from .connection import Connection
from src.common.api import \
    EmptyResponse, \
    MCTRequest, \
    MCTRequestSeries, \
    MCTResponse, \
    MCTResponseSeries
from src.common.structures import \
    DetectorFrame, \
    ImageResolution, \
    IntrinsicParameters, \
    KeyValueSimpleAny, \
    Matrix4x4
from src.detector.api import \
    CalibrationCalculateResponse, \
    CalibrationImageAddResponse, \
    CalibrationImageGetResponse, \
    CalibrationImageMetadataListResponse, \
    CalibrationResolutionListResponse, \
    CalibrationResultGetResponse, \
    CalibrationResultGetActiveResponse, \
    CalibrationResultMetadataListResponse, \
    CameraImageGetResponse, \
    CameraParametersGetResponse, \
    CameraParametersSetRequest, \
    CameraParametersSetResponse, \
    CameraResolutionGetResponse, \
    DetectorFrameGetResponse, \
    DetectorStartRequest, \
    DetectorStopRequest, \
    MarkerParametersGetResponse, \
    MarkerParametersSetRequest
import uuid


class DetectorConnection(Connection):

    configured_transform_to_reference: Matrix4x4 | None
    configured_camera_parameters: list[KeyValueSimpleAny] | None
    configured_marker_parameters: list[KeyValueSimpleAny] | None

    # These are variables used directly by the MCTController for storing data
    request_id: uuid.UUID | None
    current_resolution: ImageResolution | None
    current_intrinsic_parameters: IntrinsicParameters | None
    latest_frame: DetectorFrame | None
    recording: list[DetectorFrame] | None

    def __init__(
        self,
        component_address: MCTComponentAddress
    ):
        super().__init__(component_address=component_address)

        self.configured_transform_to_reference = None
        self.configured_camera_parameters = None
        self.configured_marker_parameters = None

        self.request_id = None
        self.current_resolution = None
        self.current_intrinsic_parameters = None
        self.latest_frame = None
        self.recording = []

    def create_deinitialization_request_series(self) -> MCTRequestSeries:
        return MCTRequestSeries(series=[DetectorStopRequest()])

    def create_initialization_request_series(self) -> MCTRequestSeries:
        series: list[MCTRequest] = [DetectorStartRequest()]
        if self.configured_camera_parameters is not None:
            series.append(CameraParametersSetRequest(parameters=self.configured_camera_parameters))
        if self.configured_marker_parameters is not None:
            series.append(MarkerParametersSetRequest(parameters=self.configured_marker_parameters))
        return MCTRequestSeries(series=series)

    def handle_deinitialization_response_series(
        self,
        response_series: MCTResponseSeries
    ) -> Connection.DeinitializationResult:
        response_count: int = len(response_series.series)
        if response_count != 1:
            self.enqueue_status_message(
                severity="warning",
                message=f"Expected exactly one response to deinitialization requests. Got {response_count}.")
        elif not isinstance(response_series.series[0], (EmptyResponse, CameraParametersSetResponse)):
            self.enqueue_status_message(
                severity="warning",
                message=f"The deinitialization response was not of the expected type EmptyResponse.")
        return Connection.DeinitializationResult.SUCCESS

    def handle_initialization_response_series(
        self,
        response_series: MCTResponseSeries
    ) -> Connection.InitializationResult:
        response_count: int = len(response_series.series)
        if response_count != 1:
            self.enqueue_status_message(
                severity="warning",
                message=f"Expected exactly one response to initialization requests. Got {response_count}.")
        elif not isinstance(response_series.series[0], EmptyResponse):
            self.enqueue_status_message(
                severity="warning",
                message=f"The initialization response was not of the expected type EmptyResponse.")
        return Connection.InitializationResult.SUCCESS

    def supported_response_types(self) -> list[type[MCTResponse]]:
        return super().supported_response_types() + [
            CalibrationCalculateResponse,
            CalibrationImageAddResponse,
            CalibrationImageGetResponse,
            CalibrationImageMetadataListResponse,
            CalibrationResolutionListResponse,
            CalibrationResultGetResponse,
            CalibrationResultGetActiveResponse,
            CalibrationResultMetadataListResponse,
            CameraImageGetResponse,
            CameraParametersGetResponse,
            CameraParametersSetResponse,
            CameraResolutionGetResponse,
            DetectorFrameGetResponse,
            MarkerParametersGetResponse]
