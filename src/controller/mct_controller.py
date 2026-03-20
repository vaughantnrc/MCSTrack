from .configuration import \
    MCTComponentConfig, \
    MCTConfiguration
from .connection import \
    Connection
from .router import Router
from src.common import \
    BaseSink, \
    CSVPoseSink, \
    DequeueStatusMessagesResponse, \
    DetectorFrame, \
    EmptyResponse, \
    ErrorResponse, \
    ImageResolution, \
    IntrinsicParameters, \
    KeyValueSimpleAny, \
    Matrix4x4, \
    MCTComponent, \
    MCTError, \
    MCTRequest, \
    MCTRequestSeries, \
    MCTResponse, \
    MCTResponseSeries, \
    MixerFrame, \
    SeverityLabel, \
    StatusMessageSource, \
    Target, \
    TimestampGetRequest, \
    TimestampGetResponse, \
    TimeSyncStartRequest, \
    TimeSyncStopRequest
from src.detector import \
    AnnotatorParametersGetResponse, \
    CameraImageGetResponse, \
    CameraParametersGetResponse, \
    CameraParametersSetResponse, \
    CameraResolutionGetRequest, \
    CameraResolutionGetResponse, \
    DetectorFrameGetRequest, \
    DetectorFrameGetResponse, \
    IntrinsicCalibrationCalculateResponse, \
    IntrinsicCalibrationImageAddResponse, \
    IntrinsicCalibrationImageGetResponse, \
    IntrinsicCalibrationImageMetadataListResponse, \
    IntrinsicCalibrationResolutionListResponse, \
    IntrinsicCalibrationResultGetResponse, \
    IntrinsicCalibrationResultGetActiveRequest, \
    IntrinsicCalibrationResultGetActiveResponse, \
    IntrinsicCalibrationResultMetadataListResponse
from src.mixer import \
    ExtrinsicCalibrationCalculateResponse, \
    ExtrinsicCalibrationImageAddResponse, \
    ExtrinsicCalibrationImageGetResponse, \
    ExtrinsicCalibrationImageMetadataListResponse, \
    ExtrinsicCalibrationResultGetResponse, \
    ExtrinsicCalibrationResultGetActiveResponse, \
    ExtrinsicCalibrationResultMetadataListResponse, \
    MixerUpdateIntrinsicParametersRequest, \
    PoseSolverAddDetectorFrameRequest, \
    PoseSolverAddTargetResponse, \
    PoseSolverGetPosesRequest, \
    PoseSolverGetPosesResponse, \
    PoseSolverSetExtrinsicRequest
import datetime
from enum import IntEnum, StrEnum
import hjson
from ipaddress import IPv4Address
import logging
import numpy
import os
from pydantic import ValidationError
from typing import Callable, Final, TypeVar
import uuid

logger = logging.getLogger(__name__)
ConnectionType = TypeVar('ConnectionType', bound=Connection)


_ROLE_LABEL: Final[str] = "controller"
_TIME_SYNC_SAMPLE_MAXIMUM_COUNT: Final[int] = 5


class ResponseSeriesNotExpected(MCTError):
    pass


class BaseCache:
    request_id: uuid.UUID | None  # TODO: ???
    init_request_id: uuid.UUID | None
    deinit_request_id: uuid.UUID | None
    network_latency_samples_seconds: list[float]
    network_latency_seconds: float
    network_plus_offset_samples_seconds: list[float]
    controller_offset_samples_seconds: list[float]
    controller_offset_seconds: float  # how much time to be ADDED to go from controller time to component
    def __init__(self):
        self.request_id = None
        self.init_request_id = None
        self.deinit_request_id = None
        self.reset_time_sync_stats()
    def reset_time_sync_stats(self):
        self.network_latency_samples_seconds = list()
        self.network_latency_seconds = 0.0
        self.network_plus_offset_samples_seconds = list()
        self.controller_offset_samples_seconds = list()
        self.controller_offset_seconds = 0.0


class DetectorCache(BaseCache):
    configured_transform_to_reference: Matrix4x4 | None
    configured_camera_parameters: list[KeyValueSimpleAny] | None
    configured_marker_parameters: list[KeyValueSimpleAny] | None
    current_resolution: ImageResolution | None
    current_intrinsic_parameters: IntrinsicParameters | None
    latest_frame: DetectorFrame | None
    def __init__(self):
        super().__init__()
        self.configured_transform_to_reference = None
        self.configured_camera_parameters = None
        self.configured_marker_parameters = None
        self.current_resolution = None
        self.current_intrinsic_parameters = None
        self.latest_frame = None


class MixerCache(BaseCache):
    configured_solver_parameters: list[KeyValueSimpleAny] | None
    configured_targets: list[Target] | None
    detector_timestamps: dict[str, datetime.datetime]
    latest_frame: MixerFrame | None
    def __init__(self):
        super().__init__()
        self.configured_solver_parameters = None
        self.configured_targets = None
        self.detector_timestamps = dict()
        self.latest_frame = None


class MCTController:

    class Status(StrEnum):
        STOPPED = "Idle"
        STARTING = "Starting"
        RUNNING = "Running"
        STOPPING = "Stopping"

    class StartupState(IntEnum):
        INITIAL = 0
        CONNECTING = 1
        TIME_SYNC_START = 2
        TIME_SYNC_STOP = 3
        # TODO: Here: Start, send params, etc
        GET_INTRINSICS = 4
        SET_INTRINSICS = 5

    _status_message_source: StatusMessageSource
    _status: Status
    _startup_state: StartupState

    _router: Router
    _detector_caches: dict[str, DetectorCache]
    _mixer_caches: dict[str, MixerCache]
    _pending_request_ids: list[uuid.UUID]

    _sinks: list[BaseSink]
    _sink_type_registry: dict[str, type[BaseSink]]
    _recording_detector: bool
    _recording_pose_solver: bool
    _recording_save_path: str | None

    _time_sync_sample_count: int

    def __init__(
        self,
        serial_identifier: str,
        send_status_messages_to_logger: bool = False
    ):
        super().__init__(
            status_source_label=serial_identifier,
            send_status_messages_to_logger=send_status_messages_to_logger)
        self.status_message_source = StatusMessageSource(
            source_label="controller",
            send_to_logger=True)

        self._router = Router()
        self._sink_type_registry = {"csv": CSVPoseSink}

        # _reset is responsible for creating and restoring the initial state; __init__ calls it to avoid duplication
        self._reset()

    def add_connections_from_configuration(
        self,
        configuration: MCTConfiguration
    ):
        def is_valid_ip_address(connection: MCTComponentConfig) -> bool:
            try:
                IPv4Address(connection.ip_address)
            except ValueError:
                self.add_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Connection {connection.label} has invalid IP address {connection.ip_address}. "
                            "It will be skipped.")
                return False
            if connection.port < 0 or connection.port > 65535:
                self.add_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Connection {connection.label} has invalid port {connection.port}. "
                            "It will be skipped.")
                return False
            return True

        response_type_list: list[MCTResponse] = [
            AnnotatorParametersGetResponse,
            CameraImageGetResponse,
            CameraParametersGetResponse,
            CameraParametersSetResponse,
            CameraResolutionGetResponse,
            DetectorFrameGetResponse,
            DequeueStatusMessagesResponse,
            EmptyResponse,
            ErrorResponse,
            ExtrinsicCalibrationCalculateResponse,
            ExtrinsicCalibrationImageAddResponse,
            ExtrinsicCalibrationImageGetResponse,
            ExtrinsicCalibrationImageMetadataListResponse,
            ExtrinsicCalibrationResultGetResponse,
            ExtrinsicCalibrationResultGetActiveResponse,
            ExtrinsicCalibrationResultMetadataListResponse,
            IntrinsicCalibrationCalculateResponse,
            IntrinsicCalibrationImageAddResponse,
            IntrinsicCalibrationImageGetResponse,
            IntrinsicCalibrationImageMetadataListResponse,
            IntrinsicCalibrationResolutionListResponse,
            IntrinsicCalibrationResultGetResponse,
            IntrinsicCalibrationResultGetActiveResponse,
            IntrinsicCalibrationResultMetadataListResponse,
            PoseSolverAddTargetResponse,
            PoseSolverGetPosesResponse,
            TimestampGetResponse]
        response_type_dict: dict[str, type[MCTResponse]] = {
            response_type.type_identifier(): response_type
            for response_type in response_type_list}

        for detector in configuration.detectors:
            if not is_valid_ip_address(detector):
                continue
            component_address: Connection.ComponentAddress = Connection.ComponentAddress(
                label=detector.label,
                role="detector",
                ip_address=IPv4Address(detector.ip_address),
                port=detector.port)
            self._router.add_connection(
                component_address=component_address,
                supported_response_types=response_type_dict)
            self._detector_caches[component_address.label] = DetectorCache()
            self._detector_caches[component_address.label].configured_transform_to_reference = \
                detector.fixed_transform_to_reference
            self._detector_caches[component_address.label].configured_camera_parameters = detector.camera_parameters
            self._detector_caches[component_address.label].configured_marker_parameters = detector.marker_parameters
        for mixer in configuration.mixers:
            if not is_valid_ip_address(mixer):
                continue
            component_address: Connection.ComponentAddress = Connection.ComponentAddress(
                label=mixer.label,
                role="mixer",
                ip_address=IPv4Address(mixer.ip_address),
                port=mixer.port)
            self._router.add_connection(
                component_address=component_address,
                supported_response_types=response_type_dict)
            self._mixer_caches[component_address.label] = MixerCache()
            self._mixer_caches[component_address.label].configured_solver_parameters = mixer.solver_parameters
            self._mixer_caches[component_address.label].configured_targets = mixer.targets

    def _advance_startup_state(self) -> None:
        if len(self._pending_request_ids) <= 0 and self._startup_state == MCTController.StartupState.CONNECTING:
            self.status_message_source.enqueue_status_message(
                severity=SeverityLabel.DEBUG,
                message="CONNECTING complete")
            detector_labels: list[str] = self.get_active_detector_labels()
            mixer_labels: list[str] = self.get_active_mixer_labels()
            component_caches: list[BaseCache] = \
                [self._detector_caches[label] for label in detector_labels] + \
                [self._mixer_caches[label] for label in mixer_labels]
            for component_cache in component_caches:
                component_cache.reset_time_sync_stats()
            component_labels: list[str] = detector_labels + mixer_labels
            request_series: MCTRequestSeries = MCTRequestSeries(series=[TimeSyncStartRequest()])
            for component_label in component_labels:
                self._pending_request_ids.append(
                    self.request_series_push(
                        connection_label=component_label,
                        request_series=request_series))
            self._time_sync_sample_count = 0
            self._startup_state = MCTController.StartupState.TIME_SYNC_START
        if len(self._pending_request_ids) <= 0 and self._startup_state == MCTController.StartupState.TIME_SYNC_START:
            self.status_message_source.enqueue_status_message(
                severity=SeverityLabel.DEBUG,
                message="TIME_SYNC complete")
            component_labels: list[str] = self.get_active_detector_labels() + self.get_active_mixer_labels()
            request_series: MCTRequestSeries = MCTRequestSeries(series=[
                TimestampGetRequest(requester_timestamp_utc_iso8601=datetime.datetime.now(tz=datetime.timezone.utc).isoformat())])
            for component_label in component_labels:
                self._pending_request_ids.append(
                    self.request_series_push(
                        connection_label=component_label,
                        request_series=request_series))
            self._time_sync_sample_count += 1
            if self._time_sync_sample_count >= _TIME_SYNC_SAMPLE_MAXIMUM_COUNT:
                request_series: MCTRequestSeries = MCTRequestSeries(series=[TimeSyncStopRequest()])
                for component_label in component_labels:
                    self._pending_request_ids.append(
                        self.request_series_push(
                            connection_label=component_label,
                            request_series=request_series))
                self._startup_state = MCTController.StartupState.TIME_SYNC_STOP
        if len(self._pending_request_ids) <= 0 and self._startup_state == MCTController.StartupState.TIME_SYNC_STOP:
            self.status_message_source.enqueue_status_message(
                severity=SeverityLabel.DEBUG,
                message="STARTING_CAPTURE complete")
            detector_labels: list[str] = self.get_active_detector_labels()
            for detector_label in detector_labels:
                request_series: MCTRequestSeries = MCTRequestSeries(
                    series=[
                        CameraResolutionGetRequest(),
                        IntrinsicCalibrationResultGetActiveRequest()])
                self._pending_request_ids.append(self.request_series_push(
                    connection_label=detector_label,
                    request_series=request_series))
            self._startup_state = MCTController.StartupState.GET_INTRINSICS
        if len(self._pending_request_ids) <= 0 and self._startup_state == MCTController.StartupState.GET_INTRINSICS:
            self.status_message_source.enqueue_status_message(
                severity=SeverityLabel.DEBUG,
                message="GET_INTRINSICS complete")
            mixer_labels: list[str] = self.get_active_mixer_labels()
            for pose_solver_label in mixer_labels:
                requests: list[MCTRequest] = list()
                for detector_label in self.get_active_detector_labels():
                    detector_cache: DetectorCache = self._detector_caches[detector_label]
                    if detector_cache.current_intrinsic_parameters is not None:
                        requests.append(MixerUpdateIntrinsicParametersRequest(
                            detector_label=detector_label,
                            intrinsic_parameters=detector_cache.current_intrinsic_parameters))
                    if detector_cache.configured_transform_to_reference is not None:
                        requests.append(PoseSolverSetExtrinsicRequest(
                            detector_label=detector_label,
                            transform_to_reference=detector_cache.configured_transform_to_reference))
                request_series: MCTRequestSeries = MCTRequestSeries(series=requests)
                self._pending_request_ids.append(self.request_series_push(
                    connection_label=pose_solver_label,
                    request_series=request_series))
            self._startup_state = MCTController.StartupState.SET_INTRINSICS
        if len(self._pending_request_ids) <= 0 and self._startup_state == MCTController.StartupState.SET_INTRINSICS:
            self.status_message_source.enqueue_status_message(
                severity=SeverityLabel.DEBUG,
                message="SET_INTRINSICS complete")
            self._startup_state = MCTController.StartupState.INITIAL
            self._status = MCTController.Status.RUNNING

    def get_active_detector_labels(self) -> list[str]:
        return [
            component_label
            for component_label in self._detector_caches.keys()
            if self._router.get_connection(label=component_label).is_active()]

    def get_active_mixer_labels(self) -> list[str]:
        return [
            component_label
            for component_label in self._mixer_caches.keys()
            if self._router.get_connection(label=component_label).is_active()]

    def get_connection_reports(self) -> list[Connection.Report]:
        return self._router.get_connection_reports()

    def get_live_detector_intrinsics(
        self,
        detector_label: str
    ) -> IntrinsicParameters | None:
        """
        returns None if the detector does not exist, or if it has not been started.
        """
        if detector_label not in self._detector_caches:
            return None
        return self._detector_caches[detector_label].current_intrinsic_parameters

    def get_live_detector_frame(
        self,
        detector_label: str
    ) -> DetectorFrame | None:
        """
        returns None if the detector does not exist, or has not been started, or if it has not yet gotten frames.
        """
        if detector_label not in self._detector_caches:
            return None
        return self._detector_caches[detector_label].latest_frame

    def get_live_pose_solver_frame(
        self,
        mixer_label: str
    ) -> MixerFrame | None:
        """
        returns None if the pose solver does not exist, or has not been started, or if it has not yet gotten frames.
        """
        if mixer_label not in self._mixer_caches:
            return None
        return self._mixer_caches[mixer_label].latest_frame

    def get_status(self) -> Status:
        return self._status

    def handle_error_response(
        self,
        response: ErrorResponse,
        component_label: str
    ):
        self.status_message_source.enqueue_status_message(
            severity=SeverityLabel.ERROR,
            message=f"Received error from {component_label}: {response.message}")

    def handle_response_calibration_result_get_active(
        self,
        response: IntrinsicCalibrationResultGetActiveResponse,
        component_label: str
    ) -> None:
        if component_label not in self._detector_caches:
            self.status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Failed to find DetectorCache associated with label {component_label}.")
            return
        detector_cache: DetectorCache = self._detector_caches[component_label]
        if response.intrinsic_calibration is None:
            if detector_cache.current_resolution is None:
                self.status_message_source.enqueue_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"No calibration was found for detector {component_label}, and failed to get resolution.")
                return
            self.status_message_source.enqueue_status_message(
                severity=SeverityLabel.WARNING,
                message=f"No calibration was found for detector {component_label}. "
                        f"Zero parameters for active resolution {detector_cache.current_resolution} will be used.")
            detector_cache.current_intrinsic_parameters = IntrinsicParameters.generate_zero_parameters(
                resolution_x_px=detector_cache.current_resolution.x_px,
                resolution_y_px=detector_cache.current_resolution.y_px)
            return
        detector_cache.current_intrinsic_parameters = response.intrinsic_calibration.calibrated_values

    def handle_response_camera_resolution_get(
        self,
        response: CameraResolutionGetResponse,
        component_label: str
    ) -> None:
        if component_label not in self._detector_caches:
            self.status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Failed to find DetectorCache associated with label {component_label}.")
            return
        detector_cache: DetectorCache = self._detector_caches[component_label]
        detector_cache.current_resolution = response.resolution

    def handle_response_detector_frame_get(
        self,
        response: DetectorFrameGetResponse,
        component_label: str
    ):
        if component_label not in self._detector_caches:
            self.status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Failed to find DetectorCache associated with label {component_label}.")
            return
        detector_cache: DetectorCache = self._detector_caches[component_label]
        frame: DetectorFrame = response.frame
        adjusted_timestamp_utc: datetime.datetime = \
            frame.timestamp_utc - datetime.timedelta(seconds=detector_cache.controller_offset_seconds)
        frame.timestamp_utc_iso8601 = adjusted_timestamp_utc.isoformat()
        detector_cache.latest_frame = frame

    def handle_response_get_poses(
        self,
        response: PoseSolverGetPosesResponse,
        component_label: str
    ) -> None:
        if component_label not in self._mixer_caches:
            self.status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Failed to find MixerCache associated with label {component_label}.")
            return
        mixer_cache: MixerCache = self._mixer_caches[component_label]
        mixer_cache.detector_poses = response.detector_poses
        mixer_cache.target_poses = response.target_poses
        mixer_cache.poses_timestamp = (
            datetime.datetime.now(tz=datetime.timezone.utc) -  # TODO: This should come from the pose solver
            datetime.timedelta(seconds=mixer_cache.controller_offset_seconds))

    def handle_response_timestamp_get(
        self,
        response: TimestampGetResponse,
        component_label: str
    ) -> None:
        cache: BaseCache
        if component_label in self._detector_caches:
            cache = self._detector_caches[component_label]
        elif component_label in self._mixer_caches:
            cache = self._mixer_caches[component_label]
        else:
            self.status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Failed to find cache associated with label {component_label}.")
            return
        utc_now: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
        requester_timestamp: datetime.datetime
        requester_timestamp = datetime.datetime.fromisoformat(response.requester_timestamp_utc_iso8601)
        round_trip_seconds: float = (utc_now - requester_timestamp).total_seconds()
        cache.network_latency_samples_seconds.append(round_trip_seconds)
        responder_timestamp: datetime.datetime
        responder_timestamp = datetime.datetime.fromisoformat(response.responder_timestamp_utc_iso8601)
        network_plus_offset_seconds: float = (responder_timestamp - requester_timestamp).total_seconds()
        cache.network_plus_offset_samples_seconds.append(network_plus_offset_seconds)
        if self._time_sync_sample_count >= _TIME_SYNC_SAMPLE_MAXIMUM_COUNT:
            cache.network_latency_seconds = float(numpy.median(cache.network_latency_samples_seconds))
            cache.controller_offset_samples_seconds = [
                network_plus_offset_sample_seconds - (cache.network_latency_seconds / 2.0)
                for network_plus_offset_sample_seconds in cache.network_plus_offset_samples_seconds]
            cache.controller_offset_seconds = float(numpy.median(cache.controller_offset_samples_seconds))
            print(f"Calculated offset to {component_label}: {cache.controller_offset_seconds}")

    def handle_response_unknown(
        self,
        response: MCTResponse,
        component_label: str
    ):
        self.status_message_source.enqueue_status_message(
            severity=SeverityLabel.ERROR,
            message=f"Received unexpected response from {component_label}: {str(type(response))}")

    def handle_response_series(
        self,
        response_series: MCTResponseSeries,
        task_description: str | None = None,
        expected_response_count: int | None = None
    ) -> bool:
        if expected_response_count is not None:
            response_count: int = len(response_series.series)
            task_text: str = str()
            if task_description is not None:
                task_text = f" during {task_description}"
            if response_count < expected_response_count:
                self.status_message_source.enqueue_status_message(
                    severity=SeverityLabel.WARNING,
                    message=f"Received a response series{task_text}, "
                            f"but it contained fewer responses ({response_count}) "
                            f"than expected ({expected_response_count}).")
            elif response_count > expected_response_count:
                self.status_message_source.enqueue_status_message(
                    severity=SeverityLabel.WARNING,
                    message=f"Received a response series{task_text}, "
                            f"but it contained more responses ({response_count}) "
                            f"than expected ({expected_response_count}).")

        success: bool = True
        response: MCTResponse
        for response in response_series.series:
            if isinstance(response, IntrinsicCalibrationResultGetActiveResponse):
                self.handle_response_calibration_result_get_active(
                    response=response,
                    component_label=response_series.responder)
            elif isinstance(response, CameraResolutionGetResponse):
                self.handle_response_camera_resolution_get(
                    response=response,
                    component_label=response_series.responder)
            elif isinstance(response, DetectorFrameGetResponse):
                self.handle_response_detector_frame_get(
                    response=response,
                    component_label=response_series.responder)
            elif isinstance(response, TimestampGetResponse):
                self.handle_response_timestamp_get(
                    response=response,
                    component_label=response_series.responder)
            elif isinstance(response, PoseSolverGetPosesResponse):
                self.handle_response_get_poses(
                    response=response,
                    component_label=response_series.responder)
            elif isinstance(response, ErrorResponse):
                self.handle_error_response(
                    response=response,
                    component_label=response_series.responder)
                success = False
            elif not isinstance(response, EmptyResponse):
                self.handle_response_unknown(
                    response=response,
                    component_label=response_series.responder)
                success = False
        return success

    def is_idle(self):
        return self._status == MCTController.Status.STOPPED

    def is_running(self):
        return self._status == MCTController.Status.RUNNING

    def is_transitioning(self):
        return self._status == MCTController.Status.STARTING or self._status == MCTController.Status.STOPPING

    def recording_start(
        self,
        save_path : str,
        record_pose_solver : bool,
        record_detector : bool
    ):

        if save_path:
            self._recording_pose_solver = record_pose_solver
            self._recording_detector = record_detector
            self._recording_save_path = save_path
        else:
            self.add_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Recording save path not defined")

    def register_sink_type(self, implementation_str: str, sink_type: type[BaseSink]) -> None:
        self._sink_type_registry[implementation_str] = sink_type

    def _reset(self):
        self._status = MCTController.Status.STOPPED
        self._startup_state = MCTController.StartupState.INITIAL

        self._pending_request_ids = list()
        self._detector_caches = dict()
        self._mixer_caches = dict()

        self._sinks = list()
        # self._sink_type_registry is excluded from reset
        self._recording_detector = False
        self._recording_pose_solver = False
        self._recording_save_path = None

        self._time_sync_sample_count = 0

    def request_series_push(
        self,
        connection_label: str,
        request_series: MCTRequestSeries
    ) -> uuid.UUID:
        return self._router.request_series_push(label=connection_label, request_series=request_series)

    def response_series_pop(
        self,
        request_series_id: uuid.UUID
    ) -> tuple[uuid.UUID | None, MCTResponseSeries | None]:
        return self._router.response_series_pop(request_series_id=request_series_id)

    def start_from_configuration_filepath(
        self,
        input_configuration_filepath: str
    ) -> None:
        if self._status != MCTController.Status.STOPPED:
            raise RuntimeError("Cannot load from configuration if controller isn't first stopped.")
        if not os.path.exists(input_configuration_filepath):
            raise IOError(f"File {input_configuration_filepath} does not exist. Configuration will not be loaded.")
        if not os.path.isfile(input_configuration_filepath):
            raise IOError(f"File {input_configuration_filepath} is not a file. Configuration will not be loaded.")
        configuration_dict: dict
        with open(input_configuration_filepath, 'r') as infile:
            configuration_dict = hjson.loads(infile.read())
        configuration: MCTConfiguration
        try:
            configuration = MCTConfiguration(**configuration_dict)
        except ValidationError as e:
            raise RuntimeError(
                f"Failed to load configuration file {input_configuration_filepath}. "
                f"Error: {e}") from None
        self._reset()
        for sink_configuration in configuration.sinks:
            if sink_configuration.implementation not in self._sink_type_registry:
                self.status_message_source.enqueue_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Unrecognized sink implementation {sink_configuration.implementation}. Skipping.")
                continue
            sink_type: type[BaseSink] = self._sink_type_registry[sink_configuration.implementation]
            sink: BaseSink = sink_type(**sink_configuration.configuration)
            self._sinks.append(sink)
        self.add_connections_from_configuration(configuration)
        self.start_up()

    def start_up(
        self
    ) -> None:
        if self._status != MCTController.Status.STOPPED:
            raise RuntimeError("Cannot start up if controller isn't first stopped.")
        self._router.start_up()
        self._startup_state = MCTController.StartupState.CONNECTING
        self._status = MCTController.Status.STARTING

    def shut_down(self) -> None:
        if self._status != MCTController.Status.RUNNING:
            raise RuntimeError("Cannot shut down if controller isn't first running.")
        self._router.shut_down()
        self._status = MCTController.Status.STOPPING

    # Right now this function doesn't update on its own - must be called externally and regularly
    def update(
        self
    ) -> None:
        self._router.update()
        if self._status == MCTController.Status.STARTING and \
           self._startup_state == MCTController.StartupState.CONNECTING:
            if self._router.is_start_up_finished():
                self._advance_startup_state()
        elif self._status == MCTController.Status.STOPPING:
            if self._router.is_shut_down_finished():
                self._status = MCTController.Status.STOPPED

        if self.is_running():
            for detector_label in self.get_active_detector_labels():
                if detector_label not in self._detector_caches:
                    self.status_message_source.enqueue_status_message(
                        severity=SeverityLabel.ERROR,
                        message=f"Failed to find DetectorCache associated with label {detector_label}.")
                    continue
                detector_cache: DetectorCache = self._detector_caches[detector_label]
                if detector_cache.request_id is not None:
                    _, detector_cache.request_id = self.update_request(
                        request_id=detector_cache.request_id)
                if detector_cache.request_id is None:
                    detector_cache.request_id = self.request_series_push(
                        connection_label=detector_label,
                        request_series=MCTRequestSeries(series=[DetectorFrameGetRequest()]))
            for mixer_label in self.get_active_mixer_labels():
                if mixer_label not in self._mixer_caches:
                    self.status_message_source.enqueue_status_message(
                        severity=SeverityLabel.ERROR,
                        message=f"Failed to find MixerCache associated with label {mixer_label}.")
                    continue
                mixer_cache: MixerCache = self._mixer_caches[mixer_label]
                if mixer_cache.request_id is not None:
                    _, mixer_cache.request_id = self.update_request(
                        request_id=mixer_cache.request_id)
                if mixer_cache.request_id is None:
                    solver_request_list: list[MCTRequest] = list()
                    detector_labels: list[str] = self.get_active_detector_labels()
                    for detector_label in detector_labels:
                        current_detector_frame: DetectorFrame | None = self.get_live_detector_frame(
                            detector_label=detector_label)
                        if current_detector_frame is None:
                            continue
                        current_detector_frame_timestamp: datetime.datetime = current_detector_frame.timestamp_utc
                        current_is_new: bool = False
                        if detector_label in mixer_cache.detector_timestamps:
                            old_detector_frame_timestamp = \
                                mixer_cache.detector_timestamps[detector_label]
                            if current_detector_frame_timestamp > old_detector_frame_timestamp:
                                current_is_new = True
                        else:
                            current_is_new = True
                        if current_is_new:
                            mixer_cache.detector_timestamps[detector_label] = \
                                current_detector_frame_timestamp
                            adjusted_detector_frame: DetectorFrame = current_detector_frame.model_copy()
                            adjusted_timestamp_utc: datetime.datetime = \
                                current_detector_frame.timestamp_utc + \
                                datetime.timedelta(seconds=mixer_cache.controller_offset_seconds)
                            adjusted_detector_frame.timestamp_utc_iso8601 = adjusted_timestamp_utc.isoformat()
                            marker_request: PoseSolverAddDetectorFrameRequest = PoseSolverAddDetectorFrameRequest(
                                detector_label=detector_label,
                                detector_frame=adjusted_detector_frame)
                            solver_request_list.append(marker_request)

                    solver_request_list.append(PoseSolverGetPosesRequest())
                    request_series: MCTRequestSeries = MCTRequestSeries(series=solver_request_list)
                    mixer_cache.request_id = self.request_series_push(
                        connection_label=mixer_label,
                        request_series=request_series)

        if len(self._pending_request_ids) > 0:
            completed_request_ids: list[uuid.UUID] = list()
            for request_id in self._pending_request_ids:
                _, remaining_request_id = self.update_request(request_id=request_id)
                if remaining_request_id is None:
                    completed_request_ids.append(request_id)
            for request_id in completed_request_ids:
                self._pending_request_ids.remove(request_id)
            if len(self._pending_request_ids) == 0:
                self._advance_startup_state()

    def update_request(
        self,
        request_id: uuid.UUID,
        task_description: str | None = None,
        expected_response_count: int | None = None
    ) -> (bool, uuid.UUID | None):
        """
        Returns a tuple of:
        - success at handling the response (False if no response has been received)
        - value that request_id shall take for subsequent iterations (None means a response series has been received)
        """

        response_series: MCTResponseSeries | None
        _, response_series = self.response_series_pop(request_series_id=request_id)
        if response_series is None:
            return False, request_id  # try again next loop

        success: bool = self.handle_response_series(
            response_series=response_series,
            task_description=task_description,
            expected_response_count=expected_response_count)
        return success, None  # We've handled the request, request_id can be set to None
