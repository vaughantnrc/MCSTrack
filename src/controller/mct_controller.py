from .configuration import \
    MCTComponentConfig, \
    MCTConfiguration
from .connection import \
    Connection
from .routing import \
    CallbackRouter, \
    ConnectionRouter
from .sequencing import \
    AbstractSequencer, \
    DetectorFrameGetSequencer, \
    DetectorStartupSequencer, \
    MixerFrameGetSequencer, \
    MixerStartupSequencer, \
    TimeSyncSequencer
from src.common import \
    BaseSink, \
    CSVPoseSink, \
    DetectorFrame, \
    ExtrinsicCalibration, \
    IntrinsicCalibration, \
    KeyValueMetaAny, \
    MCTRequestSeries, \
    MCTResponseSeries, \
    MixerFrame, \
    SeverityLabel, \
    StatusMessageSource, \
    Target
from src.detector import \
    IntrinsicCalibrationCalculateResponse, \
    IntrinsicCalibrationImageAddResponse, \
    IntrinsicCalibrationImageGetResponse, \
    IntrinsicCalibrationImageMetadataListResponse, \
    IntrinsicCalibrationResolutionListResponse, \
    IntrinsicCalibrationResultGetResponse, \
    IntrinsicCalibrationResultGetActiveResponse, \
    IntrinsicCalibrationResultMetadataListResponse, \
    DETECTOR_RESPONSE_TYPES
from src.mixer import \
    ExtrinsicCalibrationCalculateResponse, \
    ExtrinsicCalibrationImageAddResponse, \
    ExtrinsicCalibrationImageGetResponse, \
    ExtrinsicCalibrationImageMetadataListResponse, \
    ExtrinsicCalibrationResultGetResponse, \
    ExtrinsicCalibrationResultGetActiveResponse, \
    ExtrinsicCalibrationResultMetadataListResponse, \
    MIXER_RESPONSE_TYPES
import datetime
from enum import StrEnum
import hjson
from ipaddress import IPv4Address
import logging
import os
from pydantic import ValidationError
from typing import Final
import uuid


logger = logging.getLogger(__name__)

_TIME_SYNC_DEFAULT_SAMPLE_COUNT: Final[int] = 5


class TemporalOffset:
    network_latency_milliseconds: int
    clock_offset_milliseconds: int  # how much time to be ADDED to go from controller time to component
    def __init__(
        self,
        network_latency_milliseconds: int,
        clock_offset_milliseconds: int
    ):
        self.network_latency_milliseconds = network_latency_milliseconds
        self.clock_offset_milliseconds = clock_offset_milliseconds


class DetectorLiveData:
    temporal_offset: TemporalOffset | None
    camera_parameters: list[KeyValueMetaAny] | None
    annotator_parameters: list[KeyValueMetaAny] | None
    intrinsic_calibration: IntrinsicCalibration | None
    extrinsic_calibration: ExtrinsicCalibration | None
    frame: DetectorFrame | None
    def __init__(self):
        self.temporal_offset = None
        self.camera_parameters = None
        self.annotator_parameters = None
        self.intrinsic_calibration = None
        self.extrinsic_calibration = None
        self.frame = None


class MixerLiveData:
    temporal_offset: TemporalOffset | None
    pose_solver_parameters: list[KeyValueMetaAny]
    frame: MixerFrame | None
    def __init__(self):
        self.temporal_offset = None
        self.pose_solver_parameters = None
        self.frame = None


class MCTController:

    class State(StrEnum):
        IDLE = "Idle"
        CONNECTING = "Connecting"
        SYNCING = "Syncing"
        STARTING = "Starting"
        RUNNING = "Running"
        STOPPING = "Stopping"
        DISCONNECTING = "Disconnecting"

    _status_message_source: StatusMessageSource
    _sink_type_registry: dict[str, type[BaseSink]]

    _state: State
    _detector_live_data: dict[str, DetectorLiveData | None]
    _mixer_live_data: dict[str, MixerLiveData | None]
    _sinks: list[BaseSink]
    _connection_router: ConnectionRouter
    _callback_router: CallbackRouter
    _active_sequencers: list[AbstractSequencer]

    def __init__(
        self,
        serial_identifier: str,
        send_status_messages_to_logger: bool = False
    ):
        super().__init__(
            status_source_label=serial_identifier,
            send_status_messages_to_logger=send_status_messages_to_logger)
        self._status_message_source = StatusMessageSource(
            source_label="controller",
            send_to_logger=True)
        self._sink_type_registry = {"csv": CSVPoseSink}
        self._state = MCTController.State.IDLE
        self._detector_live_data = dict()
        self._mixer_live_data = dict()
        self._sinks = list()
        self._connection_router = ConnectionRouter()
        self._callback_router = CallbackRouter()

    def add_connections_from_configuration(
        self,
        configuration: MCTConfiguration
    ):
        def is_valid_ip_address(connection: MCTComponentConfig) -> bool:
            try:
                IPv4Address(connection.ip_address)
            except ValueError:
                self._status_message_source.enqueue_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Connection {connection.label} has invalid IP address {connection.ip_address}. "
                            "It will be skipped.")
                return False
            if connection.port < 0 or connection.port > 65535:
                self._status_message_source.enqueue_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Connection {connection.label} has invalid port {connection.port}. "
                            "It will be skipped.")
                return False
            return True

        for detector in configuration.detectors:
            if not is_valid_ip_address(detector):
                continue
            component_address: Connection.ComponentAddress = Connection.ComponentAddress(
                label=detector.label,
                role="detector",
                ip_address=IPv4Address(detector.ip_address),
                port=detector.port)
            self._connection_router.add_connection(
                component_address=component_address,
                supported_response_types=DETECTOR_RESPONSE_TYPES,
                status_message_source=self._status_message_source)
            self._detector_live_data[component_address.label] = None
        for mixer in configuration.mixers:
            if not is_valid_ip_address(mixer):
                continue
            component_address: Connection.ComponentAddress = Connection.ComponentAddress(
                label=mixer.label,
                role="mixer",
                ip_address=IPv4Address(mixer.ip_address),
                port=mixer.port)
            self._connection_router.add_connection(
                component_address=component_address,
                supported_response_types=MIXER_RESPONSE_TYPES,
                status_message_source=self._status_message_source)
            self._mixer_live_data[component_address.label] = None

    def get_component_labels(self) -> list[str]:
        return self.get_detector_labels() + self.get_mixer_labels()

    def get_detector_labels(self) -> list[str]:
        return list(self._detector_live_data.keys())

    def get_mixer_labels(self) -> list[str]:
        return list(self._mixer_live_data.keys())

    def get_connection_reports(self) -> list[Connection.Report]:
        return self._connection_router.get_connection_reports()

    def get_detector_live_data(
        self,
        detector_label: str
    ) -> DetectorLiveData | None:
        """
        returns None if the detector does not exist, or if it has not been started.
        """
        return self._detector_live_data.get(detector_label, None)

    def get_mixer_live_data(
        self,
        mixer_label: str
    ) -> MixerLiveData | None:
        """
        returns None if the pose solver does not exist, or has not been started, or if it has not yet gotten frames.
        """
        return self._mixer_live_data.get(mixer_label, None)

    def get_state(self) -> State:
        return self._state

    def register_sink_type(self, implementation_str: str, sink_type: type[BaseSink]) -> None:
        self._sink_type_registry[implementation_str] = sink_type

    def reset(self):
        self._state = MCTController.State.IDLE
        self._sinks.clear()
        self._detector_live_data.clear()
        self._mixer_live_data.clear()
        self._callback_router.reset()
        self._connection_router.reset()

    def request_send_custom(
        self,
        connection_label: str,
        request_series: MCTRequestSeries,
        callback: CallbackRouter.CallbackFunction = None,
        passthrough_arguments: dict[str, ...] | None = None
    ) -> uuid.UUID:
        """
        MCTRequestSeries must NOT contain any previously-sent request ID
        """
        self._connection_router.enqueue_request_series(
            label=connection_label,
            request_series=request_series)
        if callback is not None:
            if passthrough_arguments is None:
                passthrough_arguments = dict()
            self._callback_router.add_callback(
                request_id=request_series.request_id,
                callback=callback,
                passthrough_arguments=passthrough_arguments)

    def start_from_configuration_filepath(
        self,
        input_configuration_filepath: str
    ) -> None:
        if self._state != MCTController.State.IDLE:
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
        self.reset()
        for sink_configuration in configuration.sinks:
            if sink_configuration.implementation not in self._sink_type_registry:
                self._status_message_source.enqueue_status_message(
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
        if self._state != MCTController.State.IDLE:
            raise RuntimeError("Cannot start up if controller isn't first stopped.")
        self._connection_router.start_up()
        self._state = MCTController.State.CONNECTING

    def shut_down(self) -> None:
        if self._state != MCTController.State.RUNNING:
            raise RuntimeError("Cannot shut down if controller isn't first running.")
        self._connection_router.shut_down()
        self._state = MCTController.State.STOPPING

    # Right now this function doesn't update on its own - must be called externally and regularly
    def update(
        self
    ) -> None:
        self._connection_router.update()

        if self._state == MCTController.State.IDLE:
            return

        if self._state == MCTController.State.DISCONNECTING:
            if self._connection_router.is_shut_down_finished():
                self._state = MCTController.State.IDLE
                return
            else:
                return

        if self._state == MCTController.State.CONNECTING:
            if not self._connection_router.is_start_up_finished():
                return
            self._state = MCTController.State.SYNCING
            time_sync_sequencer: TimeSyncSequencer = TimeSyncSequencer(
                status_message_source=self._status_message_source,
                connection_router=self._connection_router,
                callback_router=self._callback_router)
            time_sync_sequencer.begin(
                sample_count=_TIME_SYNC_DEFAULT_SAMPLE_COUNT,
                component_labels=self.get_component_labels())
            self._active_sequencers.append(time_sync_sequencer)

        if self._state == MCTController.State.SYNCING:
            time_sync_sequencer: TimeSyncSequencer = self._active_sequencers[0]
            if time_sync_sequencer.is_finished():
                detector_labels: list[str] = self.get_detector_labels()
                for detector_label in detector_labels:
                    self._detector_live_data[detector_label].temporal_offset = TemporalOffset(
                        network_latency_milliseconds=time_sync_sequencer.data_by_component_label[detector_label].network_latency_milliseconds,
                        clock_offset_milliseconds=time_sync_sequencer.data_by_component_label[detector_label].clock_offset_milliseconds)
                mixer_labels: list[str] = self.get_mixer_labels()
                for mixer_label in mixer_labels:
                    self._mixer_live_data[mixer_label].temporal_offset = TemporalOffset(
                        network_latency_milliseconds=time_sync_sequencer.data_by_component_label[mixer_label].network_latency_milliseconds,
                        clock_offset_milliseconds=time_sync_sequencer.data_by_component_label[mixer_label].clock_offset_milliseconds)
                detector_startup_sequencer: DetectorStartupSequencer = DetectorStartupSequencer(
                    status_message_source=self._status_message_source,
                    connection_router=self._connection_router,
                    callback_router=self._callback_router)
                self._confi
                detector_startup_sequencer.begin(
                    detector_labels=detector_labels,
                    mixer_labels=mixer_labels,)
                self._active_sequencers.append(time_sync_sequencer)

        response_series_lists: list[list[MCTResponseSeries]] = self._connection_router.dequeue_response_series_lists()
        for response_series_list in response_series_lists:
            for response_series in response_series_list:
                self._callback_router.handle_callback(response_series)



        if self._state == MCTController.State.STARTING and \
           self._startup_state == MCTController.StartupState.CONNECTING:
            if self._connection_router.is_start_up_finished():
                self._advance_startup_state()
        elif self._state == MCTController.State.STOPPING:
            if self._connection_router.is_shut_down_finished():
                self._state = MCTController.State.STOPPED

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

    # def handle_response_calibration_result_get_active(
    #     self,
    #     response: IntrinsicCalibrationResultGetActiveResponse,
    #     component_label: str
    # ) -> None:
    #     detector_cache: DetectorCache = self._detector_caches[component_label]
    #     if response.intrinsic_calibration is None:
    #         if detector_cache.current_resolution is None:
    #             self.status_message_source.enqueue_status_message(
    #                 severity=SeverityLabel.ERROR,
    #                 message=f"No calibration was found for detector {component_label}, and failed to get resolution.")
    #             return
    #         self.status_message_source.enqueue_status_message(
    #             severity=SeverityLabel.WARNING,
    #             message=f"No calibration was found for detector {component_label}. "
    #                     f"Zero parameters for active resolution {detector_cache.current_resolution} will be used.")
    #         detector_cache.current_intrinsic_parameters = IntrinsicParameters.generate_zero_parameters(
    #             resolution_x_px=detector_cache.current_resolution.x_px,
    #             resolution_y_px=detector_cache.current_resolution.y_px)
    #         return
    #     detector_cache.current_intrinsic_parameters = response.intrinsic_calibration.calibrated_values
