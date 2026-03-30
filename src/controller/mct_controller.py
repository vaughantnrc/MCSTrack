from .configuration import \
    MCTComponentConfig, \
    MCTConfiguration
from .connection import \
    Connection
from .routing import \
    CallbackRouter, \
    ConnectionRouter
from .sequencing import \
    DetectorFrameGetSequencer, \
    DetectorShutdownSequencer, \
    DetectorStartupSequencer, \
    MixerFrameGetSequencer, \
    MixerShutdownSequencer, \
    MixerStartupSequencer, \
    TimeSyncSequencer
from src.common import \
    BaseSink, \
    CSVPoseSink, \
    DetectorFrame, \
    DetectorPoseMode, \
    ExtrinsicCalibration, \
    ImageResolution, \
    IntrinsicCalibration, \
    IntrinsicParameters, \
    KeyValueMetaAny, \
    KeyValueSimpleAny, \
    Matrix4x4, \
    MCTRequest, \
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
    camera_resolution: ImageResolution | None
    camera_parameters: list[KeyValueMetaAny] | None
    annotator_parameters: list[KeyValueMetaAny] | None
    intrinsic_calibration: IntrinsicCalibration | None
    frame: DetectorFrame | None
    def __init__(self):
        self.temporal_offset = None
        self.camera_resolution = None
        self.camera_parameters = None
        self.annotator_parameters = None
        self.intrinsic_calibration = None
        self.frame = None


class MixerLiveData:
    temporal_offset: TemporalOffset | None
    pose_solver_parameters: list[KeyValueMetaAny] | None
    frame: MixerFrame | None
    def __init__(self):
        self.temporal_offset = None
        self.pose_solver_parameters = None
        self.frame = None


class MCTController:

    class State(StrEnum):
        IDLE = "Idle"
        CONNECTING = "Connecting"
        STARTING = "Starting"
        RUNNING = "Running"
        STOPPING = "Stopping"
        DISCONNECTING = "Disconnecting"

    class _Sequencers:
        time_sync_sequencer: TimeSyncSequencer | None
        detector_startup_sequencer: DetectorStartupSequencer | None
        detector_frame_get_sequencer: DetectorFrameGetSequencer | None
        detector_shutdown_sequencer: DetectorShutdownSequencer | None

        mixer_startup_sequencer: MixerStartupSequencer | None
        mixer_frame_get_sequencer: MixerFrameGetSequencer | None
        mixer_shutdown_sequencer: MixerShutdownSequencer | None

        def __init__(self):
            self.reset()
        def reset(self):
            self.time_sync_sequencer = None
            self.detector_startup_sequencer = None
            self.detector_frame_get_sequencer = None
            self.detector_shutdown_sequencer = None
            self.mixer_startup_sequencer = None
            self.mixer_frame_get_sequencer = None
            self.mixer_shutdown_sequencer = None

    _status_message_source: StatusMessageSource
    _sink_type_registry: dict[str, type[BaseSink]]

    _state: State
    _configuration: MCTConfiguration | None
    _sequencers: _Sequencers
    _detector_live_data: dict[str, DetectorLiveData | None]
    _mixer_live_data: dict[str, MixerLiveData | None]
    _sinks: list[BaseSink]
    _connection_router: ConnectionRouter
    _callback_router: CallbackRouter

    def __init__(
        self,
        controller_name: str = "mct_controller",
        send_status_messages_to_logger: bool = False
    ):
        """
        :param controller_name: When this instance logs information, associate with this name.
        :param send_status_messages_to_logger: Log messages to python's logging module.
        """
        self._status_message_source = StatusMessageSource(
            source_label=controller_name,
            send_to_logger=send_status_messages_to_logger)
        self._sink_type_registry = {"csv": CSVPoseSink}
        self._state = MCTController.State.IDLE
        self._configuration = None
        self._sequencers = MCTController._Sequencers()
        self._detector_live_data = dict()
        self._mixer_live_data = dict()
        self._sinks = list()
        self._connection_router = ConnectionRouter()
        self._callback_router = CallbackRouter()

    # =================================================================================================================
    #                                              HIGH-LEVEL CONTROL
    # =================================================================================================================

    def register_sink_type(self, implementation_str: str, sink_type: type[BaseSink]) -> None:
        self._sink_type_registry[implementation_str] = sink_type

    def configure(
        self,
        configuration: MCTConfiguration
    ) -> bool:
        """
        Returns True if there are no immediate errors.
        """
        if self._state != MCTController.State.IDLE:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message="Cannot load configuration if not in the idle state.")
            return False

        return_value: bool = True
        self._configuration = configuration

        # Create connections
        def is_valid_ip_address(connection: MCTComponentConfig) -> bool:
            try:
                IPv4Address(connection.ip_address)
            except ValueError:
                self._status_message_source.enqueue_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Connection {connection.label} has invalid IP {connection.ip_address}. Skipping.")
                return False
            if connection.port < 0 or connection.port > 65535:
                self._status_message_source.enqueue_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Connection {connection.label} has invalid port {connection.port}. "
                            "It will be skipped.")
                return False
            return True

        for detector in self._configuration.detectors:
            if not is_valid_ip_address(detector):
                return_value = False
                continue
            if detector.label in self._detector_live_data or detector.label in self._mixer_live_data:
                self._status_message_source.enqueue_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Detector Configuration contains an existing label {detector.label}. Skipping.")
                return_value = False
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
            self._detector_live_data[component_address.label] = DetectorLiveData()

        for mixer in self._configuration.mixers:
            if not is_valid_ip_address(mixer):
                return_value = False
                continue
            if mixer.label in self._detector_live_data or mixer.label in self._mixer_live_data:
                self._status_message_source.enqueue_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Mixer Configuration contains an existing label {mixer.label}. Skipping.")
                return_value = False
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
            self._mixer_live_data[component_address.label] = MixerLiveData()

        for sink_configuration in self._configuration.sinks:
            if sink_configuration.implementation not in self._sink_type_registry:
                self._status_message_source.enqueue_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Unrecognized sink implementation {sink_configuration.implementation}. Skipping.")
                return_value = False
                continue
            sink_type: type[BaseSink] = self._sink_type_registry[sink_configuration.implementation]
            sink: BaseSink = sink_type(**sink_configuration.configuration)
            self._sinks.append(sink)

        return return_value

    def configure_from_filepath(
        self,
        input_configuration_filepath: str
    ) -> bool:
        if not os.path.exists(input_configuration_filepath):
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"File {input_configuration_filepath} does not exist. Configuration will not be loaded.")
            return False
        if not os.path.isfile(input_configuration_filepath):
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"File {input_configuration_filepath} is not a file. Configuration will not be loaded.")
            return False
        self.reset()
        configuration_dict: dict
        with open(input_configuration_filepath, 'r') as infile:
            configuration_dict = hjson.loads(infile.read())
        configuration: MCTConfiguration
        try:
            configuration = MCTConfiguration(**configuration_dict)
        except ValidationError as e:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Failed to load configuration file {input_configuration_filepath}. Error: {e}")
            return False
        return self.configure(configuration)

    def start_up(self) -> bool:
        """
        Returns True if there are no immediate errors.
        """
        if self._configuration is None:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message="Cannot start up if no configuration has been loaded.")
            return False
        if self._state != MCTController.State.IDLE:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message="Cannot start up if not in the idle state. Has the controller already been started?")
            return False
        if len(self._configuration.detectors) <= 0:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message="No detectors were specified in the configuration.")
            return False
        self._connection_router.start_up()
        self._state = MCTController.State.CONNECTING
        return True

    def shut_down(self) -> bool:
        """
        Returns True if there are no immediate errors.
        """
        if self._state != MCTController.State.RUNNING:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message="Cannot shut down if not in the running state. Try checking the status first?")
            return False
        self._state = MCTController.State.STOPPING
        return True

    def reset(self):
        if self._state == MCTController.State.RUNNING:
            self._connection_router.shut_down()
        self._state = MCTController.State.IDLE
        self._configuration = None
        self._sequencers.reset()
        self._detector_live_data.clear()
        self._mixer_live_data.clear()
        self._sinks.clear()
        self._callback_router.reset()
        self._connection_router.reset()

    # Right now this function doesn't update on its own - must be called externally and as frequently as possible
    def update(self) -> None:
        self._connection_router.update()

        if self._state == MCTController.State.IDLE:
            return

        if self._state == MCTController.State.DISCONNECTING:
            if self._connection_router.is_shut_down_finished():
                self._state = MCTController.State.IDLE
                self._sequencers.reset()
                return
            else:
                return

        if self._state == MCTController.State.CONNECTING:
            if not self._connection_router.is_start_up_finished():
                return
            self._state = MCTController.State.STARTING
            self._sequencers.time_sync_sequencer = TimeSyncSequencer(**self._sequencer_init_args())
            self._sequencers.time_sync_sequencer.begin(
                sample_count=_TIME_SYNC_DEFAULT_SAMPLE_COUNT,
                component_labels=self.get_remote_labels())

        # !! NO RETURN STATEMENTS AFTER THIS POINT !! - we want to make sure callback code at the end runs

        if self._state == MCTController.State.STARTING:
            if (
                self._sequencers.time_sync_sequencer is not None and
                self._sequencers.time_sync_sequencer.is_finished() and
                self._sequencers.detector_startup_sequencer is None
            ):
                for detector_label, detector_live_data in self._detector_live_data.items():
                    component_data: TimeSyncSequencer.ComponentData = \
                        self._sequencers.time_sync_sequencer.data_by_component_label[detector_label]
                    detector_live_data.temporal_offset = TemporalOffset(
                        network_latency_milliseconds=component_data.network_latency_milliseconds,
                        clock_offset_milliseconds=component_data.clock_offset_milliseconds)
                for mixer_label, mixer_live_data in self._mixer_live_data.items():
                    component_data: TimeSyncSequencer.ComponentData = \
                        self._sequencers.time_sync_sequencer.data_by_component_label[mixer_label]
                    mixer_live_data.temporal_offset = TemporalOffset(
                        network_latency_milliseconds=component_data.network_latency_milliseconds,
                        clock_offset_milliseconds=component_data.clock_offset_milliseconds)
                self._sequencers.detector_startup_sequencer = DetectorStartupSequencer(**self._sequencer_init_args())
                self._sequencers.detector_startup_sequencer.begin(
                    detector_data=[DetectorStartupSequencer.InputDetectorData(
                        detector_label=detector.label,
                        camera_parameters=detector.camera_parameters,
                        annotator_parameters=detector.annotator_parameters)
                    for detector in self._configuration.detectors])
            if (
                self._sequencers.detector_startup_sequencer is not None and
                self._sequencers.detector_startup_sequencer.is_finished() and
                self._sequencers.detector_frame_get_sequencer is None and
                self._sequencers.mixer_startup_sequencer is None
            ):
                for detector_label, detector_live_data in self._detector_live_data.items():
                    detector_data: DetectorStartupSequencer.DetectorData = \
                        self._sequencers.detector_startup_sequencer.data_by_detector_label[detector_label]
                    detector_live_data.camera_resolution = detector_data.camera_resolution
                    detector_live_data.intrinsic_calibration = detector_data.intrinsic_calibration
                self._sequencers.detector_frame_get_sequencer = DetectorFrameGetSequencer(**self._sequencer_init_args())
                self._sequencers.detector_frame_get_sequencer.begin(
                    detector_labels=list(self._detector_live_data.keys()),
                    on_frame_callback=self._on_detector_frame_received)
                if len(self._configuration.mixers) <= 0:
                    self._state = MCTController.State.RUNNING
                else:
                    detector_labels_unmatched: set[str] = set()  # Non-empty indicates clear configuration error
                    detector_labels_without_intrinsic_calibration: set[str] = set()
                    detector_labels_without_extrinsic_calibration: set[str] = set()
                    intrinsics_by_detector_label: dict[str, IntrinsicParameters] = dict()
                    for mixer in self._configuration.mixers:
                        for detector in mixer.detectors:
                            detector_label: str = detector.detector_label
                            if detector_label in self._detector_live_data:
                                if detector_label not in intrinsics_by_detector_label:
                                    detector_live_data: DetectorLiveData = self._detector_live_data[detector_label]
                                    if detector_live_data.intrinsic_calibration is not None:
                                        # Easy case, expected most of the time
                                        intrinsics_by_detector_label[detector_label] = \
                                            detector_live_data.intrinsic_calibration.calibrated_values
                                    else:
                                        detector_labels_without_intrinsic_calibration.add(detector_label)
                                        intrinsics_by_detector_label[detector_label] = \
                                            IntrinsicParameters.generate_zero_parameters(
                                                resolution_x_px=detector_live_data.camera_resolution.x_px,
                                                resolution_y_px=detector_live_data.camera_resolution.y_px)
                                if detector.pose_mode == DetectorPoseMode.STATIC_EXTERNAL:
                                    if detector.detector_to_reference is None:
                                        detector_labels_without_extrinsic_calibration.add(detector_label)
                            else:
                                detector_labels_unmatched.add(detector_label)
                    if len(detector_labels_unmatched) > 0:
                        message: str = \
                            "Mixers refer to non-existing detectors. " + \
                            "Please check the configuration for these detector labels: " + \
                            ", ".join(detector_labels_unmatched)
                        self._status_message_source.enqueue_status_message(
                            severity=SeverityLabel.ERROR,  # FOR SURE an error
                            message=message)
                    if len(detector_labels_without_intrinsic_calibration) > 0:
                        message: str = \
                            "Intrinsic parameters were missing for the following detectors, " + \
                            "and default \"zero\" parameters were used. " + \
                            "This may be normal if an intrinsic calibration has not yet been done. " + \
                            "Detector labels. " + ", ".join(detector_labels_without_intrinsic_calibration)
                        self._status_message_source.enqueue_status_message(
                            severity=SeverityLabel.WARNING,
                            message=message)
                    if len(detector_labels_without_extrinsic_calibration) > 0:
                        message: str = \
                            "Extrinsic parameters were missing in at least one place for some detectors, " + \
                            "and default \"identity\" matrices were used. " + \
                            "This may be normal if an extrinsic calibration has not yet been done. " + \
                            "Detector labels: " + ", ".join(detector_labels_without_extrinsic_calibration)
                        self._status_message_source.enqueue_status_message(
                            severity=SeverityLabel.WARNING,
                            message=message)
                    self._sequencers.mixer_startup_sequencer = MixerStartupSequencer(**self._sequencer_init_args())
                    self._sequencers.mixer_startup_sequencer.begin(
                        mixer_data=[MixerStartupSequencer.InputMixerData(
                            mixer_label=mixer.label,
                            detectors=[MixerStartupSequencer.InputMixerData.Detector(
                                detector_label=detector.detector_label,
                                intrinsic_parameters=intrinsics_by_detector_label[detector.detector_label],
                                pose_mode=detector.pose_mode,
                                extrinsic_matrix=detector.detector_to_reference)
                                for detector in mixer.detectors],
                            targets=mixer.targets,
                            solver_parameters=mixer.solver_parameters)
                            for mixer in self._configuration.mixers])
            if (
                self._sequencers.mixer_startup_sequencer is not None and
                self._sequencers.mixer_startup_sequencer.is_finished()
            ):
                self._sequencers.mixer_frame_get_sequencer = MixerFrameGetSequencer(**self._sequencer_init_args())
                self._sequencers.mixer_frame_get_sequencer.begin(
                    mixer_labels=list(self._mixer_live_data.keys()),
                    on_frame_callback=self._on_mixer_frame_received)
                self._state = MCTController.State.RUNNING

        if self._state == MCTController.State.STOPPING:
            if self._sequencers.detector_shutdown_sequencer is None:
                self._sequencers.detector_shutdown_sequencer = DetectorShutdownSequencer(**self._sequencer_init_args())
                self._sequencers.detector_shutdown_sequencer.begin(
                    component_labels=list(self._detector_live_data.keys()))
            if len(self._configuration.mixers) > 0 and self._sequencers.mixer_shutdown_sequencer is None:
                self._sequencers.mixer_shutdown_sequencer = MixerShutdownSequencer(**self._sequencer_init_args())
                self._sequencers.mixer_shutdown_sequencer.begin(
                    component_labels=list(self._mixer_live_data.keys()))
            if (
                self._sequencers.detector_shutdown_sequencer is not None and
                self._sequencers.detector_shutdown_sequencer.is_finished() and
                len(self._configuration.mixers) <= 0 or (
                    self._sequencers.mixer_shutdown_sequencer is not None and
                    self._sequencers.mixer_shutdown_sequencer.is_finished())
            ):
                self._connection_router.shut_down()
                self._state = MCTController.State.DISCONNECTING

        response_series_lists: list[list[MCTResponseSeries]] = self._connection_router.dequeue_response_series_lists()
        for response_series_list in response_series_lists:
            for response_series in response_series_list:
                try:
                    self._callback_router.handle_callback(response_series)
                except Exception as e:
                    self._status_message_source.enqueue_status_message(
                        severity=SeverityLabel.ERROR,
                        message=f"Exception occurred on handling response from {response_series.responder}: {e}")

    def _sequencer_init_args(self):
        return {
            "status_message_source": self._status_message_source,
            "connection_router": self._connection_router,
            "callback_router": self._callback_router}

    def _on_detector_frame_received(
        self,
        detector_label: str,
        detector_data: DetectorFrameGetSequencer.OutputDetectorData
    ):
        self._detector_live_data[detector_label].frame = detector_data.frame
        if self._sequencers.mixer_frame_get_sequencer is not None:
            self._sequencers.mixer_frame_get_sequencer.set_latest_detector_frame(
                detector_label=detector_label,
                frame=DetectorFrame(
                    annotations=detector_data.frame.annotations,
                    timestamp_utc_iso8601=detector_data.frame.timestamp_utc_iso8601,
                    image_resolution=detector_data.frame.image_resolution,
                    image_base64=None))  # Explicitly exclude image; even if it is present, the mixer doesn't need it

    def _on_mixer_frame_received(
        self,
        mixer_label: str,
        mixer_data: MixerFrameGetSequencer.OutputMixerData
    ):
        self._mixer_live_data[mixer_label].frame = mixer_data.frame

    # =================================================================================================================
    #                                          DATA ACCESS/MANIPULATION
    # =================================================================================================================

    def get_configuration(self) -> MCTConfiguration:
        return self._configuration.model_copy(deep=True)

    def get_controller_state(self) -> State:
        return self._state

    def get_connection_reports(self) -> list[Connection.Report]:
        return self._connection_router.get_connection_reports()

    def get_live_detector_data(
        self,
        detector_label: str
    ) -> DetectorLiveData | None:
        """
        returns None if the detector does not exist, or if it has not yet gotten any frames.
        """
        return self._detector_live_data.get(detector_label, None)

    def get_live_mixer_data(
        self,
        mixer_label: str
    ) -> MixerLiveData | None:
        """
        returns None if the mixer does not exist, or if it has not yet gotten frames.
        """
        return self._mixer_live_data.get(mixer_label, None)

    def get_remote_labels(self) -> list[str]:
        return self.get_remote_labels_detectors() + self.get_remote_labels_mixer()

    def get_remote_labels_detectors(self) -> list[str]:
        return list(self._detector_live_data.keys())

    def get_remote_labels_mixer(self) -> list[str]:
        return list(self._mixer_live_data.keys())

    def send_custom_request(
        self,
        connection_label: str,
        requests: list[MCTRequest],
        callback: CallbackRouter.CallbackFunction = None,
        passthrough_arguments: dict[str, ...] | None = None
    ) -> uuid.UUID:
        request_series = MCTRequestSeries(series=requests)
        request_id: uuid.UUID = uuid.UUID(request_series.request_id)
        self._connection_router.enqueue_request_series(
            label=connection_label,
            request_series=request_series)
        if callback is not None:
            if passthrough_arguments is None:
                passthrough_arguments = dict()
            self._callback_router.add_callback(
                request_id=request_id,
                callback=callback,
                passthrough_arguments=passthrough_arguments)
        return request_id

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
