from .routing import \
    CallbackRouter, \
    ConnectionRouter
from src.common import \
    Annotator, \
    Camera, \
    DequeueStatusMessagesRequest, \
    DequeueStatusMessagesResponse, \
    DetectorFrame, \
    EmptyResponse, \
    ErrorResponse, \
    ImageResolution, \
    IntrinsicParameters, \
    ExtrinsicCalibration, \
    KeyValueMetaAny, \
    KeyValueSimpleAny, \
    MCTRequest, \
    MCTRequestSeries, \
    MCTResponse, \
    MCTResponseSeries, \
    MixerFrame, \
    SeverityLabel, \
    StatusMessageSource, \
    TimestampGetRequest, \
    TimestampGetResponse, \
    TimeSyncStartRequest, \
    TimeSyncStopRequest, IntrinsicCalibration, ImageFormat
from src.detector import \
    AnnotatorParametersGetRequest, \
    AnnotatorParametersGetResponse, \
    AnnotatorParametersSetRequest, \
    CameraParametersGetRequest, \
    CameraParametersGetResponse, \
    CameraParametersSetRequest, \
    CameraParametersSetResponse, \
    DetectorFrameGetRequest, \
    DetectorFrameGetResponse, \
    DetectorQueryRequest, \
    DetectorQueryResponse, \
    DetectorStartRequest, \
    DetectorStopRequest, \
    IntrinsicCalibrationResultGetActiveRequest, \
    IntrinsicCalibrationResultGetActiveResponse
from src.mixer import \
    Mixer, \
    MixerFrameGetRequest, \
    MixerFrameGetResponse, \
    MixerIntrinsicUpdateRequest, \
    MixerQueryRequest, \
    MixerQueryResponse, \
    MixerStartRequest, \
    MixerStopRequest, \
    PoseSolverDetectorFrameAddRequest, \
    PoseSolverExtrinsicSetRequest, \
    PoseSolverExtrinsicClearRequest
import abc
import datetime
from enum import StrEnum
import logging
import statistics
from typing import Callable, Final
import uuid


logger = logging.getLogger(__name__)


_TIME_SYNC_DEFAULT_SAMPLE_COUNT: Final[int] = 5


class AbstractSequencer(abc.ABC):

    _status_message_source: StatusMessageSource
    _connection_router: ConnectionRouter
    _callback_router: CallbackRouter

    _pending_request_ids: list[uuid.UUID]

    def __init__(
        self,
        status_message_source: StatusMessageSource,
        connection_router: ConnectionRouter,
        callback_router: CallbackRouter
    ):
        self._status_message_source = status_message_source
        self._connection_router = connection_router
        self._callback_router = callback_router
        self._pending_request_ids = list()

    @abc.abstractmethod
    def is_finished(self) -> bool: pass

    def reset(self) -> None:
        for request_id in self._pending_request_ids:
            self._callback_router.remove_callback(request_id)

    def _report_response_series_and_errors(
        self,
        response_series: MCTResponseSeries,
        expected_types: list[type[MCTResponse]]
    ) -> int:
        """
        Return the number of errors found
        """
        errors_found: int = 0
        label: str = response_series.responder
        if response_series.request_id not in self._pending_request_ids:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Received unexpected response with ID {response_series.request_id} from {label}.")
            errors_found += 1
        self._pending_request_ids.remove(response_series.request_id)
        if len(response_series.series) != len(expected_types):
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Unexpected (incorrect) number of responses from {label}. Expected {len(expected_types)}.")
            errors_found += 1
            for response_index, response in response_series.series:
                if isinstance(response, ErrorResponse):
                    self._status_message_source.enqueue_status_message(
                        severity=SeverityLabel.ERROR,
                        message=f"Error at index {response_index}: {response.message}",
                        source_label=label)
                errors_found += 1
            return errors_found
        for response_index, response in response_series.series:
            expected_type: type = expected_types[response_index]
            expected_type_name: str = expected_type.__name__
            if isinstance(response, ErrorResponse):
                self._status_message_source.enqueue_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Error in place of {expected_type_name}: {response.message}",
                    source_label=label)
                errors_found += 1
            elif not isinstance(response, expected_type):
                self._status_message_source.enqueue_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Unexpected response in place of {expected_type_name}: {type(response).__name__}")
                errors_found += 1
        return errors_found

    def _send_request_series(
        self,
        component_label: str,
        requests: list[MCTRequest],
        callback: CallbackRouter.CallbackFunction
    ) -> uuid.UUID:
        request_series: MCTRequestSeries = MCTRequestSeries(series=requests)
        request_id: uuid.UUID = request_series.request_id
        self._connection_router.enqueue_request_series(label=component_label, request_series=request_series)
        self._callback_router.add_callback(request_id=request_id, callback=callback)
        self._pending_request_ids.append(request_id)
        return request_id


class TimeSyncSequencer(AbstractSequencer):

    class State(StrEnum):
        INITIAL = "Initial"
        STARTING = "Starting"
        SYNCING = "Syncing"
        STOPPING = "Stopping"
        FINISHED = "Finished"

    class ComponentData:
        class Sample:
            local_sent_datetime: datetime.datetime
            remote_received_datetime: datetime.datetime
            local_received_datetime: datetime.datetime
            def __init__(
                self,
                local_sent_timestamp_iso8601,
                remote_received_timestamp_iso8601,
                local_received_timestamp_iso8601
            ):
                self.local_sent_datetime = datetime.datetime.fromisoformat(local_sent_timestamp_iso8601)
                self.remote_received_datetime = datetime.datetime.fromisoformat(remote_received_timestamp_iso8601)
                self.local_received_datetime = datetime.datetime.fromisoformat(local_received_timestamp_iso8601)

        samples: list[Sample]
        _network_round_delay_milliseconds: float | None
        _clock_offset_to_remote_milliseconds: float | None

        def __init__(self):
            self.samples = list()
            self._network_round_delay_milliseconds = None
            self._clock_offset_to_remote_milliseconds = None

        def _calculate(self):
            round_delays_ms: list[float] = list()  # Caused by network
            offsets_ms: list[float] = list()       # Caused by difference in computer clock time
            for sample in self.samples:
                round_delay_delta: datetime.timedelta = sample.local_sent_datetime - sample.local_received_datetime
                round_delay_ms: float = round_delay_delta / datetime.timedelta(milliseconds=1)
                round_delays_ms.append(round_delay_ms)
                one_way_delay_ms: int = round(round_delay_ms / 2.0)
                one_way_delay_delta: datetime.timedelta = datetime.timedelta(milliseconds=one_way_delay_ms)
                local_datetime: datetime.datetime = sample.local_sent_datetime + one_way_delay_delta
                remote_datetime: datetime.datetime = sample.remote_received_datetime
                offset_delta: datetime.timedelta = remote_datetime - local_datetime
                offset_ms: float = offset_delta / datetime.timedelta(milliseconds=1)
                offsets_ms.append(offset_ms)
            self._network_round_delay_milliseconds = statistics.median(round_delays_ms)
            self._clock_offset_to_remote_milliseconds = statistics.median(offsets_ms)

        @property
        def network_latency_milliseconds(self) -> float:
            """Delay from round trip (caveat that it involves some processing)."""
            if self._network_round_delay_milliseconds is None:
                self._calculate()
            return self._network_round_delay_milliseconds

        @property
        def clock_offset_milliseconds(self) -> float:
            """The amount to add to the local time to get the remote time"""
            if self._clock_offset_to_remote_milliseconds is None:
                self._calculate()
            return self._clock_offset_to_remote_milliseconds

    data_by_component_label: dict[str, ComponentData]
    _sample_count: int
    _state: State

    def __init__(
        self,
        status_message_source: StatusMessageSource,
        connection_router: ConnectionRouter,
        callback_router: CallbackRouter
    ):
        super().__init__(
            status_message_source=status_message_source,
            connection_router=connection_router,
            callback_router=callback_router)
        self._state = TimeSyncSequencer.State.INITIAL
        self._sample_count = _TIME_SYNC_DEFAULT_SAMPLE_COUNT
        self.data_by_component_label = dict()

    def begin(
        self,
        sample_count: int,
        component_labels: list[str]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="TimeSyncSequencer.begin()")
        if self._state != TimeSyncSequencer.State.INITIAL:
            message: str = \
                f"TimeSyncSequencer.begin() called when in an incorrect state {self._state}. " + \
                f"Try calling reset()."
            self._status_message_source.enqueue_status_message(severity=SeverityLabel.ERROR, message=message)
            return
        self._sample_count = sample_count
        if len(component_labels) == 0:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"No component labels were provided to TimeSyncSequencer")
            self._state = TimeSyncSequencer.State.FINISHED
            return
        self.data_by_component_label.clear()
        for component_label in component_labels:
            self.data_by_component_label[component_label] = TimeSyncSequencer.ComponentData()
        self._request_1_start()

    def reset(self):
        super().reset()
        self._state = TimeSyncSequencer.State.INITIAL
        self.data_by_component_label.clear()

    def is_finished(self) -> bool:
        return self._state == TimeSyncSequencer.State.FINISHED

    def _request_1_start(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="TimeSyncSequencer._request_1_start()")
        for component_label in self.data_by_component_label.keys():
            self._send_request_series(
                component_label=component_label,
                requests=[TimeSyncStartRequest()],
                callback=self._request_1_start_responded)
        self._state = TimeSyncSequencer.State.STARTING

    def _request_1_start_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="TimeSyncSequencer._request_1_start_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[EmptyResponse]
        ):
            return
        if len(self._pending_request_ids) > 0:
            return
        self._request_2_timestamp()

    def _request_2_timestamp(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="TimeSyncSequencer._request_2_timestamp()")
        for component_label in self.data_by_component_label.keys():
            now_utc_iso8601: str = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
            self._send_request_series(
                component_label=component_label,
                requests=[TimestampGetRequest(requester_timestamp_utc_iso8601=now_utc_iso8601)],
                callback=self._request_2_timestamp_responded)
        self._state = TimeSyncSequencer.State.SYNCING

    def _request_2_timestamp_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        now_utc_iso8601 = uuid.uuid4()
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="TimeSyncSequencer._request_2_timestamp_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[TimestampGetResponse]
        ):
            return
        response: TimestampGetResponse = response_series.series[0]
        component_label: str = response_series.responder
        self.data_by_component_label[component_label].samples.append(TimeSyncSequencer.ComponentData.Sample(
            local_sent_timestamp_iso8601=response.requester_timestamp_utc_iso8601,
            remote_received_timestamp_iso8601=response.responder_timestamp_utc_iso8601,
            local_received_timestamp_iso8601=now_utc_iso8601))
        if len(self._pending_request_ids) > 0:
            return
        samples_collected: int = len(self.data_by_component_label[component_label].samples)
        if samples_collected < self._sample_count:
            self._request_2_timestamp()
        else:
            self._request_3_stop()

    def _request_3_stop(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="TimeSyncSequencer._request_3_stop()")
        for component_label in self.data_by_component_label.keys():
            self._send_request_series(
                component_label=component_label,
                requests=[TimeSyncStopRequest()],
                callback=self._request_3_stop_responded)
        self._state = TimeSyncSequencer.State.STOPPING

    def _request_3_stop_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="TimeSyncSequencer._request_3_stop_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[EmptyResponse]
        ):
            return
        if len(self._pending_request_ids) > 0:
            return
        self._state = TimeSyncSequencer.State.FINISHED


# noinspection DuplicatedCode
class DetectorStartupSequencer(AbstractSequencer):

    class DetectorData:
        camera_parameters: list[KeyValueSimpleAny] | list[KeyValueMetaAny]
        annotator_parameters: list[KeyValueSimpleAny] | list[KeyValueMetaAny]
        camera_resolution: ImageResolution | None
        intrinsic_calibration: IntrinsicCalibration | None
        done_query: bool
        done_reset: bool
        done_start: bool
        done_set_parameters: bool
        done_get_parameters: bool

        def __init__(
            self,
            camera_parameters: list[KeyValueSimpleAny],
            annotator_parameters: list[KeyValueSimpleAny]
        ):
            self.camera_parameters = camera_parameters
            self.annotator_parameters = annotator_parameters
            self.camera_resolution = None
            self.intrinsic_calibration = None
            self.done_query = False
            self.done_reset = False
            self.done_start = False
            self.done_set_parameters = False
            self.done_get_parameters = False

    class State(StrEnum):
        INITIAL = "Initial"
        QUERYING = "Querying"
        RESET = "Reset"
        STARTING = "Starting"
        SET_PARAMETERS = "SetParameters"
        GET_PARAMETERS = "GetParameters"
        FINISHED = "Finished"

    _state: State
    data_by_detector_label: dict[str, DetectorData]

    def __init__(
        self,
        status_message_source: StatusMessageSource,
        connection_router: ConnectionRouter,
        callback_router: CallbackRouter
    ):
        super().__init__(
            status_message_source=status_message_source,
            connection_router=connection_router,
            callback_router=callback_router)
        self._state = DetectorStartupSequencer.State.INITIAL
        self.data_by_detector_label = dict()

    def begin(
        self,
        detector_labels: list[str],
        camera_parameters: list[list[KeyValueSimpleAny]],
        annotator_parameters: list[list[KeyValueSimpleAny]]
    ) -> None:
        if self._state != DetectorStartupSequencer.State.INITIAL:
            message: str = \
                f"DetectorStartupSequencer.begin() called when in an incorrect state {self._state}. " + \
                f"Try calling reset()."
            self._status_message_source.enqueue_status_message(severity=SeverityLabel.ERROR, message=message)
            return
        detector_count: int = len(detector_labels)
        if detector_count == 0:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"No inputs were provided to DetectorStartupSequencer")
            return
        if detector_count != len(camera_parameters) or detector_count != len(annotator_parameters):
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Detector label and parameter lists are of different lengths.")
            return
        self.data_by_detector_label.clear()
        for detector_index in range(detector_count):
            detector_label: str = detector_labels[detector_index]
            self.data_by_detector_label[detector_label] = DetectorStartupSequencer.DetectorData(
                camera_parameters=camera_parameters[detector_index],
                annotator_parameters=annotator_parameters[detector_index])
        self._request_1_query()

    def reset(self) -> None:
        super().reset()
        self._state = DetectorStartupSequencer.State.INITIAL

    def is_finished(self) -> bool:
        return self._state == DetectorStartupSequencer.State.FINISHED

    def _request_1_query(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_1_query()")
        for detector_label in self.data_by_detector_label.keys():
            self._send_request_series(
                component_label=detector_label,
                requests=[DetectorQueryRequest()],
                callback=self._request_1_query_responded)
        self._state = DetectorStartupSequencer.State.QUERYING

    def _request_1_query_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_1_query_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[DetectorQueryResponse]
        ):
            return
        detector_label: str = response_series.responder
        response: DetectorQueryResponse = response_series.series[0]
        self.data_by_detector_label[detector_label].done_query = True
        if response.camera_status == Camera.Status.RUNNING and response.annotator_status == Annotator.Status.RUNNING:
            self.data_by_detector_label[detector_label].done_reset = True
            self.data_by_detector_label[detector_label].done_start = True  # These steps are not needed
        elif response.camera_status != Camera.Status.STOPPED and response.annotator_status == Annotator.Status.STOPPED:
            self.data_by_detector_label[detector_label].done_reset = True  # Not needed
        if len(self._pending_request_ids) > 0:
            return
        detectors_needing_reset_count: int = sum([
            (1 - self.data_by_detector_label[detector_label].done_reset)
            for detector_label in self.data_by_detector_label.keys()])
        if detectors_needing_reset_count > 0:
            self._request_2_reset()
            return
        detectors_needing_start_count: int = sum([
            (1 - self.data_by_detector_label[detector_label].done_start)
            for detector_label in self.data_by_detector_label.keys()])
        if detectors_needing_start_count > 0:
            self._request_3_start()
            return
        self._request_4_set_parameters()

    def _request_2_reset(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_2_reset()")
        for detector_label in self.data_by_detector_label.keys():
            if not self.data_by_detector_label[detector_label].done_reset:
                self._send_request_series(
                    component_label=detector_label,
                    requests=[DetectorStopRequest()],
                    callback=self._request_2_reset_responded)
        self._state = DetectorStartupSequencer.State.RESET

    def _request_2_reset_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_2_reset_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[EmptyResponse]
        ):
            return
        detector_label: str = response_series.responder
        self.data_by_detector_label[detector_label].done_reset = True
        if len(self._pending_request_ids) > 0:
            return
        detectors_needing_start_count: int = sum([
            (1 - self.data_by_detector_label[detector_label].done_start)
            for detector_label in self.data_by_detector_label.keys()])
        if detectors_needing_start_count > 0:
            self._request_3_start()
            return
        self._request_4_set_parameters()

    def _request_3_start(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_3_start()")
        for detector_label in self.data_by_detector_label.keys():
            if not self.data_by_detector_label[detector_label].done_start:
                self._send_request_series(
                    component_label=detector_label,
                    requests=[DetectorStartRequest()],
                    callback=self._request_3_start_responded)
        self._state = DetectorStartupSequencer.State.STARTING

    def _request_3_start_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_3_start_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[EmptyResponse]
        ):
            return
        detector_label: str = response_series.responder
        self.data_by_detector_label[detector_label].done_start = True
        if len(self._pending_request_ids) > 0:
            return
        self._request_4_set_parameters()

    def _request_4_set_parameters(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_4_set_parameters()")
        for detector_label in self.data_by_detector_label.keys():
            self._send_request_series(
                component_label=detector_label,
                requests=[
                    CameraParametersSetRequest(
                        parameters=self.data_by_detector_label[detector_label].camera_parameters),
                    AnnotatorParametersSetRequest(
                        parameters=self.data_by_detector_label[detector_label].annotator_parameters)],
                callback=self._request_4_set_parameters_responded)
        self._state = DetectorStartupSequencer.State.SET_PARAMETERS

    def _request_4_set_parameters_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_4_set_parameters_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[CameraParametersSetResponse, EmptyResponse]
        ):
            return
        detector_label: str = response_series.responder
        self.data_by_detector_label[detector_label].done_set_parameters = True
        if len(self._pending_request_ids) > 0:
            return
        self._request_5_get_parameters()

    def _request_5_get_parameters(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_5_get_parameters()")
        for detector_label in self.data_by_detector_label.keys():
            self._send_request_series(
                component_label=detector_label,
                requests=[
                    CameraParametersGetRequest(),
                    AnnotatorParametersGetRequest(),
                    IntrinsicCalibrationResultGetActiveRequest()],
                callback=self._request_5_get_parameters_responded)
        self._state = DetectorStartupSequencer.State.GET_PARAMETERS

    def _request_5_get_parameters_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_5_get_parameters_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[
                CameraParametersGetResponse,
                AnnotatorParametersGetResponse,
                IntrinsicCalibrationResultGetActiveResponse]
        ):
            return
        detector_label: str = response_series.responder
        camera_response: CameraParametersGetResponse = response_series.series[0]
        self.data_by_detector_label[detector_label].camera_parameters = camera_response.parameters
        self.data_by_detector_label[detector_label].camera_resolution = camera_response.resolution
        annotator_response: AnnotatorParametersGetResponse = response_series.series[1]
        self.data_by_detector_label[detector_label].annotator_parameters = annotator_response.parameters
        calibration_response: IntrinsicCalibrationResultGetActiveResponse = response_series.series[2]
        self.data_by_detector_label[detector_label].intrinsic_calibration = calibration_response.intrinsic_calibration
        self.data_by_detector_label[detector_label].done_get_parameters = True
        if len(self._pending_request_ids) > 0:
            return
        self._state = DetectorStartupSequencer.State.FINISHED


# noinspection DuplicatedCode
class MixerStartupSequencer(AbstractSequencer):

    class MixerData:
        pose_solver_parameters: list[KeyValueSimpleAny] | list[KeyValueMetaAny]
        extrinsic_calibration: ExtrinsicCalibration | None
        done_query: bool
        done_reset: bool
        done_start: bool
        done_clear: bool
        done_set_intrinsics: bool
        done_get_extrinsics: bool

        def __init__(
            self,
            pose_solver_parameters: list[KeyValueSimpleAny]
        ):
            self.pose_solver_parameters = pose_solver_parameters
            self.extrinsic_calibration = None
            self.done_query = False
            self.done_reset = False
            self.done_start = False
            self.done_clear = False
            self.done_set_intrinsics = False
            self.done_get_extrinsics = False

    class State(StrEnum):
        INITIAL = "Initial"
        QUERYING = "Querying"
        RESET = "Reset"
        STARTING = "Starting"
        CLEAR_EXTRINSICS = "ClearExtrinsics",
        SET_INTRINSICS = "SetIntrinsics"
        SET_EXTRINSICS = "SetExtrinsics"
        FINISHED = "Finished"

    _state: State
    _intrinsics_by_detector_label: dict[str, IntrinsicParameters]
    _extrinsics_by_detector_label: dict[str, IntrinsicParameters] | None
    data_by_mixer_label: dict[str, MixerData]

    def __init__(
        self,
        status_message_source: StatusMessageSource,
        connection_router: ConnectionRouter,
        callback_router: CallbackRouter
    ):
        super().__init__(
            status_message_source=status_message_source,
            connection_router=connection_router,
            callback_router=callback_router)
        self._state = MixerStartupSequencer.State.INITIAL
        self._intrinsics_by_detector_label = dict()
        self._extrinsics_by_detector_label = None
        self.data_by_mixer_label = dict()

    def begin(
        self,
        mixer_labels: list[str],
        pose_solver_parameters: list[list[KeyValueSimpleAny]],
        intrinsics_by_detector_label: dict[str, IntrinsicCalibration],
        extrinsics_by_detector_label: dict[str, ExtrinsicCalibration] | None = None
    ) -> None:
        if self._state != MixerStartupSequencer.State.INITIAL:
            message: str = \
                f"MixerStartupSequencer.begin() called when in an incorrect state {self._state}. " + \
                f"Try calling reset()."
            self._status_message_source.enqueue_status_message(severity=SeverityLabel.ERROR, message=message)
            return
        mixer_count: int = len(mixer_labels)
        if mixer_count == 0:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"No inputs were provided to MixerStartupSequencer")
            return
        if mixer_count != len(pose_solver_parameters):
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Mixer label and intrinsic lists are of different lengths.")
            return
        if extrinsics_by_detector_label is not None:
            if mixer_count != len(extrinsics_by_detector_label):
                self._status_message_source.enqueue_status_message(
                    severity=SeverityLabel.ERROR,
                    message=f"Mixer label and extrinsic lists are of different lengths.")
                return
        self._intrinsics_by_detector_label = intrinsics_by_detector_label
        self._extrinsics_by_detector_label = extrinsics_by_detector_label
        self.data_by_mixer_label.clear()
        for mixer_index in range(mixer_count):
            mixer_label: str = mixer_labels[mixer_index]
            self.data_by_mixer_label[mixer_label] = MixerStartupSequencer.MixerData(
                pose_solver_parameters=pose_solver_parameters[mixer_label])
        self._request_1_query()

    def reset(self) -> None:
        super().reset()
        self._state = MixerStartupSequencer.State.INITIAL

    def is_finished(self) -> bool:
        return self._state == MixerStartupSequencer.State.FINISHED

    def _request_1_query(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="MixerStartupSequencer._request_1_query()")
        for mixer_label in self.data_by_mixer_label.keys():
            self._send_request_series(
                component_label=mixer_label,
                requests=[MixerQueryRequest()],
                callback=self._request_1_query_responded)
        self._state = MixerStartupSequencer.State.QUERYING

    def _request_1_query_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="MixerStartupSequencer._request_1_query_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[MixerQueryResponse]
        ):
            return
        mixer_label: str = response_series.responder
        response: MixerQueryResponse = response_series.series[0]
        self.data_by_mixer_label[mixer_label].done_query = True
        if response.mixer_status == Mixer.Status.RUNNING:
            self.data_by_mixer_label[mixer_label].done_reset = True
            self.data_by_mixer_label[mixer_label].done_start = True  # These steps are not needed
        elif response.mixer_status != Mixer.Status.STOPPED:
            self.data_by_mixer_label[mixer_label].done_reset = True  # Not needed
        if len(self._pending_request_ids) > 0:
            return
        mixers_needing_reset_count: int = sum([
            (1 - self.data_by_mixer_label[mixer_label].done_reset)
            for mixer_label in self.data_by_mixer_label.keys()])
        if mixers_needing_reset_count > 0:
            self._request_2_reset()
            return
        mixers_needing_start_count: int = sum([
            (1 - self.data_by_mixer_label[mixer_label].done_start)
            for mixer_label in self.data_by_mixer_label.keys()])
        if mixers_needing_start_count > 0:
            self._request_3_start()
            return
        self._request_5_set_intrinsics()

    def _request_2_reset(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="MixerStartupSequencer._request_2_reset()")
        for mixer_label in self.data_by_mixer_label.keys():
            if not self.data_by_mixer_label[mixer_label].done_reset:
                self._send_request_series(
                    component_label=mixer_label,
                    requests=[MixerStopRequest()],
                    callback=self._request_2_reset_responded)
        self._state = MixerStartupSequencer.State.RESET

    def _request_2_reset_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="MixerStartupSequencer._request_2_reset_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[EmptyResponse]
        ):
            return
        mixer_label: str = response_series.responder
        self.data_by_mixer_label[mixer_label].done_reset = True
        if len(self._pending_request_ids) > 0:
            return
        self._request_3_start()

    def _request_3_start(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="MixerStartupSequencer._request_3_start()")
        for mixer_label in self.data_by_mixer_label.keys():
            if not self.data_by_mixer_label[mixer_label].done_start:
                self._send_request_series(
                    component_label=mixer_label,
                    requests=[MixerStartRequest()],
                    callback=self._request_3_start_responded)
        self._state = MixerStartupSequencer.State.STARTING

    def _request_3_start_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="MixerStartupSequencer._request_3_start_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[EmptyResponse]
        ):
            return
        mixer_label: str = response_series.responder
        self.data_by_mixer_label[mixer_label].done_start = True
        if len(self._pending_request_ids) > 0:
            return
        self._request_4_clear_extrinsics()

    def _request_4_clear_extrinsics(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_4_clear_extrinsics()")
        for mixer_label in self.data_by_mixer_label.keys():
            self._send_request_series(
                component_label=mixer_label,
                requests=[PoseSolverExtrinsicClearRequest()],
                callback=self._request_4_clear_extrinsics_responded)
        self._state = MixerStartupSequencer.State.CLEAR_EXTRINSICS

    def _request_4_clear_extrinsics_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_4_clear_extrinsics_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[EmptyResponse]
        ):
            return
        mixer_label: str = response_series.responder
        self.data_by_mixer_label[mixer_label].done_clear = True
        if len(self._pending_request_ids) > 0:
            return
        self._request_5_set_intrinsics()

    def _request_5_set_intrinsics(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_5_set_intrinsics()")
        for mixer_label in self.data_by_mixer_label.keys():
            requests: list[MCTRequest] = [
                MixerIntrinsicUpdateRequest(
                    detector_label=detector_label,
                    intrinsic_parameters=self._intrinsics_by_detector_label[detector_label])
                for detector_label in self._intrinsics_by_detector_label.keys()]
            self._send_request_series(
                component_label=mixer_label,
                requests=requests,
                callback=self._request_5_set_intrinsics_responded)
        self._state = DetectorStartupSequencer.State.SET_PARAMETERS

    def _request_5_set_intrinsics_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorStartupSequencer._request_5_set_intrinsics_responded()")
        expected_types: list[type[MCTResponse]] = [EmptyResponse] * len(self._intrinsics_by_detector_label)
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=expected_types
        ):
            return
        mixer_label: str = response_series.responder
        self.data_by_mixer_label[mixer_label].done_set_intrinsics = True
        if len(self._pending_request_ids) > 0:
            return
        if self._extrinsics_by_detector_label is not None:
            self._request_6_set_extrinsics()
            return
        self._state = MixerStartupSequencer.State.FINISHED

    def _request_6_set_extrinsics(self):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="MixerStartupSequencer._request_6_set_extrinsics()")
        for mixer_label in self.data_by_mixer_label.keys():
            requests: list[MCTRequest] = [
                PoseSolverExtrinsicSetRequest(
                    detector_label=detector_label,
                    transform_to_reference=self._extrinsics_by_detector_label[detector_label].get_matrix())
                for detector_label in self._extrinsics_by_detector_label.keys()]
            self._send_request_series(
                component_label=mixer_label,
                requests=requests,
                callback=self._request_5_set_intrinsics_responded)
        self._state = MixerStartupSequencer.State.SET_EXTRINSICS

    def _request_6_set_extrinsics_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="MixerStartupSequencer._request_6_set_extrinsics_responded()")
        expected_types: list[type[MCTResponse]] = [EmptyResponse] * len(self._intrinsics_by_detector_label)
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=expected_types
        ):
            return
        mixer_label: str = response_series.responder
        self.data_by_mixer_label[mixer_label].done_set_intrinsics = True
        if len(self._pending_request_ids) > 0:
            return
        self._state = MixerStartupSequencer.State.FINISHED


# noinspection DuplicatedCode
class DetectorFrameGetSequencer(AbstractSequencer):

    class State(StrEnum):
        IDLE = "Idle"
        RUNNING = "Running"

    class DetectorData:
        frame: DetectorFrame | None

        def __init__(self):
            self.frame = None

    _state: State
    data_by_detector_label: dict[str, DetectorData]

    _include_detected: bool
    _include_rejected: bool
    _include_image: bool
    _requested_image_resolution: ImageResolution | None
    _requested_image_format: ImageFormat
    _on_frame_callback: Callable[[str, DetectorData], None] | None

    def __init__(
        self,
        status_message_source: StatusMessageSource,
        connection_router: ConnectionRouter,
        callback_router: CallbackRouter
    ):
        super().__init__(
            status_message_source=status_message_source,
            connection_router=connection_router,
            callback_router=callback_router)
        self._state = DetectorFrameGetSequencer.State.IDLE
        self._include_detected = False
        self._include_rejected = False
        self._include_image = False
        self._requested_image_resolution = None
        self._requested_image_format = ImageFormat.FORMAT_PNG
        self._on_frame_callback = None
        self.data_by_detector_label = dict()

    def begin(
        self,
        detector_labels: list[str],
        include_detected: bool,
        include_rejected: bool,
        include_image: bool,
        requested_image_format: ImageFormat = ImageFormat.FORMAT_JPG,
        requested_image_resolution: ImageResolution | None = None
    ) -> None:
        """
        :param detector_labels:
        :param include_detected:
        :param include_rejected:
        :param include_image:
        :param requested_image_format: Either ".png" or ".jpg"
        :param requested_image_resolution: If not None, Detectors will scale images before sending.
        """
        if self._state != DetectorFrameGetSequencer.State.IDLE:
            message: str = \
                f"DetectorFrameGetSequencer.begin() called when in an incorrect state {self._state}. " + \
                f"Try calling reset()."
            self._status_message_source.enqueue_status_message(severity=SeverityLabel.ERROR, message=message)
            return
        detector_count: int = len(detector_labels)
        if detector_count == 0:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"No inputs were provided to DetectorFrameGetSequencer")
            return
        if not self._include_detected and not self._include_rejected and not self._include_image:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"No outputs are being requested from Detector frames.")
            return
        self.data_by_detector_label.clear()
        self._include_detected = include_detected
        self._include_rejected = include_rejected
        self._include_image = include_image
        self._requested_image_format = requested_image_format
        self._requested_image_resolution = requested_image_resolution
        for detector_label in detector_labels:
            self.data_by_detector_label[detector_label] = DetectorFrameGetSequencer.DetectorData()
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.DEBUG,
                message=f"DetectorFrameGetSequencer - starting loop for detector {detector_label}")
            self._request_frame_get(detector_label=detector_label)
        self._state = DetectorFrameGetSequencer.State.RUNNING

    def reset(self) -> None:
        super().reset()
        self._state = DetectorFrameGetSequencer.State.IDLE

    def is_finished(self) -> bool:
        return False

    def _request_frame_get(self, detector_label: str):
        requests: list[MCTRequest] = list()
        requests.append(DetectorFrameGetRequest(
            include_detected=self._include_detected,
            include_rejected=self._include_rejected,
            include_image=self._include_image,
            image_format=self._requested_image_format,
            image_resolution=self._requested_image_resolution))
        self._send_request_series(
            component_label=detector_label,
            requests=requests,
            callback=self._request_frame_get_responded)

    def _request_frame_get_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="DetectorFrameGetSequencer._request_frame_get_responded()")
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=[DetectorFrameGetResponse]
        ):
            return
        detector_label: str = response_series.responder
        response: DetectorFrameGetResponse = response_series.series[0]
        self.data_by_detector_label[detector_label].frame = response.frame
        if self._on_frame_callback is not None:
            self._on_frame_callback(detector_label, self.data_by_detector_label[detector_label])
        self._request_frame_get(detector_label=detector_label)


# noinspection DuplicatedCode
class MixerFrameGetSequencer(AbstractSequencer):

    class State(StrEnum):
        IDLE = "Idle"
        RUNNING = "Running"

    class MixerData:
        detector_labels_needing_frame_send: list[str]
        detector_labels_send_count_by_request: dict[uuid.UUID, int]
        frame: MixerFrame | None

        def __init__(self):
            self.detector_labels_needing_frame_send = list()
            self.detector_labels_sent_count = dict()
            self.frame = None

    _state: State
    _frame_by_detector_label: dict[str, DetectorFrame]
    data_by_mixer_label: dict[str, MixerData]

    _on_frame_callback: Callable[[str, MixerData], None] | None

    def __init__(
        self,
        status_message_source: StatusMessageSource,
        connection_router: ConnectionRouter,
        callback_router: CallbackRouter
    ):
        super().__init__(
            status_message_source=status_message_source,
            connection_router=connection_router,
            callback_router=callback_router)
        self._state = MixerFrameGetSequencer.State.IDLE
        self._latest_frame_by_detector_label = dict()
        self.data_by_mixer_label = dict()
        self._on_frame_callback = None

    def begin(
        self,
        mixer_labels: list[str],
        on_frame_callback: Callable[[str, MixerFrame], None] | None = None
    ) -> None:
        if self._state != MixerFrameGetSequencer.State.IDLE:
            message: str = \
                f"MixerFrameGetSequencer.begin() called when in an incorrect state {self._state}. " + \
                f"Try calling reset()."
            self._status_message_source.enqueue_status_message(severity=SeverityLabel.ERROR, message=message)
            return
        mixer_count: int = len(mixer_labels)
        if mixer_count == 0:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"No inputs were provided to MixerFrameGetSequencer")
            return
        self._latest_frame_by_detector_label.clear()
        self.data_by_mixer_label.clear()
        self._on_frame_callback = on_frame_callback
        for mixer_label in mixer_labels:
            self.data_by_mixer_label[mixer_label] = MixerFrameGetSequencer.MixerData()
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.DEBUG,
                message=f"MixerFrameGetSequencer - starting loop for Mixer {mixer_label}")
            self._request_frame_get(mixer_label=mixer_label)
        self._state = MixerFrameGetSequencer.State.RUNNING

    def reset(self) -> None:
        super().reset()
        self._state = MixerFrameGetSequencer.State.IDLE

    def is_finished(self) -> bool:
        return False

    def set_latest_detector_frame(self, detector_label: str, frame: DetectorFrame) -> None:
        self._latest_frame_by_detector_label[detector_label] = frame
        for mixer_data in self.data_by_mixer_label.values():
            mixer_data.detector_labels_needing_frame_send.append(detector_label)

    def _request_frame_get(self, mixer_label: str) -> None:
        requests: list[MCTRequest] = list()
        mixer_data: MixerFrameGetSequencer.MixerData = self.data_by_mixer_label[mixer_label]
        for detector_label in mixer_data.detector_labels_needing_frame_send:
            requests.append(PoseSolverDetectorFrameAddRequest(
                detector_label=detector_label,
                detector_frame=self._frame_by_detector_label[detector_label]))
        detector_labels_send_count: int = len(requests)
        requests.append(MixerFrameGetRequest())
        request_id: uuid.UUID = self._send_request_series(
            component_label=mixer_label,
            requests=requests,
            callback=self._request_frame_get_responded)
        mixer_data.detector_labels_send_count_by_request[request_id] = detector_labels_send_count
        mixer_data.detector_labels_needing_frame_send.clear()

    def _request_frame_get_responded(
        self,
        response_series: MCTResponseSeries,
        _passthrough_parameters: dict[str, ...]
    ):
        self._status_message_source.enqueue_status_message(
            severity=SeverityLabel.DEBUG,
            message="MixerFrameGetSequencer._request_frame_get_responded()")
        mixer_label: str = response_series.responder
        mixer_data: MixerFrameGetSequencer.MixerData = self.data_by_mixer_label[mixer_label]
        request_id: uuid.UUID = response_series.request_id
        expected_types: list[type] = [EmptyResponse] * mixer_data.detector_labels_send_count_by_request[request_id]
        expected_types.append(MixerFrameGetResponse)
        if self._report_response_series_and_errors(
            response_series=response_series,
            expected_types=expected_types
        ):
            return
        mixer_data.detector_labels_send_count_by_request.pop(request_id)
        response: MixerFrameGetResponse = response_series.series[0]
        mixer_data.frame = response.frame
        if self._on_frame_callback is not None:
            self._on_frame_callback(mixer_label, mixer_data.frame)
        self._request_frame_get(mixer_label=mixer_label)
