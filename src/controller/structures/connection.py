from .mct_component_address import MCTComponentAddress
from .connection_report import ConnectionReport
from src.common import \
    DequeueStatusMessagesResponse, \
    EmptyResponse, \
    ErrorResponse, \
    MCTRequestSeries, \
    MCTResponse, \
    MCTResponseSeries, \
    TimestampGetResponse
from src.common.structures import \
    MCTParsable, \
    SeverityLabel, \
    StatusMessage
import abc
import datetime
from enum import StrEnum
import json
from typing import Final
import uuid
from websockets import ConnectionClosed
from websockets.sync.client import connect, ClientConnection


_ATTEMPT_COUNT_MAXIMUM: Final[int] = 3
_ATTEMPT_TIME_GAP_SECONDS: Final[float] = 5.0


class Connection(abc.ABC):
    """
    A connection represents the interface with a remote component
    """

    # Each connection shall be in one of the states listed below.

    class State(StrEnum):
        # This is the normal progression cycle ending back in "Inactive"
        INACTIVE: Final[str] = "Inactive"
        CONNECTING: Final[str] = "Connecting"
        INITIALIZING: Final[str] = "Initializing"
        RUNNING: Final[str] = "Running"
        RECONNECTING: Final[str] = "Reconnecting"  # Only if connection gets lost
        NORMAL_DEINITIALIZING: Final[str] = "Deinitializing"   # normal means not in a failure state
        NORMAL_DISCONNECTING: Final[str] = "Disconnecting"
        # States below indicate abnormal/failed states
        FAILURE: Final[str] = "Failure"
        FAILURE_DISCONNECTING: Final[str] = "Failure - Disconnecting"
        FAILURE_DEINITIALIZING: Final[str] = "Failure - Deinitializing"

    class ConnectionResult:
        success: bool
        error_message: str

        def __init__(
            self,
            success: bool,
            error_message: str = ""
        ):
            self.success = success
            self.error_message = error_message

    class DeinitializationResult(StrEnum):
        IN_PROGRESS: Final[str] = "In Progress"
        SUCCESS: Final[str] = "Success"
        FAILURE: Final[str] = "Failure"

    class InitializationResult(StrEnum):
        IN_PROGRESS: Final[str] = "In Progress"
        SUCCESS: Final[str] = "Success"
        FAILURE: Final[str] = "Failure"

    class SendRecvResult(StrEnum):
        NORMAL: Final[str] = "Normal"
        FAILURE: Final[str] = "Failure"

    class PopResponseSeriesResult:
        class Status(StrEnum):
            QUEUED: Final[str] = "Exists"
            IN_PROGRESS: Final[str] = "In Progress"
            RESPONDED: Final[str] = "Responded"
            UNTRACKED: Final[str] = "Untracked"  # this suggests an error has occurred
        status: Status
        response_series: MCTResponseSeries | None

        def __init__(
            self,
            status: Status,
            response_series: MCTResponseSeries | None = None
        ):
            self.status = status
            self.response_series = response_series

    # treat as immutable
    _component_address: MCTComponentAddress

    _state: State

    _status_message_queue: list[StatusMessage]

    _socket: ClientConnection | None
    _attempt_count: int
    _next_attempt_timestamp_utc: datetime.datetime
    _init_request_id: uuid.UUID | None
    _deinit_request_id: uuid.UUID | None

    # Requests are handled one at a time, with results being appended to a Response queue
    _request_series_queue: list[tuple[MCTRequestSeries, uuid.UUID]]
    _current_request_id: uuid.UUID | None
    _response_series_queue: dict[uuid.UUID, MCTResponseSeries]

    network_latency_samples_seconds: list[float]
    network_latency_seconds: float
    network_plus_offset_samples_seconds: list[float]
    controller_offset_samples_seconds: list[float]
    controller_offset_seconds: float  # how much time to be ADDED to go from controller time to component

    def __init__(
        self,
        component_address: MCTComponentAddress
    ):
        self._component_address = component_address

        self._state = Connection.State.INACTIVE

        self._status_message_queue = list()

        self._socket = None
        self._attempt_count = 0
        self._next_attempt_timestamp_utc = datetime.datetime.min
        self._init_request_id = None
        self._deinit_request_id = None

        self._request_series_queue = list()
        self._current_request_id = None
        self._response_series_queue = dict()

        self.reset_time_sync_stats()

    @abc.abstractmethod
    def create_deinitialization_request_series(self) -> MCTRequestSeries: ...

    @abc.abstractmethod
    def create_initialization_request_series(self) -> MCTRequestSeries: ...

    def dequeue_status_messages(self) -> list[StatusMessage]:
        status_messages = self._status_message_queue
        self._status_message_queue = list()
        return status_messages

    def enqueue_request_series(
        self,
        request_series: MCTRequestSeries
    ) -> uuid.UUID:
        """
        Returns the id that can be used to get the result (when it is ready)
        """
        if not self.is_active():
            raise RuntimeError("Connection is not active. Cannot yet make requests.")
        request_series_id: uuid.UUID = uuid.uuid4()
        self._request_series_queue.append((request_series, request_series_id))
        return request_series_id

    def enqueue_status_message(
        self,
        severity: SeverityLabel,
        message: str
    ) -> None:
        """
        Meant to be called by subclasses.
        Add a status message to report to a user.
        """
        self._status_message_queue.append(
            StatusMessage(
                source_label=self._component_address.label,
                severity=severity,
                message=message,
                timestamp_utc_iso8601=datetime.datetime.now(tz=datetime.timezone.utc).isoformat()))

    def get_current_state(self) -> str:
        return self._state

    def get_label(self) -> str:
        return self._component_address.label

    def get_report(self) -> ConnectionReport:
        return ConnectionReport(
            label=self._component_address.label,
            role=self._component_address.role,
            ip_address=str(self._component_address.ip_address),
            port=int(self._component_address.port),
            status=self.get_current_state())

    def get_role(self) -> str:
        return self._component_address.role

    @abc.abstractmethod
    def handle_initialization_response_series(
        self,
        response_series: MCTResponseSeries
    ) -> InitializationResult: ...

    @abc.abstractmethod
    def handle_deinitialization_response_series(
        self,
        response_series: MCTResponseSeries
    ) -> DeinitializationResult: ...

    def is_shut_down(self) -> bool:
        return self._state == Connection.State.INACTIVE

    def is_start_up_finished(self) -> bool:
        """
        Returns true if startup has completed successfully, or if it failed but finished cleanup.
        """
        return self.is_active() or self._state == Connection.State.FAILURE

    def is_active(self) -> bool:
        return self._state == Connection.State.RUNNING or self._state == Connection.State.RECONNECTING

    def pop_response_series_if_responded(
        self,
        request_series_id: uuid.UUID
    ) -> PopResponseSeriesResult:
        if request_series_id in self._response_series_queue:
            return Connection.PopResponseSeriesResult(
                status=Connection.PopResponseSeriesResult.Status.RESPONDED,
                response_series=self._response_series_queue.pop(request_series_id))
        if request_series_id == self._current_request_id:
            return Connection.PopResponseSeriesResult(
                status=Connection.PopResponseSeriesResult.Status.IN_PROGRESS)
        for _, queued_id in self._request_series_queue:
            if queued_id == request_series_id:
                return Connection.PopResponseSeriesResult(
                    status=Connection.PopResponseSeriesResult.Status.QUEUED)
        # Getting past this point indicates the request was not made, or has already been removed.
        return Connection.PopResponseSeriesResult(
            status=Connection.PopResponseSeriesResult.Status.UNTRACKED)

    def reset_time_sync_stats(self):
        self.network_latency_samples_seconds = list()
        self.network_latency_seconds = 0.0
        self.network_plus_offset_samples_seconds = list()
        self.controller_offset_samples_seconds = list()
        self.controller_offset_seconds = 0.0

    def _send_recv(self) -> SendRecvResult:

        def _response_series_converter(
            response_series_dict: dict
        ) -> MCTResponseSeries:
            series_list: list[MCTResponse] = MCTParsable.parse_dynamic_series_list(
                parsable_series_dict=response_series_dict,
                supported_types=self.supported_response_types())
            return MCTResponseSeries(series=series_list)

        if self._current_request_id is None and len(self._request_series_queue) > 0:
            request_series: MCTRequestSeries
            (request_series, self._current_request_id) = self._request_series_queue[0]
            request_series_as_str: str = request_series.model_dump_json()
            try:
                self._socket.send(request_series_as_str)
                self._request_series_queue.pop(0)
            except ConnectionClosed as e:
                self._state = Connection.State.FAILURE
                self.enqueue_status_message(
                    severity="error",
                    message=f"Connection is closed for {self._component_address.label}. Cannot send. {str(e)}")
                return Connection.SendRecvResult.FAILURE

        if self._current_request_id is not None:
            try:
                response_series_as_str: str = self._socket.recv(timeout=0.0)
                response_series_as_dict: dict = json.loads(response_series_as_str)
                response_series: MCTResponseSeries = _response_series_converter(response_series_as_dict)
                response_series.responder = self._component_address.label
                self._response_series_queue[self._current_request_id] = response_series
                self._current_request_id = None
            except TimeoutError:
                pass
            except ConnectionClosed as e:
                self._state = Connection.State.FAILURE
                self.enqueue_status_message(
                    severity="error",
                    message=f"Connection is closed for {self._component_address.label}. Cannot receive. {str(e)}")
                return Connection.SendRecvResult.FAILURE

        # TODO: Migrate this outside the class
        # for response in response_series.series:
        #     if isinstance(response, DequeueStatusMessagesResponse):
        #         for status_message in response.status_messages:
        #             status_message_dict = status_message.model_dump()
        #             status_message_dict["source_label"] = self._component_address.label
        #             self._enqueue_status_message(**status_message_dict)
        return Connection.SendRecvResult.NORMAL

    def shut_down(self) -> None:
        if self.is_active():
            self._state = Connection.State.NORMAL_DEINITIALIZING
        elif self._state == Connection.State.FAILURE:
            self._state = Connection.State.INACTIVE
        else:
            raise RuntimeError(
                f"Cannot shut down connection {self._component_address.label}. "
                "It is not in a (stable) started up state. "
                f"Current state: {self._state}")

    def start_up(self) -> None:
        if not self.is_shut_down():
            raise RuntimeError(
                f"Cannot start up connection {self._component_address.label}. "
                "It is not in a shut down state. "
                f"Current state: {self._state}")
        self._state = Connection.State.CONNECTING
        self._attempt_count = 0
        self._next_attempt_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)

    @abc.abstractmethod
    def supported_response_types(self) -> list[type[MCTResponse]]:
        return [
            DequeueStatusMessagesResponse,
            EmptyResponse,
            ErrorResponse,
            TimestampGetResponse]

    def _try_connect(self) -> ConnectionResult:
        uri: str = f"ws://{self._component_address.ip_address}:{self._component_address.port}/websocket"
        try:
            self._socket = connect(
                uri=uri,
                open_timeout=None,
                close_timeout=None,
                max_size=None)  # Default max_size may have trouble with large uncompressed images
            return Connection.ConnectionResult(success=True)
        except ConnectionError as e:
            return Connection.ConnectionResult(success=False, error_message=str(e))

    def update(self) -> None:
        if self._state == Connection.State.FAILURE or \
           self._state == Connection.State.INACTIVE:
            return
        elif self._state == Connection.State.CONNECTING:
            self._update_in_connecting_state()
        elif self._state == Connection.State.INITIALIZING:
            self._update_in_initializing_state()
        elif self._state == Connection.State.RUNNING:
            self._update_in_running_state()
        elif self._state == Connection.State.RECONNECTING:
            self._update_in_reconnecting_state()
        elif self._state == Connection.State.NORMAL_DEINITIALIZING:
            self._update_in_normal_deinitializing_state()
        elif self._state == Connection.State.NORMAL_DISCONNECTING:
            self._update_in_normal_disconnecting_state()
        elif self._state == Connection.State.FAILURE_DEINITIALIZING:
            self._update_in_failure_deinitializing_state()
        elif self._state == Connection.State.FAILURE_DISCONNECTING:
            self._update_in_failure_disconnecting_state()

    def _update_deinitialization_result(self) -> DeinitializationResult:
        if self._deinit_request_id is None:
            self._deinit_request_id = uuid.uuid4()
            self._request_series_queue.append((self.create_deinitialization_request_series(), self._deinit_request_id))

        send_recv_result: Connection.SendRecvResult = self._send_recv()
        if send_recv_result != Connection.SendRecvResult.NORMAL:
            return Connection.DeinitializationResult.FAILURE

        response_result: Connection.PopResponseSeriesResult = self.pop_response_series_if_responded(
            request_series_id=self._deinit_request_id)
        if response_result.status == Connection.PopResponseSeriesResult.Status.UNTRACKED:
            self.enqueue_status_message(
                severity="error",
                message=f"The current request ID is not recognized.")
            self._deinit_request_id = None
            return Connection.DeinitializationResult.FAILURE

        if response_result.status == Connection.PopResponseSeriesResult.Status.RESPONDED:
            self._deinit_request_id = None
            return self.handle_deinitialization_response_series(response_series=response_result.response_series)

        return Connection.DeinitializationResult.IN_PROGRESS

    def _update_initialization_result(self) -> InitializationResult:
        if self._init_request_id is None:
            self._init_request_id = uuid.uuid4()
            self._request_series_queue.append((self.create_initialization_request_series(), self._init_request_id))

        send_recv_result: Connection.SendRecvResult = self._send_recv()
        if send_recv_result != Connection.SendRecvResult.NORMAL:
            return Connection.InitializationResult.FAILURE

        response_result: Connection.PopResponseSeriesResult = self.pop_response_series_if_responded(
            request_series_id=self._init_request_id)
        if response_result.status == Connection.PopResponseSeriesResult.Status.UNTRACKED:
            self.enqueue_status_message(
                severity="error",
                message=f"The current request ID is not recognized.")
            self._init_request_id = None
            return Connection.InitializationResult.FAILURE

        if response_result.status == Connection.PopResponseSeriesResult.Status.RESPONDED:
            self._init_request_id = None
            return self.handle_initialization_response_series(response_series=response_result.response_series)

        return Connection.InitializationResult.IN_PROGRESS

    def _update_in_connecting_state(self) -> None:
        now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
        if now_utc >= self._next_attempt_timestamp_utc:
            self._attempt_count += 1
            connection_result: Connection.ConnectionResult = self._try_connect()
            if connection_result.success:
                message = f"Connection successful."
                self.enqueue_status_message(severity="info", message=message)
                self._state = Connection.State.INITIALIZING
            else:
                if self._attempt_count >= _ATTEMPT_COUNT_MAXIMUM:
                    message = \
                        f"Failed to connect, received error: {str(connection_result.error_message)}. "\
                        f"Connection is being aborted after {self._attempt_count} attempts."
                    self.enqueue_status_message(severity="error", message=message)
                    self._state = Connection.State.FAILURE
                else:
                    message: str = \
                        f"Failed to connect, received error: {str(connection_result.error_message)}. "\
                        f"Will retry in {_ATTEMPT_TIME_GAP_SECONDS} seconds."
                    self.enqueue_status_message(severity="warning", message=message)
                    self._next_attempt_timestamp_utc = now_utc + datetime.timedelta(
                        seconds=_ATTEMPT_TIME_GAP_SECONDS)

    def _update_in_failure_deinitializing_state(self) -> None:
        deinitialization_result: Connection.DeinitializationResult = self._update_deinitialization_result()
        if deinitialization_result != Connection.DeinitializationResult.IN_PROGRESS:
            self._state = Connection.State.FAILURE_DISCONNECTING

    def _update_in_failure_disconnecting_state(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._socket = None
        self._state = Connection.State.FAILURE

    def _update_in_initializing_state(self) -> None:
        initialization_result: Connection.InitializationResult = self._update_initialization_result()
        if initialization_result == Connection.InitializationResult.SUCCESS:
            self._state = Connection.State.RUNNING
        elif initialization_result == Connection.InitializationResult.FAILURE:
            self._state = Connection.State.FAILURE_DEINITIALIZING

    def _update_in_normal_deinitializing_state(self) -> None:
        deinitialization_result: Connection.DeinitializationResult = self._update_deinitialization_result()
        if deinitialization_result == Connection.DeinitializationResult.SUCCESS:
            self._state = Connection.State.NORMAL_DISCONNECTING
        elif deinitialization_result == Connection.DeinitializationResult.FAILURE:
            self._state = Connection.State.FAILURE_DISCONNECTING

    def _update_in_normal_disconnecting_state(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self._socket = None
        self._state = Connection.State.INACTIVE

    def _update_in_reconnecting_state(self) -> None:
        now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
        if now_utc >= self._next_attempt_timestamp_utc:
            connection_result: Connection.ConnectionResult = self._try_connect()
            if connection_result.success:
                message = f"Reconnection successful."
                self.enqueue_status_message(severity="info", message=message)
                self._state = Connection.State.RUNNING
            else:
                message: str = \
                    f"Failed to reconnect, received error: {str(connection_result.error_message)}. "\
                    f"Will retry in {_ATTEMPT_TIME_GAP_SECONDS} seconds."
                self.enqueue_status_message(severity="warning", message=message)
                self._next_attempt_timestamp_utc = now_utc + datetime.timedelta(
                    seconds=_ATTEMPT_TIME_GAP_SECONDS)

    def _update_in_running_state(self) -> None:
        self._send_recv()
