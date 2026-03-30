from src.common import \
    MCTDeserializable, \
    MCTRequestSeries, \
    MCTResponse, \
    MCTResponseSeries, \
    SeverityLabel, \
    StatusMessageSource
import datetime
from enum import StrEnum
from ipaddress import IPv4Address
import json
from typing import Final
from websockets import ConnectionClosed
from websockets.sync.client import connect, ClientConnection


_ATTEMPT_COUNT_MAXIMUM: Final[int] = 3
_ATTEMPT_TIME_GAP_SECONDS: Final[float] = 5.0


class Connection:
    """
    A connection represents the interface with a remote component
    """

    # Each connection shall be in one of the states listed below.

    class State(StrEnum):
        # This is the normal progression cycle ending back in "Inactive"
        INACTIVE = "Inactive"
        CONNECTING = "Connecting"
        RUNNING = "Running"
        RECONNECTING = "Reconnecting"  # Only if connection gets lost
        NORMAL_DISCONNECTING = "Disconnecting"
        # States below indicate abnormal/failed states
        FAILURE = "Failure"
        FAILURE_DISCONNECTING = "Failure - Disconnecting"

    class ComponentAddress:
        """
        Information used to establish a connection,
        there is nothing that should change here without a user's explicit input.
        """

        def __init__(
            self,
            label: str,
            role: str,
            ip_address: IPv4Address,
            port: int
        ):
            self.label = label
            self.role = role
            self.ip_address = ip_address
            self.port = port

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
        IN_PROGRESS = "In Progress"
        SUCCESS = "Success"
        FAILURE = "Failure"

    class InitializationResult(StrEnum):
        IN_PROGRESS = "In Progress"
        SUCCESS = "Success"
        FAILURE = "Failure"

    class SendRecvResult(StrEnum):
        NORMAL = "Normal"
        FAILURE = "Failure"

    class PopResponseSeriesResult:
        class Status(StrEnum):
            QUEUED = "Exists"
            IN_PROGRESS = "In Progress"
            RESPONDED = "Responded"
            UNTRACKED = "Untracked"  # this suggests an error has occurred
        status: Status
        response_series: MCTResponseSeries | None

        def __init__(
            self,
            status: Status,
            response_series: MCTResponseSeries | None = None
        ):
            self.status = status
            self.response_series = response_series

    class Report:
        """
        Human-readable information that shall be shown to a user about a connection.
        """
        label: str
        role: str
        ip_address: str
        port: int
        status: str

        def __init__(
            self,
            label: str,
            role: str,
            ip_address: str,
            port: int,
            status: str
        ):
            self.label = label
            self.role = role
            self.ip_address = ip_address
            self.port = port
            self.status = status

        def __eq__(self, other):
            if not isinstance(other, Connection.Report):
                return False
            return (
                    self.label == other.label and
                    self.role == other.role and
                    self.ip_address == other.ip_address and
                    self.port == other.port and
                    self.status == other.status)

    # treat as immutable
    _component_address: ComponentAddress
    _supported_response_types: dict[str, type[MCTResponse]]

    _status_message_source: StatusMessageSource

    _state: State

    _socket: ClientConnection | None
    _attempt_count: int
    _next_attempt_timestamp_utc: datetime.datetime

    # Requests are handled one at a time, with results being appended to a queue
    _waiting_for_response: bool
    _request_series_queue: list[MCTRequestSeries]
    _response_series_queue: list[MCTResponseSeries]

    def __init__(
        self,
        component_address: ComponentAddress,
        supported_response_types: list[type[MCTResponse]],
        status_message_source: StatusMessageSource
    ):
        self._component_address = component_address
        self._supported_response_types = {
            response_type.type_identifier(): response_type
            for response_type in supported_response_types}

        self._status_message_source = status_message_source

        self._state = Connection.State.INACTIVE

        self._socket = None
        self._attempt_count = 0
        self._next_attempt_timestamp_utc = datetime.datetime.min

        self._waiting_for_response = False
        self._request_series_queue = list()
        self._response_series_queue = list()

    def dequeue_response_series_list(self) -> list[MCTResponseSeries]:
        return_value: list[MCTResponseSeries] = list(self._response_series_queue)
        self._response_series_queue.clear()
        return return_value

    def enqueue_request_series(
        self,
        request_series: MCTRequestSeries
    ) -> None:
        if not self.is_active():
            raise RuntimeError("Connection is not active. Cannot yet make requests.")
        self._request_series_queue.append(request_series)

    def get_current_state(self) -> str:
        return self._state

    def get_label(self) -> str:
        return self._component_address.label

    def get_report(self) -> Report:
        return Connection.Report(
            label=self._component_address.label,
            role=self._component_address.role,
            ip_address=str(self._component_address.ip_address),
            port=int(self._component_address.port),
            status=self.get_current_state())

    def get_role(self) -> str:
        return self._component_address.role

    def is_shut_down(self) -> bool:
        return self._state == Connection.State.INACTIVE

    def is_start_up_finished(self) -> bool:
        """
        Returns true if startup has completed successfully, or if it failed but finished cleanup.
        """
        return self.is_active() or self._state == Connection.State.FAILURE

    def is_active(self) -> bool:
        return self._state == Connection.State.RUNNING or self._state == Connection.State.RECONNECTING

    def _send_recv(self) -> SendRecvResult:

        def _response_series_converter(
            response_series_dict: dict
        ) -> MCTResponseSeries:
            series_list: list[MCTResponse] = MCTDeserializable.deserialize_series_list(
                series_dict=response_series_dict,
                supported_types=self._supported_response_types)
            return MCTResponseSeries(
                request_id=response_series_dict.get("request_id", ""),
                series=series_list)

        if self._waiting_for_response and len(self._request_series_queue) <= 0:
            self._status_message_source.enqueue_status_message(
                source_label=self._component_address.label,
                severity=SeverityLabel.ERROR,
                message=f"Connection is in an inconsistent state - waiting for response but no requests made.")
            self._waiting_for_response = False

        if self._waiting_for_response:
            try:
                response_series_as_str: str = self._socket.recv(timeout=0.0)
                response_series_as_dict: dict = json.loads(response_series_as_str)
                response_series: MCTResponseSeries = _response_series_converter(response_series_as_dict)
                response_series.responder = self._component_address.label
                self._response_series_queue.append(response_series)
                self._request_series_queue.pop(0)
                self._waiting_for_response = False
            except TimeoutError:
                pass
            except ConnectionClosed as e:
                self._state = Connection.State.FAILURE
                self._status_message_source.enqueue_status_message(
                    source_label=self._component_address.label,
                    severity=SeverityLabel.ERROR,
                    message=f"Connection is closed for {self._component_address.label}. Cannot receive. {str(e)}")
                return Connection.SendRecvResult.FAILURE

        if not self._waiting_for_response and len(self._request_series_queue) > 0:
            request_series: MCTRequestSeries = self._request_series_queue[0]
            request_series_as_str: str = request_series.model_dump_json()
            try:
                self._socket.send(request_series_as_str)
                self._waiting_for_response = True
            except ConnectionClosed as e:
                self._state = Connection.State.FAILURE
                self._status_message_source.enqueue_status_message(
                    source_label=self._component_address.label,
                    severity=SeverityLabel.ERROR,
                    message=f"Connection is closed for {self._component_address.label}. Cannot send. {str(e)}")
                return Connection.SendRecvResult.FAILURE

        return Connection.SendRecvResult.NORMAL

    def shut_down(self) -> None:
        if self.is_active():
            self._state = Connection.State.NORMAL_DISCONNECTING
        elif self._state == Connection.State.FAILURE:
            self._state = Connection.State.INACTIVE
        elif self._state != Connection.State.INACTIVE:
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
        elif self._state == Connection.State.RUNNING:
            self._update_in_running_state()
        elif self._state == Connection.State.RECONNECTING:
            self._update_in_reconnecting_state()
        elif self._state == Connection.State.NORMAL_DISCONNECTING:
            self._update_in_normal_disconnecting_state()
        elif self._state == Connection.State.FAILURE_DISCONNECTING:
            self._update_in_failure_disconnecting_state()

    def _update_in_connecting_state(self) -> None:
        now_utc = datetime.datetime.now(tz=datetime.timezone.utc)
        if now_utc >= self._next_attempt_timestamp_utc:
            self._attempt_count += 1
            connection_result: Connection.ConnectionResult = self._try_connect()
            if connection_result.success:
                message = f"Connection successful."
                self._status_message_source.enqueue_status_message(
                    source_label=self._component_address.label,
                    severity=SeverityLabel.INFO,
                    message=message)
                self._state = Connection.State.RUNNING
            else:
                if self._attempt_count >= _ATTEMPT_COUNT_MAXIMUM:
                    message = \
                        f"Failed to connect, received error: {str(connection_result.error_message)}. "\
                        f"Connection is being aborted after {self._attempt_count} attempts."
                    self._status_message_source.enqueue_status_message(
                        source_label=self._component_address.label,
                        severity=SeverityLabel.ERROR,
                        message=message)
                    self._state = Connection.State.FAILURE
                else:
                    message: str = \
                        f"Failed to connect, received error: {str(connection_result.error_message)}. "\
                        f"Will retry in {_ATTEMPT_TIME_GAP_SECONDS} seconds."
                    self._status_message_source.enqueue_status_message(
                        source_label=self._component_address.label,
                        severity=SeverityLabel.WARNING,
                        message=message)
                    self._next_attempt_timestamp_utc = now_utc + datetime.timedelta(
                        seconds=_ATTEMPT_TIME_GAP_SECONDS)

    def _update_in_failure_disconnecting_state(self) -> None:
        if self._socket is not None:
            if self._waiting_for_response:  # TCP will not close properly unless all pending data is transmitted
                try:
                    self._socket.recv(timeout=0.0)
                except TimeoutError:
                    return  # Try again shortly
                self._waiting_for_response = False
            self._socket.close()
            self._socket = None
        self._socket = None
        self._state = Connection.State.FAILURE

    def _update_in_normal_disconnecting_state(self) -> None:
        if self._socket is not None:
            if self._waiting_for_response:  # TCP will not close properly unless all pending data is transmitted
                try:
                    self._socket.recv(timeout=0.0)
                except TimeoutError:
                    return  # Try again shortly
                self._waiting_for_response = False
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
                self._status_message_source.enqueue_status_message(
                    source_label=self._component_address.label,
                    severity=SeverityLabel.INFO,
                    message=message)
                self._state = Connection.State.RUNNING
            else:
                message: str = \
                    f"Failed to reconnect, received error: {str(connection_result.error_message)}. "\
                    f"Will retry in {_ATTEMPT_TIME_GAP_SECONDS} seconds."
                self._status_message_source.enqueue_status_message(
                    source_label=self._component_address.label,
                    severity=SeverityLabel.WARNING,
                    message=message)
                self._next_attempt_timestamp_utc = now_utc + datetime.timedelta(
                    seconds=_ATTEMPT_TIME_GAP_SECONDS)

    def _update_in_running_state(self) -> None:
        self._send_recv()
