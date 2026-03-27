from .connection import \
    Connection
from src.common import \
    MCTRequestSeries, \
    MCTResponse, \
    MCTResponseSeries, \
    StatusMessageSource
import logging
import uuid
from typing import Callable

logger = logging.getLogger(__name__)


class CallbackRouter:
    """
    Scope: Maintaining a list of functions to eventually call back once certain data are available.
    """

    CallbackFunction = Callable[
        [MCTResponseSeries,  # response from remote MCTComponent
        dict[str, ...]],     # pass-through arguments
        None]
    Callback = tuple[CallbackFunction, dict[str, ...]]

    _callbacks_by_id: dict[uuid.UUID, Callback]

    def __init__(self):
        self._callbacks_by_id = dict()

    def add_callback(
        self,
        request_id: uuid.UUID,
        callback: CallbackFunction,
        passthrough_arguments: dict[str, ...] | None = None
    ):
        if passthrough_arguments is None:
            passthrough_arguments = dict()
        self._callbacks_by_id[request_id] = (callback, passthrough_arguments)

    def handle_callback(
        self,
        response_series: MCTResponseSeries
    ) -> None:
        request_id: uuid.UUID = response_series.request_id
        callback_tuple: CallbackRouter.Callback | None = self._callbacks_by_id.pop(request_id, None)
        if callback_tuple is not None:
            callback: CallbackRouter.CallbackFunction = callback_tuple[0]
            passthrough_arguments: dict[str, ...] = callback_tuple[1]
            callback(response_series, passthrough_arguments)

    def remove_callback(
        self,
        request_id: uuid.UUID
    ) -> None:
        self._callbacks_by_id.pop(request_id, None)

    def reset(self):
        self._callbacks_by_id = dict()


class ConnectionRouter:
    """
    Scope: Maintaining a list of Connections.
    """

    _connections_by_label: dict[str, Connection]

    def __init__(
        self
    ):
        self._connections_by_label = dict()

    def add_connection(
        self,
        component_address: Connection.ComponentAddress,
        supported_response_types: dict[str, type[MCTResponse]],
        status_message_source: StatusMessageSource
    ) -> None:
        label = component_address.label
        if label in self._connections_by_label:
            raise RuntimeError(f"Connection associated with label {label} already exists.")
        return_value: Connection = Connection(
            component_address=component_address,
            supported_response_types=supported_response_types,
            status_message_source=status_message_source)
        self._connections_by_label[label] = return_value
        return return_value

    def get_connection_reports(self) -> list[Connection.Report]:
        return_value: list[Connection.Report] = list()
        for connection in self._connections_by_label.values():
            return_value.append(connection.get_report())
        return return_value

    def get_connection(
        self,
        label: str
    ) -> Connection | None:
        if label not in self._connections_by_label:
            return None
        return self._connections_by_label[label]

    def is_shut_down_finished(self) -> bool:
        finished: bool = True
        for connection in self._connections_by_label.values():
            if not connection.is_shut_down():
                finished = False
                break
        return finished

    def is_start_up_finished(self) -> bool:
        finished: bool = True
        for connection in self._connections_by_label.values():
            if not connection.is_start_up_finished():
                finished = False
                break
        return finished

    def remove_connection(
        self,
        label: str
    ):
        if label not in self._connections_by_label:
            raise RuntimeError(f"Connection associated with label {label} does not exist.")
        self._connections_by_label.pop(label)

    def reset(self):
        self._connections_by_label = dict()

    def enqueue_request_series(
        self,
        label: str,
        request_series: MCTRequestSeries
    ) -> None:
        if label not in self._connections_by_label:
            raise RuntimeError(f"Failed to find connection with label {label}.")
        elif not self._connections_by_label[label].is_active():
            raise RuntimeError(f"Connection with label {label} is not active.")
        self._connections_by_label[label].enqueue_request_series(request_series=request_series)

    def dequeue_response_series_lists(self) -> list[list[MCTResponseSeries]]:
        return_value: list[list[MCTResponseSeries]] = list()
        for connection in self._connections_by_label.values():
            return_value.append(connection.dequeue_response_series_list())
        return return_value

    def start_up(self) -> None:
        for connection in self._connections_by_label.values():
            connection.start_up()

    def shut_down(self) -> None:
        for connection in self._connections_by_label.values():
            connection.shut_down()

    # Right now this function doesn't update on its own - must be called externally and regularly
    def update(
        self
    ) -> None:
        for connection in self._connections_by_label.values():
            connection.update()
