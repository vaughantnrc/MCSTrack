from .connection import \
    Connection
from src.common import \
    MCTError, \
    MCTRequestSeries, \
    MCTResponse, \
    MCTResponseSeries
import logging
import uuid

logger = logging.getLogger(__name__)


class ResponseSeriesNotExpected(MCTError):
    pass


class Router:
    _connections_by_label: dict[str, Connection]

    def __init__(
        self
    ):
        self._connections_by_label = dict()

    def add_connection(
        self,
        component_address: Connection.ComponentAddress,
        supported_response_types: dict[str, type[MCTResponse]]
    ) -> None:
        label = component_address.label
        if label in self._connections_by_label:
            raise RuntimeError(f"Connection associated with label {label} already exists.")
        return_value: Connection = Connection(
            component_address=component_address,
            supported_response_types=supported_response_types)
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

    def _reset(self):
        self._connections_by_label = dict()

    def request_series_push(
        self,
        label: str,
        request_series: MCTRequestSeries
    ) -> uuid.UUID:
        if label not in self._connections_by_label:
            raise RuntimeError(f"Failed to find connection with label {label}.")
        elif not self._connections_by_label[label].is_active():
            raise RuntimeError(f"Connection with label {label} is not active.")
        return self._connections_by_label[label].enqueue_request_series(request_series=request_series)

    def response_series_pop(
        self,
        request_series_id: uuid.UUID
    ) -> tuple[uuid.UUID | None, MCTResponseSeries | None]:
        """
        Only "pop" if there is a response (not None).
        Return value is a tuple whose elements comprise:
          - UUID of the request if no response has been received, or None
          - MCTResponseSeries if a response has been received, or None
        The dual return values allow easier reassignment of completed request ID's in calling code
        """
        for connection in self._connections_by_label.values():
            response_result: Connection.PopResponseSeriesResult = connection.pop_response_series_if_responded(
                request_series_id=request_series_id)
            if response_result.status == Connection.PopResponseSeriesResult.Status.UNTRACKED:
                continue
            elif response_result.status == Connection.PopResponseSeriesResult.Status.RESPONDED:
                return None, response_result.response_series
            else:  # queued, in progress
                return request_series_id, None  # Connection is tracking desired request series, waiting for response
        # Cannot be found
        raise ResponseSeriesNotExpected()

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
