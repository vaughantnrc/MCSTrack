from .api import \
    DequeueStatusMessagesRequest, \
    DequeueStatusMessagesResponse, \
    EmptyResponse, \
    ErrorResponse, \
    MCTRequest, \
    MCTRequestSeries, \
    MCTResponse, \
    MCTResponseSeries, \
    TimestampGetRequest, \
    TimestampGetResponse, \
    TimeSyncStartRequest, \
    TimeSyncStopRequest
from .exceptions import MCTParsingError
from .status_messages import \
    SeverityLabel, \
    StatusMessage, \
    StatusMessageSource
from .structures import MCTParsable
from .util import \
    PythonUtils
import abc
import datetime
from fastapi import WebSocket, WebSocketDisconnect
import logging
from typing import Callable, Optional, TypeVar


logger = logging.getLogger(__name__)


ParsableDynamicSingle = TypeVar('ParsableDynamicSingle', bound=MCTParsable)


class MCTComponent(abc.ABC):

    _status_message_source: StatusMessageSource
    time_sync_active: bool

    def __init__(
        self,
        status_source_label: str,
        send_status_messages_to_logger: bool = False
    ):
        self._status_message_source = StatusMessageSource(
            source_label=status_source_label,
            send_to_logger=send_status_messages_to_logger)
        self.time_sync_active = False

    def add_status_message(
        self,
        severity: SeverityLabel,
        message: str,
        source_label: Optional[str] = None,
        timestamp_utc_iso8601: Optional[datetime.datetime | str] = None
    ):
        self._status_message_source.enqueue_status_message(
            source_label=source_label,
            severity=severity,
            message=message,
            timestamp_utc_iso8601=timestamp_utc_iso8601)

    def add_status_subscriber(
        self,
        client_identifier: str
    ):
        self._status_message_source.add_status_subscriber(subscriber_label=client_identifier)

    def parse_dynamic_series_list(
        self,
        parsable_series_dict: dict,
        supported_types: list[type[ParsableDynamicSingle]]
    ) -> list[ParsableDynamicSingle]:
        try:
            return MCTParsable.parse_dynamic_series_list(
                parsable_series_dict=parsable_series_dict,
                supported_types=supported_types)
        except MCTParsingError as e:
            self.add_status_message(
                severity="error",
                message=e.message)
            raise e

    def dequeue_status_messages(self, **kwargs) -> DequeueStatusMessagesResponse:
        """
        :key client_identifier: str
        """
        client_identifier: str = PythonUtils.get_kwarg(
            kwargs=kwargs,
            key="client_identifier",
            arg_type=str)
        status_messages: list[StatusMessage] = self._status_message_source.pop_new_status_messages(
            subscriber_label=client_identifier)
        return DequeueStatusMessagesResponse(
            status_messages=status_messages)

    def get_status_message_source(self):
        return self._status_message_source

    @abc.abstractmethod
    def supported_request_types(self) -> dict[type[MCTRequest], Callable[[dict], MCTResponse]]:
        """
        All subclasses are expected to implement this method, even if it is simply a call to super().
        :return:
            A mapping between request type and the function that is meant to handle it.
            The function shall generally accept a kwargs dict, and the kwargs dict may contain:
            - client_identifier: str unique to the component making the request
            - request: Of type derived from MCTRequest that contains information specific to the request
        """
        return {DequeueStatusMessagesRequest: self.dequeue_status_messages,
                TimestampGetRequest: self.timestamp_get,
                TimeSyncStartRequest: self.time_sync_start,
                TimeSyncStopRequest: self.time_sync_stop}
    
    def timestamp_get(self, **kwargs) -> TimestampGetResponse:
        request: TimestampGetRequest = PythonUtils.get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=TimestampGetRequest)
        timestamp_utc_iso8601 : str = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        return TimestampGetResponse(
            requester_timestamp_utc_iso8601=request.requester_timestamp_utc_iso8601,
            responder_timestamp_utc_iso8601=timestamp_utc_iso8601)
    
    def time_sync_start(self, **kwargs) -> EmptyResponse:
        self.time_sync_active = True
        return EmptyResponse()
    
    def time_sync_stop(self, **kwargs) -> EmptyResponse:
        self.time_sync_active = False
        return EmptyResponse()
    
    async def websocket_handler(self, websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            client_identifier: str = f"{websocket.client.host}:{websocket.client.port}"
            self.add_status_subscriber(client_identifier=client_identifier)
            request_series: MCTRequestSeries
            response_series: MCTResponseSeries
            while True:
                request_series_dict = await websocket.receive_json()
                try:
                    request_series_list: list[MCTRequest] = self.parse_dynamic_series_list(
                        parsable_series_dict=request_series_dict,
                        supported_types=list(self.supported_request_types().keys()))
                except MCTParsingError as e:
                    logger.exception(str(e))
                    await websocket.send_json(MCTResponseSeries().model_dump())
                    continue
                response_series: MCTResponseSeries = self.websocket_handle_requests(
                    client_identifier=client_identifier,
                    request_series=MCTRequestSeries(series=request_series_list))
                await websocket.send_json(response_series.model_dump())
        except WebSocketDisconnect as e:
            print(f"DISCONNECTED: {str(e)}")
            logger.info(str(e))

    def websocket_handle_requests(
        self,
        client_identifier: str,
        request_series: MCTRequestSeries
    ) -> MCTResponseSeries:
        request_map: dict[type[MCTRequest], Callable] = self.supported_request_types()
        response_series: list[MCTResponse] = list()
        for request in request_series.series:
            # noinspection PyBroadException
            try:
                if type(request) in request_map:
                    response_function: Callable[[dict], MCTResponse] = request_map[type(request)]
                    response: MCTResponse = response_function(
                        client_identifier=client_identifier,
                        request=request)
                    response_series.append(response)
                else:
                    message: str = f"Received unimplemented parsable_type: {request.parsable_type}."
                    logger.error(message)
                    self.add_status_message(severity="error", message=message)
                    response_series.append(ErrorResponse(message=message))
            except Exception as e:
                message: str = f"Internal error. Failed to process request series."
                logger.error(message + " " + str(e))
                self.add_status_message(severity="error", message=message)
                response_series.append(ErrorResponse(message=message))
        return MCTResponseSeries(
            series=response_series)
