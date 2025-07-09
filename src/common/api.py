from .status_messages import StatusMessage
from .structures import MCTParsable
import abc
from pydantic import BaseModel, Field, SerializeAsAny
from typing import Final, Literal


class MCTRequest(BaseModel, MCTParsable, abc.ABC):
    parsable_type: str


class MCTRequestSeries(BaseModel):
    series: list[SerializeAsAny[MCTRequest]] = Field()


class MCTResponse(BaseModel, MCTParsable, abc.ABC):
    parsable_type: str


class MCTResponseSeries(BaseModel):
    series: list[SerializeAsAny[MCTResponse]] = Field(default=list())
    responder: str = Field(default=str())


class EmptyResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "empty"

    @staticmethod
    def parsable_type_identifier() -> str:
        return EmptyResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class ErrorResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "error"

    @staticmethod
    def parsable_type_identifier() -> str:
        return ErrorResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    message: str = Field()


class DequeueStatusMessagesRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "dequeue_status_messages"

    @staticmethod
    def parsable_type_identifier() -> str:
        return DequeueStatusMessagesRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class DequeueStatusMessagesResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "dequeue_status_messages"

    @staticmethod
    def parsable_type_identifier() -> str:
        return DequeueStatusMessagesResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    status_messages: list[StatusMessage] = Field()


class TimeSyncStartRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "time_sync_start_request"

    @staticmethod
    def parsable_type_identifier() -> str:
        return TimeSyncStartRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class TimeSyncStopRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "time_sync_stop_request"

    @staticmethod
    def parsable_type_identifier() -> str:
        return TimeSyncStopRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)


class TimestampGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "timestamp_get_request"

    @staticmethod
    def parsable_type_identifier() -> str:
        return TimestampGetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    requester_timestamp_utc_iso8601: str = Field()


class TimestampGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "timestamp_get_response"

    @staticmethod
    def parsable_type_identifier() -> str:
        return TimestampGetResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    requester_timestamp_utc_iso8601: str = Field()
    responder_timestamp_utc_iso8601: str = Field()
