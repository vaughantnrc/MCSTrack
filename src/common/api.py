from .serialization import MCTDeserializable
from .status import StatusMessage
import abc
from pydantic import BaseModel, Field, SerializeAsAny


class MCTRequest(BaseModel, MCTDeserializable, abc.ABC):
    parsable_type: str


class MCTRequestSeries(BaseModel):
    series: list[SerializeAsAny[MCTRequest]] = Field()


class MCTResponse(BaseModel, MCTDeserializable, abc.ABC):
    parsable_type: str


class MCTResponseSeries(BaseModel):
    series: list[SerializeAsAny[MCTResponse]] = Field(default=list())
    responder: str = Field(default=str())


class EmptyResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "empty"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())


class ErrorResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "error"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())

    message: str = Field()


class DequeueStatusMessagesRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "dequeue_status_messages"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())


class DequeueStatusMessagesResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "dequeue_status_messages"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())

    status_messages: list[StatusMessage] = Field()


class TimeSyncStartRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "time_sync_start_request"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())


class TimeSyncStopRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "time_sync_stop_request"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())


class TimestampGetRequest(MCTRequest):
    @staticmethod
    def type_identifier() -> str:
        return "timestamp_get_request"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())

    requester_timestamp_utc_iso8601: str = Field()


class TimestampGetResponse(MCTResponse):
    @staticmethod
    def type_identifier() -> str:
        return "timestamp_get_response"

    # noinspection PyTypeHints
    parsable_type: str = Field(default=type_identifier())

    requester_timestamp_utc_iso8601: str = Field()
    responder_timestamp_utc_iso8601: str = Field()
