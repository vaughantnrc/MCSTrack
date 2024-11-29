from .mct_response import MCTResponse
from pydantic import Field
from typing import Final, Literal


class TimestampGetResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "timestamp_get_response"

    @staticmethod
    def parsable_type_identifier() -> str:
        return TimestampGetResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    requester_timestamp_utc_iso8601: str = Field()
    responder_timestamp_utc_iso8601: str = Field()
