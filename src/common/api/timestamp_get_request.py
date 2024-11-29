from .mct_request import MCTRequest
from pydantic import Field
from typing import Final, Literal


class TimestampGetRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "timestamp_get_request"

    @staticmethod
    def parsable_type_identifier() -> str:
        return TimestampGetRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    requester_timestamp_utc_iso8601: str = Field()
