from .mct_request import MCTRequest
from pydantic import Field
from typing import Final, Literal


class TimeSyncStartRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "time_sync_start_request"

    @staticmethod
    def parsable_type_identifier() -> str:
        return TimeSyncStartRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)
