from .mct_request import MCTRequest
from pydantic import Field
from typing import Final, Literal


class TimeSyncStopRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "time_sync_stop_request"

    @staticmethod
    def parsable_type_identifier() -> str:
        return TimeSyncStopRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)
