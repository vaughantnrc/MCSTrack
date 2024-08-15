from .mct_request import MCTRequest
from pydantic import Field


class TimeSyncStopRequest(MCTRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "time_sync_stop_request"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
