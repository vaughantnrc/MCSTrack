from .mct_request import MCTRequest
from pydantic import Field


class TimeSyncStartRequest(MCTRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "time_sync_start_request"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
