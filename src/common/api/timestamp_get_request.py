from .mct_request import MCTRequest
from pydantic import Field


class TimestampGetRequest(MCTRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "timestamp_get_request"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    requester_timestamp_utc_iso8601: str = Field()
