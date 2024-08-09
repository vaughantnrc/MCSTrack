import datetime
from .mct_response import MCTResponse
from src.common.structures.status_message import StatusMessage
from pydantic import Field


class TimestampGetResponse(MCTResponse):

    @staticmethod
    def parsable_type_identifier() -> str:
        return "timestamp_get_response"

    def timestamp_utc(self):
        return datetime.datetime.fromisoformat(self.timestamp_utc_iso8601)

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    requester_timestamp_utc_iso8601: str = Field()
    responder_timestamp_utc_iso8601: str = Field()
