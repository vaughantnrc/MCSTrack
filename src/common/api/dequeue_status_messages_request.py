from .mcast_request import MCastRequest
from pydantic import Field


class DequeueStatusMessagesRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "dequeue_status_messages"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
