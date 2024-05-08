from .mcast_response import MCastResponse
from pydantic import Field


class EmptyResponse(MCastResponse):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "empty"

    parsable_type = Field(default=parsable_type_identifier(), const=True)
