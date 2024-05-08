from .mcast_response import MCastResponse
from pydantic import Field


class ErrorResponse(MCastResponse):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "error"

    parsable_type = Field(default=parsable_type_identifier(), const=True)

    message: str = Field()
