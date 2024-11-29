from .mct_response import MCTResponse
from pydantic import Field
from typing import Final, Literal


class ErrorResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "error"

    @staticmethod
    def parsable_type_identifier() -> str:
        return ErrorResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    message: str = Field()
