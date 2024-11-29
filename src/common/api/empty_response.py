from .mct_response import MCTResponse
from pydantic import Field
from typing import Final, Literal


class EmptyResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "empty"

    @staticmethod
    def parsable_type_identifier() -> str:
        return EmptyResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)
