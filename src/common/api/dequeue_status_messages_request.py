from .mct_request import MCTRequest
from pydantic import Field
from typing import Final, Literal


class DequeueStatusMessagesRequest(MCTRequest):
    _TYPE_IDENTIFIER: Final[str] = "dequeue_status_messages"

    @staticmethod
    def parsable_type_identifier() -> str:
        return DequeueStatusMessagesRequest._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)
