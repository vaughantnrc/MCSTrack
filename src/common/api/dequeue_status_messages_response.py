from .mct_response import MCTResponse
from src.common.structures.status_message import StatusMessage
from pydantic import Field
from typing import Final, Literal


class DequeueStatusMessagesResponse(MCTResponse):
    _TYPE_IDENTIFIER: Final[str] = "dequeue_status_messages"

    @staticmethod
    def parsable_type_identifier() -> str:
        return DequeueStatusMessagesResponse._TYPE_IDENTIFIER

    # noinspection PyTypeHints
    parsable_type: Literal[_TYPE_IDENTIFIER] = Field(default=_TYPE_IDENTIFIER)

    status_messages: list[StatusMessage] = Field()
