import abc
import datetime
from typing import Final, Literal
from websockets import WebSocketClientProtocol


# TODO 2024-04-08: I don't see any particular need to have subclasses of this.
#                  It may make sense just to have a generic "dynamic" class,
#                  or to have the connector store these members more directly.
class BaseComponentConnectionDynamic(abc.ABC):

    ATTEMPT_COUNT_MAXIMUM: Final[int] = 3
    ATTEMPT_TIME_GAP_SECONDS: Final[float] = 5.0

    status: Literal["disconnecting", "disconnected", "connecting", "connected", "aborted"]
    socket: WebSocketClientProtocol | None
    attempt_count: int
    next_attempt_timestamp_utc: datetime.datetime

    def __init__(self):
        self.status = "disconnected"
        self.socket = None
        self.attempt_count = 0
        self.next_attempt_timestamp_utc = datetime.datetime.min
