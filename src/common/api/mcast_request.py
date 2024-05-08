from .mcast_parsable import MCastParsable
import abc
from pydantic import BaseModel


class MCastRequest(BaseModel, MCastParsable, abc.ABC):
    parsable_type: str
