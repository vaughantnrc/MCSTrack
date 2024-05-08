from .mcast_parsable import MCastParsable
import abc
from pydantic import BaseModel


class MCastResponse(BaseModel, MCastParsable, abc.ABC):
    parsable_type: str
