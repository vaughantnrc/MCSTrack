from src.common.api.mcast_response import MCastResponse
from pydantic import BaseModel, Field


class MCastResponseSeries(BaseModel):
    series: list[MCastResponse] = Field(default=list())
    responder: str = Field(default=str())
