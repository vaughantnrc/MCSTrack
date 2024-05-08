from src.common.api.mcast_request import MCastRequest
from pydantic import BaseModel, Field


class MCastRequestSeries(BaseModel):
    series: list[MCastRequest] = Field()
