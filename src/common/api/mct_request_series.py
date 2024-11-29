from .mct_request import MCTRequest
from pydantic import BaseModel, Field, SerializeAsAny


class MCTRequestSeries(BaseModel):
    series: list[SerializeAsAny[MCTRequest]] = Field()
