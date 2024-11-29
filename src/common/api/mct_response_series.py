from .mct_response import MCTResponse
from pydantic import BaseModel, Field, SerializeAsAny


class MCTResponseSeries(BaseModel):
    series: list[SerializeAsAny[MCTResponse]] = Field(default=list())
    responder: str = Field(default=str())
