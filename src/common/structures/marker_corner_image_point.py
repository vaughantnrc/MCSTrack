from pydantic import BaseModel, Field


class MarkerCornerImagePoint(BaseModel):
    x_px: float = Field()
    y_px: float = Field()
