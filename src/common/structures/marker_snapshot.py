from .marker_corner_image_point import MarkerCornerImagePoint
from pydantic import BaseModel, Field


class MarkerSnapshot(BaseModel):
    label: str = Field()
    corner_image_points: list[MarkerCornerImagePoint] = Field()
