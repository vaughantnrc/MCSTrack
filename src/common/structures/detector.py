from .image import ImageResolution
import datetime
from enum import StrEnum
from pydantic import BaseModel, Field
from typing import Final


class CaptureFormat(StrEnum):
    FORMAT_PNG: Final[str] = ".png"
    FORMAT_JPG: Final[str] = ".jpg"


class MarkerCornerImagePoint(BaseModel):
    # TODO: Some types of markers may not refer to "corners" per se, so it may be worth renaming this class
    x_px: float = Field()
    y_px: float = Field()


class MarkerSnapshot(BaseModel):
    label: str = Field()
    corner_image_points: list[MarkerCornerImagePoint] = Field()


class DetectorFrame(BaseModel):
    detected_marker_snapshots: list[MarkerSnapshot] | None = Field()
    rejected_marker_snapshots: list[MarkerSnapshot] | None = Field()
    timestamp_utc_iso8601: str = Field()
    image_resolution: ImageResolution = Field()

    def timestamp_utc(self):
        return datetime.datetime.fromisoformat(self.timestamp_utc_iso8601)


class MarkerCorners:
    # TODO: Remove this class in favour of DetectorFrame
    detector_label: str
    marker_id: int
    points: list[list[float]]
    timestamp: datetime.datetime

    def __init__(
        self,
        detector_label: str,
        marker_id: int,
        points: list[list[float]],
        timestamp: datetime.datetime
    ):
        self.detector_label = detector_label
        self.marker_id = marker_id
        self.points = points
        self.timestamp = timestamp
