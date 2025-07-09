from .calibration_image_state import CalibrationImageState
import datetime
from pydantic import BaseModel, Field


class CalibrationImageMetadata(BaseModel):
    identifier: str = Field()
    label: str = Field(default_factory=str)
    timestamp_utc: str = Field(default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc).isoformat())
    state: CalibrationImageState = Field(default=CalibrationImageState.SELECT)
