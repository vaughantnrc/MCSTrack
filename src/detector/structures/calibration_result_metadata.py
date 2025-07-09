from .calibration_result_state import CalibrationResultState
import datetime
from pydantic import BaseModel, Field


class CalibrationResultMetadata(BaseModel):
    identifier: str = Field()
    label: str = Field(default_factory=str)
    timestamp_utc_iso8601: str = Field(default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc).isoformat())
    image_identifiers: list[str] = Field(default_factory=list)
    state: CalibrationResultState = Field(default=CalibrationResultState.RETAIN)

    def timestamp_utc(self):
        return datetime.datetime.fromisoformat(self.timestamp_utc_iso8601)
