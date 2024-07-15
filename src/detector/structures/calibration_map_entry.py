from .calibration_map_value import CalibrationMapValue
from src.common.structures.detector_resolution import DetectorResolution
from pydantic import BaseModel, Field


class CalibrationMapEntry(BaseModel):
    key: DetectorResolution = Field()
    value: CalibrationMapValue = Field()
