from .calibration_image_metadata import CalibrationImageMetadata
from .calibration_result_metadata import CalibrationResultMetadata
from pydantic import BaseModel, Field


class CalibrationMapValue(BaseModel):
    image_metadata_list: list[CalibrationImageMetadata] = Field(default_factory=list)
    result_metadata_list: list[CalibrationResultMetadata] = Field(default_factory=list)
