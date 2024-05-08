from src.common import MCastRequest
from pydantic import Field


class ListCalibrationDetectorResolutionsRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "list_calibration_detector_resolutions"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
