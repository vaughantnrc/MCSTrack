from src.common import MCastResponse
from src.common.structures import DetectorResolution
from pydantic import Field


class ListCalibrationDetectorResolutionsResponse(MCastResponse):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "list_calibration_detector_resolutions"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    detector_resolutions: list[DetectorResolution] = Field()
