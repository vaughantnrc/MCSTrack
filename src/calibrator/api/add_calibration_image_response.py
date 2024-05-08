from src.common import MCastResponse
from pydantic import Field


class AddCalibrationImageResponse(MCastResponse):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "add_calibration_image"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    image_identifier: str = Field()
