from src.common import MCastRequest
from pydantic import Field


class GetCalibrationImageRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "get_calibration_image"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    image_identifier: str = Field()
