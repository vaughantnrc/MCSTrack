from src.common import MCastRequest
from src.common.structures import ImageResolution
from pydantic import Field


class ListCalibrationResultMetadataRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "get_calibration_result_metadata_list"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    detector_serial_identifier: str = Field()
    image_resolution: ImageResolution = Field()
