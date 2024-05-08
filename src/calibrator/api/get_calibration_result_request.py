from src.common import MCastRequest
from pydantic import Field


class GetCalibrationResultRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "get_calibration_result"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    result_identifier: str = Field()
