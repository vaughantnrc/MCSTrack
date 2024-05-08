from ..structures import CalibrationResultState
from src.common import MCastRequest
from pydantic import Field


class UpdateCalibrationResultMetadataRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "update_calibration_result_metadata"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    result_identifier: str = Field()
    result_state: CalibrationResultState = Field()
