from src.common import MCastRequest
from pydantic import Field


class DeleteStagedRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "clear_calibration_data_staged_for_deletion"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
