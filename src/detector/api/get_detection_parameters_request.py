from src.common import MCastRequest
from pydantic import Field


class GetDetectionParametersRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "get_detection_parameters"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
