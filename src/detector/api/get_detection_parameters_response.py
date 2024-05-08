from src.common import MCastResponse
from src.common.structures import DetectionParameters
from pydantic import Field


class GetDetectionParametersResponse(MCastResponse):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "get_detection_parameters"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    parameters: DetectionParameters = Field(default_factory=DetectionParameters)
