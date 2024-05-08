from src.common import MCastRequest
from src.common.structures import DetectionParameters
from pydantic import Field


class SetDetectionParametersRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "set_detection_parameters"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    parameters: DetectionParameters = Field(default_factory=DetectionParameters)
