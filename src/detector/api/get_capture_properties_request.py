from src.common import MCastRequest
from pydantic import Field


class GetCapturePropertiesRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "get_capture_properties"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
