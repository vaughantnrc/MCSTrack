from src.common import MCastResponse
from pydantic import Field


class GetCaptureDeviceResponse(MCastResponse):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "get_capture_device"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    capture_device_id: str = Field()
