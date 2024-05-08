from src.common import MCastRequest
from pydantic import Field


class SetCaptureDeviceRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "set_capture_device"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    capture_device_id: str = Field()
