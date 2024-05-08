from src.common import MCastResponse
from src.common.structures import CaptureFormat
from pydantic import Field


class GetCaptureImageResponse(MCastResponse):

    @staticmethod
    def parsable_type_identifier() -> str:
        return "get_capture_image"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    format: CaptureFormat = Field()
    image_base64: str = Field()
