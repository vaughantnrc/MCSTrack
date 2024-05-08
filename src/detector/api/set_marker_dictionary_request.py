from src.common import MCastRequest
from src.common.structures import MarkerDefinition
from pydantic import Field


class SetMarkerDictionaryRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "set_marker_dictionary"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    marker_size_bits: int = Field()
    bits_by_marker_base64: list[MarkerDefinition] = Field()
