from src.common import MCTRequest
from src.common.structures import IntrinsicParameters
from pydantic import Field


class SetIntrinsicParametersRequest(MCTRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "set_intrinsic_parameters"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
    detector_label: str = Field()
    intrinsic_parameters: IntrinsicParameters = Field()
