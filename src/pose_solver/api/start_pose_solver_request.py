from src.common import MCastRequest
from pydantic import Field


class StartPoseSolverRequest(MCastRequest):
    @staticmethod
    def parsable_type_identifier() -> str:
        return "start_pose_solver"

    parsable_type: str = Field(default=parsable_type_identifier(), const=True)
