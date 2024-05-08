from pydantic import BaseModel, Field


class PoseSolverConfiguration(BaseModel):
    serial_identifier: str = Field()
