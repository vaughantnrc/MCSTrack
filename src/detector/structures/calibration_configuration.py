from pydantic import BaseModel, Field


class CalibratorConfiguration(BaseModel):
    data_path: str = Field()
