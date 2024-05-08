from pydantic import BaseModel, Field


class CalibratorConfiguration(BaseModel):
    serial_identifier: str = Field()
    data_path: str = Field()
