from .camera_connection import CameraConnection
from src.common.structures import IntrinsicParameters
from pydantic import BaseModel, Field


class DetectorConfiguration(BaseModel):
    """
    Top-level schema for Detector initialization data
    """
    serial_identifier: str = Field()
    camera_connection: CameraConnection = Field()
    camera_intrinsic_parameters: dict[str, IntrinsicParameters] | None = Field(default=dict())
