from pydantic import BaseModel, Field
from typing import Union


class CalibratorConfiguration(BaseModel):
    data_path: str = Field()


class CameraConfiguration(BaseModel):
    driver: str = Field()
    capture_device: Union[str, int] = Field()  # Not used by all drivers (notably it IS used by OpenCV)


class MarkerConfiguration(BaseModel):
    method: str = Field()


class DetectorConfiguration(BaseModel):
    """
    Top-level schema for Detector initialization data
    """
    calibrator_configuration: CalibratorConfiguration = Field()
    camera_configuration: CameraConfiguration = Field()
    marker_configuration: MarkerConfiguration = Field()
