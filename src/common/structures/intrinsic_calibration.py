from .image_resolution import ImageResolution
from .intrinsic_parameters import IntrinsicParameters
from .key_value_structures import KeyValueSimpleAny
from .vec3 import Vec3
from pydantic import BaseModel, Field


class IntrinsicCalibrationFrameResult(BaseModel):
    image_identifier: str = Field()
    translation: Vec3 = Field()
    rotation: Vec3 = Field()
    translation_stdev: Vec3 = Field()
    rotation_stdev: Vec3 = Field()
    reprojection_error: float = Field()


class IntrinsicCalibration(BaseModel):
    timestamp_utc: str = Field()
    image_resolution: ImageResolution = Field()
    reprojection_error: float = Field()
    calibrated_values: IntrinsicParameters = Field()
    calibrated_stdevs: list[float] = Field()
    marker_parameters: list[KeyValueSimpleAny] = Field()
    frame_results: list[IntrinsicCalibrationFrameResult] = Field(default=list())
