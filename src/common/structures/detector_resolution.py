from .image_resolution import ImageResolution
from pydantic import BaseModel, Field


class DetectorResolution(BaseModel):
    detector_serial_identifier: str = Field()
    image_resolution: ImageResolution = Field()

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return \
            self.detector_serial_identifier == other.detector_serial_identifier and \
            self.image_resolution.x_px == other.image_resolution.x_px and \
            self.image_resolution.y_px == other.image_resolution.y_px

    def __hash__(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return f"{self.detector_serial_identifier}_{str(self.image_resolution)}"
