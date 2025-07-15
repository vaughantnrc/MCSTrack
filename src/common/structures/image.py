from enum import StrEnum
import math
from pydantic import BaseModel, Field
from typing import ClassVar


class Annotation(BaseModel):
    """
    A distinct point as detected on a detector image.
    """

    # These can denote that multiple landmarks are related if they share the same
    # "base label" (the part before the first and only occurrence of this character).
    RELATION_CHARACTER: ClassVar[str] = "$"

    UNIDENTIFIED_LABEL: ClassVar[str] = str()

    feature_label: str = Field()  # Empty indicates that something was detected but not identified
    x_px: float = Field()
    y_px: float = Field()

    def base_feature_label(self) -> str:
        """
        Part of the label before the RELATION_CHARACTER.
        """
        if self.RELATION_CHARACTER not in self.feature_label:
            return self.feature_label
        return self.feature_label[0:self.feature_label.index(self.RELATION_CHARACTER)]


class ImageFormat(StrEnum):
    FORMAT_PNG = ".png"
    FORMAT_JPG = ".jpg"


class ImageResolution(BaseModel):
    x_px: int = Field()
    y_px: int = Field()

    def __eq__(self, other) -> bool:
        if type(self) is not type(other):
            return False
        return \
            self.x_px == other.x_px and \
            self.y_px == other.y_px

    def __hash__(self) -> int:
        return hash(str(self))

    def __lt__(self, other):
        if not isinstance(other, ImageResolution):
            raise ValueError()
        if self.x_px < other.x_px:
            return True
        elif self.x_px > other.x_px:
            return False
        elif self.y_px < other.y_px:
            return True
        else:
            return False

    def __str__(self):
        return f"{self.x_px}x{self.y_px}"

    @staticmethod
    def from_str(in_str: str) -> 'ImageResolution':
        if 'x' not in in_str:
            raise ValueError("in_str is expected to contain delimiter 'x'.")
        parts: list[str] = in_str.split('x')
        if len(parts) > 2:
            raise ValueError("in_str is expected to contain exactly one 'x'.")
        x_px = int(parts[0])
        y_px = int(parts[1])
        return ImageResolution(x_px=x_px, y_px=y_px)


class IntrinsicParameters(BaseModel):
    """
    Camera intrinsic parameters (focal length, optical center, distortion coefficients).
    See OpenCV's documentation: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    See Wikipedia article: https://en.wikipedia.org/wiki/Distortion_%28optics%29
    """

    focal_length_x_px: float = Field()
    focal_length_y_px: float = Field()
    optical_center_x_px: float = Field()
    optical_center_y_px: float = Field()

    radial_distortion_coefficients: list[float] = Field()  # k1, k2, k3 etc in OpenCV

    tangential_distortion_coefficients: list[float] = Field()  # p1, p2 in OpenCV

    def as_array(self) -> list[float]:
        return_value: list[float] = [
            self.focal_length_x_px,
            self.focal_length_y_px,
            self.optical_center_x_px,
            self.optical_center_y_px]
        return_value += self.get_distortion_coefficients()
        return return_value

    def get_matrix(self) -> list[list[float]]:
        """calibration matrix expected by OpenCV in some operations"""
        return \
            [[self.focal_length_x_px, 0.0, self.optical_center_x_px],
             [0.0, self.focal_length_y_px, self.optical_center_y_px],
             [0.0, 0.0, 1.0]]

    def get_distortion_coefficients(self) -> list[float]:
        """
        Distortion coefficients in array format expected by OpenCV in some operations.
        See https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
        calibrateCamera() documentation describes order of distortion coefficients that OpenCV works with
        """
        coefficients: list[float] = [
            self.radial_distortion_coefficients[0],
            self.radial_distortion_coefficients[1],
            self.tangential_distortion_coefficients[0],
            self.tangential_distortion_coefficients[1]]
        coefficients += self.radial_distortion_coefficients[2:]
        return coefficients

    @staticmethod
    def generate_zero_parameters(
        resolution_x_px: int,
        resolution_y_px: int,
        fov_x_degrees: float = 45.0,
        fov_y_degrees: float = 45.0
    ) -> "IntrinsicParameters":
        optical_center_x_px: int = int(round(resolution_x_px/2.0))
        fov_x_radians: float = fov_x_degrees * math.pi / 180.0
        focal_length_x_px = (resolution_x_px / 2.0) / math.tan(fov_x_radians / 2.0)
        optical_center_y_px: int = int(round(resolution_y_px/2.0))
        fov_y_radians: float = fov_y_degrees * math.pi / 180.0
        focal_length_y_px = (resolution_y_px / 2.0) / math.tan(fov_y_radians / 2.0)
        return IntrinsicParameters(
            focal_length_x_px=focal_length_x_px,
            focal_length_y_px=focal_length_y_px,
            optical_center_x_px=optical_center_x_px,
            optical_center_y_px=optical_center_y_px,
            radial_distortion_coefficients=[0.0, 0.0, 0.0],
            tangential_distortion_coefficients=[0.0, 0.0])


class IntrinsicCalibration(BaseModel):
    timestamp_utc: str = Field()
    image_resolution: ImageResolution = Field()
    calibrated_values: IntrinsicParameters = Field()
    supplemental_data: dict = Field()
