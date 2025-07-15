import base64
import cv2
from enum import StrEnum
import logging
import math  # Python's math module, not the one from this project!
import numpy
from pydantic import BaseModel, Field
from typing import ClassVar, Literal, Final


logger = logging.getLogger(__file__)

ColorMode = Literal["color", "greyscale"]


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


class ImageUtils:
    """
    A "class" to group related static functions, like in a namespace.
    The class itself is not meant to be instantiated.
    """

    def __init__(self):
        raise RuntimeError(f"{__class__.__name__} is not meant to be instantiated.")

    @staticmethod
    def base64_to_image(
        input_base64: str,
        color_mode: ColorMode = "color"
    ) -> numpy.ndarray:
        """
        Assumes 8 bits per component
        """

        image_bytes: bytes = base64.b64decode(s=input_base64)

        color_flag: int = 0
        if color_mode == "color":
            color_flag |= cv2.IMREAD_COLOR
        elif color_mode == "greyscale":
            color_flag |= cv2.IMREAD_GRAYSCALE
        else:
            logger.warning(f"Unsupported color mode specified: {color_mode}")

        opencv_image: numpy.ndarray = cv2.imdecode(
            buf=numpy.frombuffer(
                buffer=image_bytes,
                dtype=numpy.uint8),
            flags=color_flag)
        return opencv_image

    @staticmethod
    def black_image(
        resolution_px: tuple[int, int],
    ) -> numpy.ndarray:
        return numpy.zeros((resolution_px[1], resolution_px[0], 3), dtype=numpy.uint8)

    @staticmethod
    def bytes_to_base64(
        image_bytes: bytes
    ) -> str:
        return base64.b64encode(image_bytes).decode("ascii")

    @staticmethod
    def image_resize_to_fit(
        opencv_image: numpy.ndarray,
        available_size: tuple[int, int]  # x, y
    ) -> numpy.ndarray:
        # note: opencv height represented by 1st dimension
        source_resolution_px: tuple[int, int] = (opencv_image.shape[1], opencv_image.shape[0])
        image_width_px, image_height_px = ImageUtils.scale_factor_for_available_space_px(
            source_resolution_px=source_resolution_px,
            available_size_px=available_size)
        return cv2.resize(
            src=opencv_image,
            dsize=(image_width_px, image_height_px))

    @staticmethod
    def image_to_base64(
        image_data: numpy.ndarray,
        image_format: ImageFormat = ".png",
    ) -> str:
        """
        :param image_data: Expected to be an OpenCV image *or* a numpy.ndarray (theoretically - to be confirmed)
        :param image_format: e.g. ".jpg", ".png"...
        :return: base64 string representing the image
        """
        encoded_image_rgb_bytes: bytes = ImageUtils.image_to_bytes(
            image_data=image_data,
            image_format=image_format)
        encoded_image_rgb_base64: str = ImageUtils.bytes_to_base64(encoded_image_rgb_bytes)
        return encoded_image_rgb_base64

    @staticmethod
    def image_to_bytes(
        image_data: numpy.ndarray,
        image_format: ImageFormat = ".png",
    ) -> bytes:
        """
        :param image_data: Expected to be an OpenCV image *or* a numpy.ndarray (theoretically - to be confirmed)
        :param image_format: e.g. ".jpg", ".png"...
        :return: base64 string representing the image
        """
        encoded_image_rgb_single_row: numpy.array
        encoded, encoded_image_rgb_single_row = cv2.imencode(image_format, image_data)
        encoded_image_rgb_bytes: bytes = encoded_image_rgb_single_row.tobytes()
        return encoded_image_rgb_bytes

    @staticmethod
    def scale_factor_for_available_space_px(
        source_resolution_px: tuple[int, int],
        available_size_px: tuple[int, int]
    ) -> tuple[int, int]:
        source_width_px: int = source_resolution_px[0]
        source_height_px: int = source_resolution_px[1]
        available_width_px: int = available_size_px[0]
        available_height_px: int = available_size_px[1]
        scale: float = min(
            available_width_px / float(source_width_px),
            available_height_px / float(source_height_px))
        return int(round(source_width_px * scale)), int(round(source_height_px * scale))

    class StandardResolutions:
        RES_640X360: Final[ImageResolution] = ImageResolution(x_px=640, y_px=360)
        RES_640X480: Final[ImageResolution] = ImageResolution(x_px=640, y_px=480)
        RES_800X600: Final[ImageResolution] = ImageResolution(x_px=800, y_px=600)
        RES_1024X768: Final[ImageResolution] = ImageResolution(x_px=1024, y_px=768)
        RES_1280X720: Final[ImageResolution] = ImageResolution(x_px=1280, y_px=720)
        RES_1280X800: Final[ImageResolution] = ImageResolution(x_px=1280, y_px=800)
        RES_1280X1024: Final[ImageResolution] = ImageResolution(x_px=1280, y_px=1024)
        RES_1920X1080: Final[ImageResolution] = ImageResolution(x_px=1920, y_px=1080)

        @staticmethod
        def as_list():
            return [
                ImageUtils.StandardResolutions.RES_640X360,
                ImageUtils.StandardResolutions.RES_640X480,
                ImageUtils.StandardResolutions.RES_800X600,
                ImageUtils.StandardResolutions.RES_1024X768,
                ImageUtils.StandardResolutions.RES_1280X720,
                ImageUtils.StandardResolutions.RES_1280X800,
                ImageUtils.StandardResolutions.RES_1280X1024,
                ImageUtils.StandardResolutions.RES_1920X1080]
