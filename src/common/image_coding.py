from .structures import CaptureFormat
import base64
import cv2
import logging
import numpy
from typing import Literal


logger = logging.getLogger(__file__)

ColorMode = Literal["color", "greyscale"]


class ImageCoding:
    """
    A "class" to group related static functions, like in a namespace.
    The class itself is not meant to be instantiated.
    """

    def __init__(self):
        raise RuntimeError("ImageCoding is not meant to be instantiated.")

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
    def bytes_to_base64(
        image_bytes: bytes
    ) -> str:
        return base64.b64encode(image_bytes).decode("ascii")

    @staticmethod
    def image_to_base64(
        image_data: numpy.ndarray,
        image_format: CaptureFormat = ".png",
    ) -> str:
        """
        :param image_data: Expected to be an OpenCV image *or* a numpy.ndarray (theoretically - to be confirmed)
        :param image_format: e.g. ".jpg", ".png"...
        :return: base64 string representing the image
        """
        encoded_image_rgb_bytes: bytes = ImageCoding.image_to_bytes(
            image_data=image_data,
            image_format=image_format)
        encoded_image_rgb_base64: str = ImageCoding.bytes_to_base64(encoded_image_rgb_bytes)
        return encoded_image_rgb_base64

    @staticmethod
    def image_to_bytes(
        image_data: numpy.ndarray,
        image_format: CaptureFormat = ".png",
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
