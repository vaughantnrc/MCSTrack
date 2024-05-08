import cv2
import numpy


class ImageUtils:
    """
    A "class" to group related static functions, like in a namespace.
    The class itself is not meant to be instantiated.
    """

    def __init__(self):
        raise RuntimeError("ImageUtils is not meant to be instantiated.")

    @staticmethod
    def image_resize_to_fit(
        opencv_image,
        available_size: tuple[int, int]  # x, y
    ) -> numpy.ndarray:
        available_width_px: int = available_size[0]
        available_height_px: int = available_size[1]
        image_width_px: int = opencv_image.shape[1]
        image_height_px: int = opencv_image.shape[0]
        scale: float = min(
            available_width_px / float(image_width_px),
            available_height_px / float(image_height_px))
        image_width_px = int(round(image_width_px * scale))
        image_height_px = int(round(image_height_px * scale))
        return cv2.resize(
            src=opencv_image,
            dsize=(image_width_px, image_height_px))
