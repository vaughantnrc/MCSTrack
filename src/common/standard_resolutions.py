from .structures import ImageResolution
from typing import Final


class StandardResolutions:
    RES_640X480: Final[ImageResolution] = ImageResolution(x_px=640, y_px=480)
    RES_1280X720: Final[ImageResolution] = ImageResolution(x_px=1280, y_px=720)
    RES_1920X1080: Final[ImageResolution] = ImageResolution(x_px=1920, y_px=1080)
