from .exceptions import \
    MCTError
from .status_messages import \
    SeverityLabel, \
    StatusMessageSource
from .structures import \
    ImageFormat, \
    ImageResolution, \
    KeyValueSimpleAny, \
    KeyValueMetaAbstract
import abc
import base64
import cv2
import datetime
from enum import StrEnum
import numpy
from pydantic import BaseModel, Field
from typing import Union


class _Configuration(BaseModel):
    driver: str = Field()
    capture_device: Union[str, int] = Field()  # Not used by all drivers (notably it IS used by OpenCV)


class _Status(StrEnum):
    STOPPED = "STOPPED"
    RUNNING = "RUNNING"
    FAILURE = "FAILURE"


class MCTCameraRuntimeError(MCTError):
    message: str

    def __init__(self, message: str, *args):
        super().__init__(args)
        self.message = message


class Camera(abc.ABC):
    """
    Functions may raise MCTCameraRuntimeError
    """

    Status: type[_Status] = _Status
    Configuration: type[_Configuration] = _Configuration

    _configuration: Configuration
    _status: Status
    _status_message_source: StatusMessageSource

    def __init__(
        self,
        configuration: Configuration,
        status_message_source: StatusMessageSource
    ):
        self._configuration = configuration
        self._status_message_source = status_message_source

    def __del__(self):
        pass

    def add_status_message(
        self,
        severity: SeverityLabel,
        message: str
    ) -> None:
        self._status_message_source.enqueue_status_message(severity=severity, message=message)

    def get_encoded_image(
        self,
        image_format: ImageFormat,
        requested_resolution: ImageResolution | None  # None means to not alter the image dimensions
    ) -> str:
        image: numpy.ndarray = self.get_image()
        if requested_resolution is not None:
            image = cv2.resize(src=image, dsize=(requested_resolution.x_px, requested_resolution.y_px))
        encoded_frame: bool
        encoded_image_rgb_single_row: numpy.array
        encoded, encoded_image_rgb_single_row = cv2.imencode(image_format, image)
        encoded_image_rgb_bytes: bytes = encoded_image_rgb_single_row.tobytes()
        # noinspection PyTypeChecker
        encoded_image_rgb_base64: str = base64.b64encode(encoded_image_rgb_bytes)
        return encoded_image_rgb_base64

    def get_status(self) -> Status:
        return self._status

    def set_status(self, status: Status) -> None:
        self._status = status

    @abc.abstractmethod
    def get_changed_timestamp(self) -> datetime.datetime: ...

    @abc.abstractmethod
    def get_image(self) -> numpy.ndarray: ...

    @abc.abstractmethod
    def get_parameters(self) -> list[KeyValueMetaAbstract]: ...

    @abc.abstractmethod
    def get_resolution(self) -> ImageResolution: ...

    @staticmethod
    @abc.abstractmethod
    def get_type_identifier() -> str: ...  # Unique string associated with this type

    @abc.abstractmethod
    def set_parameters(self, parameters: list[KeyValueSimpleAny]) -> None: ...

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def update(self) -> None: ...
