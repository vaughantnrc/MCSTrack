from .exceptions import MCTDetectorRuntimeError
from .structures import ImageRecorderConfiguration
from src.common import \
    ImageCoding, \
    StatusMessageSource
from src.common.util import IOUtils
import base64
import datetime
from enum import StrEnum
import io
import logging
import numpy
import os
import shutil
from typing import Final
import zipfile


logger = logging.getLogger(__name__)


class ImageRecorder:

    class Status(StrEnum):
        IDLE: Final[str] = "IDLE"
        RUNNING: Final[str] = "RUNNING"

    _configuration: ImageRecorderConfiguration
    _status_message_source: StatusMessageSource
    _recording_status: Status
    _recording_last_timestamp: datetime.datetime | None
    _recording_image_count: int
    _remaining_time_seconds: float

    IMAGE_FORMAT: Final[str] = ".png"  # work in lossless image format
    ZIP_IMAGE_PATH: Final[str] = "images"

    def __init__(
        self,
        configuration: ImageRecorderConfiguration,
        status_message_source: StatusMessageSource
    ):
        self._configuration = configuration
        self._status_message_source = status_message_source
        self._recording_status = ImageRecorder.Status.IDLE
        self._recording_last_timestamp = None
        self._recording_image_count = 0
        self._remaining_time_seconds = 0.0

    def clear(self) -> None:
        image_path_contents: list[str] = os.listdir(self._configuration.image_path)
        for image_path_content in image_path_contents:
            if image_path_content.endswith(ImageRecorder.IMAGE_FORMAT):
                os.remove(os.path.join(self._configuration.image_path, image_path_content))
        self._recording_image_count = 0

    def get_image_count(self) -> int:
        return self._recording_image_count

    def get_remaining_time_seconds(self) -> float:
        return self._remaining_time_seconds

    def get_status(self) -> Status:
        return self._recording_status

    def retrieve_zip_base64(self) -> str:
        bytes_io: io.BytesIO = io.BytesIO()
        with zipfile.ZipFile(bytes_io, 'a', zipfile.ZIP_DEFLATED, False) as zip_memory:
            image_path_contents: list[str] = os.listdir(self._configuration.image_path)
            for image_path_content in image_path_contents:
                local_filepath: str = os.path.join(self._configuration.image_path, image_path_content)
                if not image_path_content.endswith(ImageRecorder.IMAGE_FORMAT):
                    logger.warning(f"While retrieving recording, found non-image file {local_filepath}.")
                    continue
                zip_filepath: str = os.path.join(ImageRecorder.ZIP_IMAGE_PATH, image_path_content)
                with open(local_filepath, 'rb') as image_file:
                    zip_memory.writestr(zip_filepath, image_file.read())
        archive_bytes: bytes = bytes_io.getvalue()
        archive_base64: str = base64.b64encode(archive_bytes).decode("ascii")
        return archive_base64

    def start(
        self,
        recording_duration_seconds: float
    ) -> None:
        image_path_exists: bool = IOUtils.exists(
            path=self._configuration.image_path,
            pathtype="path",
            create_path=True,
            on_error_for_user=lambda msg: self._status_message_source.enqueue_status_message(
                severity="error",
                message=msg),
            on_error_for_dev=logger.error)
        if not image_path_exists:
            general_message: str = f"Could not find or create recording path."
            self._status_message_source.enqueue_status_message(
                severity="error",
                message=general_message)
            detailed_message: str = f"{self._configuration.image_path} does not exist and could not be created."
            logger.error(detailed_message)
            self._recording_status = ImageRecorder.Status.IDLE
            raise MCTDetectorRuntimeError(message=general_message)
        self.clear()
        if len(os.listdir(self._configuration.image_path)) > 0:
            general_message: str = f"Non-image contents were found in image path and could not be safely deleted."
            self._status_message_source.enqueue_status_message(
                severity="error",
                message=general_message)
            detailed_message: str = f"{self._configuration.image_path} contains non-image contents, will not clear."
            logger.error(detailed_message)
            self._recording_status = ImageRecorder.Status.IDLE
            raise MCTDetectorRuntimeError(message=general_message)
        self._recording_status = ImageRecorder.Status.RUNNING
        self._remaining_time_seconds = recording_duration_seconds
        return

    def stop(self) -> None:
        self._recording_status = ImageRecorder.Status.IDLE
        self._recording_last_timestamp = None
        self._remaining_time_seconds = 0.0

    # noinspection DuplicatedCode
    def update(
        self,
        image_data: numpy.ndarray,
        image_timestamp: datetime.datetime
    ) -> None:

        if self._recording_status != ImageRecorder.Status.RUNNING:
            return

        if self._recording_last_timestamp is not None:
            if image_timestamp <= self._recording_last_timestamp:
                return
            self._remaining_time_seconds -= (image_timestamp - self._recording_last_timestamp).total_seconds()

        self._recording_last_timestamp = image_timestamp
        image_filename = image_timestamp.isoformat()\
            .replace(':', '')\
            .replace('-', '')\
            .replace('T', '')\
            .replace('.', '') + ImageRecorder.IMAGE_FORMAT
        image_filepath = os.path.join(self._configuration.image_path, image_filename)
        # noinspection PyTypeChecker
        image_bytes = ImageCoding.image_to_bytes(image_data=image_data, image_format=ImageRecorder.IMAGE_FORMAT)
        image_byte_count: int = len(image_bytes)

        _, _, free_disk_byte_count = shutil.disk_usage(self._configuration.image_path)
        if free_disk_byte_count < image_byte_count:
            message: str = "Stopping image recording before writing current image due to limited disk space."
            self._status_message_source.enqueue_status_message(
                severity="error",
                message=message)
            logger.info(message)
            self.stop()
            return  # Don't even try to write image

        if (free_disk_byte_count - image_byte_count) < self._configuration.min_disk_byte_count:
            message: str = "Stopping image recording after writing current image due to limited disk space."
            self._status_message_source.enqueue_status_message(
                severity="error",
                message=message)
            logger.info(message)
            self.stop()

        if self._remaining_time_seconds < 0.0:
            self.stop()

        with (open(image_filepath, 'wb') as in_file):
            in_file.write(image_bytes)
        self._recording_image_count += 1
