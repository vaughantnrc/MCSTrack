from .image_processing import \
    ImageResolution, \
    ImageUtils, \
    IntrinsicCalibration
from .serialization import \
    IOUtils
from .status import \
    MCTError, \
    SeverityLabel, \
    StatusMessageSource
import abc
import datetime
from enum import StrEnum
import json
from json import JSONDecodeError
import logging
import numpy
import os
from pydantic import BaseModel, Field, ValidationError
from typing import Final
import uuid


logger = logging.getLogger(__name__)


class MCTIntrinsicCalibrationError(MCTError):
    message: str

    def __init__(self, message: str, *args):
        super().__init__(args)
        self.message = message


class _Configuration(BaseModel):
    data_path: str = Field()


class _ImageState(StrEnum):
    IGNORE = "ignore"
    SELECT = "select"
    DELETE = "delete"  # stage for deletion


class _ImageMetadata(BaseModel):
    identifier: str = Field()
    label: str = Field(default_factory=str)  # human-readable label
    timestamp_utc: str = Field(default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc).isoformat())
    state: _ImageState = Field(default=_ImageState.SELECT)


class _ResultState(StrEnum):
    # indicate to use this calibration (as opposed to simply storing it)
    # normally there shall only ever be one ACTIVE calibration for a given image resolution
    ACTIVE = "active"

    # store the calibration, but don't mark it for use
    RETAIN = "retain"

    # stage for deletion
    DELETE = "delete"


class _ResultMetadata(BaseModel):
    identifier: str = Field()
    label: str = Field(default_factory=str)
    timestamp_utc_iso8601: str = Field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc).isoformat())
    image_identifiers: list[str] = Field(default_factory=list)
    state: _ResultState = Field(default=_ResultState.RETAIN)

    def timestamp_utc(self):
        return datetime.datetime.fromisoformat(self.timestamp_utc_iso8601)


class _DataMapValue(BaseModel):
    image_metadata_list: list[_ImageMetadata] = Field(default_factory=list)
    result_metadata_list: list[_ResultMetadata] = Field(default_factory=list)


class _DataMapEntry(BaseModel):
    key: ImageResolution = Field()
    value: _DataMapValue = Field()


class _DataMap(BaseModel):
    entries: list[_DataMapEntry] = Field(default_factory=list)

    def as_dict(self) -> dict[ImageResolution, _DataMapValue]:
        return_value: dict[ImageResolution, _DataMapValue] = dict()
        for entry in self.entries:
            if entry.key not in return_value:
                return_value[entry.key] = _DataMapValue()
            for image_metadata in entry.value.image_metadata_list:
                return_value[entry.key].image_metadata_list.append(image_metadata)
            for result_metadata in entry.value.result_metadata_list:
                return_value[entry.key].result_metadata_list.append(result_metadata)
        return return_value

    @staticmethod
    def from_dict(in_dict: dict[ImageResolution, _DataMapValue]):
        entries: list[_DataMapEntry] = list()
        for key in in_dict.keys():
            entries.append(_DataMapEntry(key=key, value=in_dict[key]))
        return _DataMap(entries=entries)


class IntrinsicCalibrator(abc.ABC):
    Configuration: type[_Configuration] = _Configuration
    ImageState: type[_ImageState] = _ImageState
    ImageMetadata: type[_ImageMetadata] = _ImageMetadata
    ResultState: type[_ResultState] = _ResultState
    ResultMetadata: type[_ResultMetadata] = _ResultMetadata
    DataMap: type[_DataMap] = _DataMap

    _configuration: Configuration
    _calibration_map: dict[ImageResolution, _DataMapValue]
    _status_message_source: StatusMessageSource

    CALIBRATION_MAP_FILENAME: Final[str] = "calibration_map.json"

    IMAGE_FORMAT: Final[str] = ".png"  # work in lossless image format
    RESULT_FORMAT: Final[str] = ".json"

    def __init__(
        self,
        configuration: Configuration,
        status_message_source: StatusMessageSource
    ):
        self._configuration = configuration
        self._status_message_source = status_message_source
        if not self._exists_on_filesystem(path=self._configuration.data_path, pathtype="path", create_path=True):
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.CRITICAL,
                message="Data path does not exist and could not be created.")
            detailed_message: str = f"{self._configuration.data_path} does not exist and could not be created."
            logger.critical(detailed_message)
            raise RuntimeError(detailed_message)
        if not self.load():
            message: str = "The calibration map could not be loaded or created. "\
                           "In order to avoid data loss, the software will now abort. " \
                           "Please manually correct or remove the file in the filesystem."
            logger.critical(message)
            self._status_message_source.enqueue_status_message(severity=SeverityLabel.CRITICAL, message=message)
            raise RuntimeError(message)

    def add_image(
        self,
        image_base64: str
    ) -> str:  # id of image
        image_data: numpy.ndarray = ImageUtils.base64_to_image(input_base64=image_base64, color_mode="color")
        map_key: ImageResolution = ImageResolution(x_px=image_data.shape[1], y_px=image_data.shape[0])
        # Before making any changes to the calibration map, make sure folders exist,
        # and that this file does not somehow already exist (highly unlikely)
        key_path: str = self._path_for_map_key(map_key=map_key)
        if not self._exists_on_filesystem(path=key_path, pathtype="path", create_path=True):
            raise MCTIntrinsicCalibrationError(message=f"Failed to create storage location for input image.")
        image_identifier: str = str(uuid.uuid4())
        image_filepath = self._image_filepath(
            map_key=map_key,
            image_identifier=image_identifier)
        if os.path.exists(image_filepath):
            raise MCTIntrinsicCalibrationError(
                message=f"Image {image_identifier} appears to already exist. This is never expected to occur. "
                        f"Please try again, and if this error continues to occur then please report a bug.")
        if map_key not in self._calibration_map:
            self._calibration_map[map_key] = _DataMapValue()
        self._calibration_map[map_key].image_metadata_list.append(
            IntrinsicCalibrator.ImageMetadata(identifier=image_identifier))
        # noinspection PyTypeChecker
        image_bytes = ImageUtils.image_to_bytes(image_data=image_data, image_format=IntrinsicCalibrator.IMAGE_FORMAT)
        with (open(image_filepath, 'wb') as in_file):
            in_file.write(image_bytes)
        self.save()
        return image_identifier

    def calculate(
        self,
        image_resolution: ImageResolution
    ) -> tuple[str, IntrinsicCalibration]:
        """
        :returns: a tuple containing a result identifier (GUID as string) and the IntrinsicCalibration structure
        """

        calibration_key: ImageResolution = image_resolution
        if calibration_key not in self._calibration_map:
            raise MCTIntrinsicCalibrationError(
                message=f"No images for given resolution {str(image_resolution)} found.")

        result_identifier: str = str(uuid.uuid4())
        result_filepath = self._result_filepath(
            map_key=calibration_key,
            result_identifier=result_identifier)

        calibration_value: _DataMapValue = self._calibration_map[calibration_key]
        # don't load images right away in case of memory constraints
        image_identifiers: list[str] = list()
        for image_metadata in calibration_value.image_metadata_list:
            if image_metadata.state != _ImageState.SELECT:
                continue
            image_filepath: str = self._image_filepath(
                map_key=calibration_key,
                image_identifier=image_metadata.identifier)
            if not self._exists_on_filesystem(path=image_filepath, pathtype="filepath"):
                self._status_message_source.enqueue_status_message(
                    severity=SeverityLabel.WARNING,
                    message=f"Image {image_metadata.identifier} was not found. "
                            f"It will be omitted from the calibration.")
                continue
            image_identifiers.append(image_metadata.identifier)

        intrinsic_calibration, image_identifiers = self._calculate_implementation(
            image_resolution=image_resolution,
            image_identifiers=image_identifiers)

        IOUtils.json_write(
            filepath=result_filepath,
            json_dict=intrinsic_calibration.model_dump(),
            on_error_for_user=lambda msg: self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=msg),
            on_error_for_dev=logger.error,
            ignore_none=True)
        result_metadata: IntrinsicCalibrator.ResultMetadata = IntrinsicCalibrator.ResultMetadata(
            identifier=result_identifier,
            image_identifiers=image_identifiers)
        if len(self._calibration_map[calibration_key].result_metadata_list) == 0:
            result_metadata.state = _ResultState.ACTIVE  # No active result yet, so make this one active
        self._calibration_map[calibration_key].result_metadata_list.append(result_metadata)
        self.save()
        return result_identifier, intrinsic_calibration

    @abc.abstractmethod
    def _calculate_implementation(
        self,
        image_resolution: ImageResolution,
        image_identifiers: list[str]
    ) -> tuple[IntrinsicCalibration, list[str]]:  # image_identifiers that were actually used in calibration
        pass

    def _delete_if_exists(self, filepath: str):
        try:
            os.remove(filepath)
        except FileNotFoundError as e:
            logger.error(e)
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Failed to remove a file from the calibrator because it does not exist. "
                        f"See its internal log for details.")
        except OSError as e:
            logger.error(e)
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=f"Failed to remove a file from the calibrator due to an unexpected reason. "
                        f"See its internal log for details.")

    def delete_staged(self) -> None:
        for calibration_key in self._calibration_map.keys():
            calibration_value: _DataMapValue = self._calibration_map[calibration_key]
            image_indices_to_delete: list = list()
            for image_index, image in enumerate(calibration_value.image_metadata_list):
                if image.state == _ImageState.DELETE:
                    self._delete_if_exists(self._image_filepath(
                        map_key=calibration_key,
                        image_identifier=image.identifier))
                    image_indices_to_delete.append(image_index)
            for i in reversed(image_indices_to_delete):
                del calibration_value.image_metadata_list[i]
            result_indices_to_delete: list = list()
            for result_index, result in enumerate(calibration_value.result_metadata_list):
                if result.state == _ResultState.DELETE:
                    self._delete_if_exists(self._result_filepath(
                        map_key=calibration_key,
                        result_identifier=result.identifier))
                    result_indices_to_delete.append(result_index)
            for i in reversed(result_indices_to_delete):
                del calibration_value.result_metadata_list[i]
        self.save()

    def _exists_on_filesystem(
        self,
        path: str,
        pathtype: IOUtils.PathType,
        create_path: bool = False
    ) -> bool:
        return IOUtils.exists(
            path=path,
            pathtype=pathtype,
            create_path=create_path,
            on_error_for_user=lambda msg: self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=msg),
            on_error_for_dev=logger.error)

    # noinspection DuplicatedCode
    def get_image(
        self,
        image_identifier: str
    ) -> str:  # image in base64
        found_count: int = 0
        matching_image_resolution: ImageResolution | None = None
        for image_resolution in self._calibration_map:
            for image in self._calibration_map[image_resolution].image_metadata_list:
                if image.identifier == image_identifier:
                    found_count += 1
                    matching_image_resolution = image_resolution
                    break
        if found_count < 1:
            raise MCTIntrinsicCalibrationError(
                message=f"Image identifier {image_identifier} is not associated with any image.")
        elif found_count > 1:
            raise MCTIntrinsicCalibrationError(
                message=f"Image identifier {image_identifier} is associated with multiple images.")

        image_filepath = self._image_filepath(
            map_key=matching_image_resolution,
            image_identifier=image_identifier)
        if not os.path.exists(image_filepath):
            raise MCTIntrinsicCalibrationError(
                message=f"File does not exist for image {image_identifier} "
                        f"and given resolution {str(matching_image_resolution)}.")
        image_bytes: bytes
        try:
            with (open(image_filepath, 'rb') as in_file):
                image_bytes = in_file.read()
        except OSError:
            raise MCTIntrinsicCalibrationError(
                message=f"Failed to open image {image_identifier} for "
                        f"given resolution {str(matching_image_resolution)}.")
        image_base64 = ImageUtils.bytes_to_base64(image_bytes=image_bytes)
        return image_base64

    # noinspection DuplicatedCode
    def get_result(
        self,
        result_identifier: str
    ) -> IntrinsicCalibration:
        found_count: int = 0
        matching_image_resolution: ImageResolution | None = None
        for image_resolution in self._calibration_map:
            for result in self._calibration_map[image_resolution].result_metadata_list:
                if result.identifier == result_identifier:
                    found_count += 1
                    matching_image_resolution = image_resolution
                    break
        if found_count < 1:
            raise MCTIntrinsicCalibrationError(
                message=f"Image identifier {result_identifier} is not associated with any result.")
        elif found_count > 1:
            raise MCTIntrinsicCalibrationError(
                message=f"Image identifier {result_identifier} is associated with multiple results.")

        return self._get_result_calibration_from_file(
            image_resolution=matching_image_resolution,
            result_identifier=result_identifier)

    def get_result_active(
        self,
        image_resolution: ImageResolution
    ) -> IntrinsicCalibration | None:
        active_count: int = 0
        matched_metadata: IntrinsicCalibrator.ResultMetadata | None = None
        if image_resolution in self._calibration_map:
            result_count: int = len(self._calibration_map[image_resolution].result_metadata_list)
            if result_count > 0:
                matched_metadata = self._calibration_map[image_resolution].result_metadata_list[0]
                if matched_metadata.state == _ResultState.ACTIVE:
                    active_count = 1
                for result_index in range(1, result_count):
                    result_metadata = self._calibration_map[image_resolution].result_metadata_list[result_index]
                    if matched_metadata.state == _ResultState.DELETE:
                        matched_metadata = result_metadata
                        continue  # basically we ignore any data staged for DELETE
                    elif matched_metadata.state == _ResultState.RETAIN:
                        if result_metadata.state == _ResultState.ACTIVE:
                            active_count += 1
                            matched_metadata = result_metadata
                            continue  # ACTIVE shall of course take priority
                        elif result_metadata.timestamp_utc() > matched_metadata.timestamp_utc():
                            matched_metadata = result_metadata
                    else:  # matched_result_metadata.state == CalibrationResultState.ACTIVE:
                        if result_metadata.state == _ResultState.ACTIVE:
                            # BOTH metadata are marked ACTIVE. This is not expected to occur. Indicates a problem.
                            active_count += 1
                            if result_metadata.timestamp_utc() > matched_metadata.timestamp_utc():
                                matched_metadata = result_metadata
        if matched_metadata is None or \
           matched_metadata.state == _ResultState.DELETE:  # no result that is not marked DELETE
            return None

        if active_count < 1:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.WARNING,
                message=f"No result metadata is active for resolution {str(image_resolution)}."
                        "Returning latest result.")
        elif active_count > 1:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.WARNING,
                message=f"Multiple result metadata are active for resolution {str(image_resolution)}. "
                        "Returning latest active result. "
                        "To recover from this ambiguous state, it is strong recommended to explicitly set "
                        "one of the results as \"active\", which will reset others to \"retain\".")

        return self._get_result_calibration_from_file(
            image_resolution=image_resolution,
            result_identifier=matched_metadata.identifier)

    def _get_result_calibration_from_file(
        self,
        image_resolution: ImageResolution,
        result_identifier: str
    ) -> IntrinsicCalibration:
        result_filepath = self._result_filepath(
            map_key=image_resolution,
            result_identifier=result_identifier)
        if not os.path.exists(result_filepath):
            raise MCTIntrinsicCalibrationError(
                message=f"File does not exist for result {result_identifier} "
                        f"and given resolution {str(image_resolution)}.")
        result_json_raw: str
        try:
            with (open(result_filepath, 'r') as in_file):
                result_json_raw = in_file.read()
        except OSError:
            raise MCTIntrinsicCalibrationError(
                message=f"Failed to open result {result_identifier} for "
                        f"given resolution {str(image_resolution)}.")
        result_json_dict: dict
        try:
            result_json_dict = dict(json.loads(result_json_raw))
        except JSONDecodeError:
            raise MCTIntrinsicCalibrationError(
                message=f"Failed to parse result {result_identifier} for "
                        f"given resolution {str(image_resolution)}.")
        intrinsic_calibration: IntrinsicCalibration = IntrinsicCalibration(**result_json_dict)
        return intrinsic_calibration

    def _image_filepath(
        self,
        map_key: ImageResolution,
        image_identifier: str
    ) -> str:
        key_path: str = self._path_for_map_key(map_key=map_key)
        return os.path.join(
            key_path,
            image_identifier + IntrinsicCalibrator.IMAGE_FORMAT)

    def list_resolutions(self) -> list[ImageResolution]:
        resolutions: list[ImageResolution] = list(self._calibration_map.keys())
        return resolutions

    # noinspection DuplicatedCode
    def list_image_metadata(
        self,
        image_resolution: ImageResolution
    ) -> list[ImageMetadata]:
        image_metadata_list: list[IntrinsicCalibrator.ImageMetadata] = list()
        map_key: ImageResolution = image_resolution
        if map_key in self._calibration_map:
            image_metadata_list = self._calibration_map[map_key].image_metadata_list
        return image_metadata_list

    # noinspection DuplicatedCode
    def list_result_metadata(
        self,
        image_resolution: ImageResolution
    ) -> list[ResultMetadata]:
        result_metadata_list: list[IntrinsicCalibrator.ResultMetadata] = list()
        map_key: ImageResolution = image_resolution
        if map_key in self._calibration_map:
            result_metadata_list = self._calibration_map[map_key].result_metadata_list
        return result_metadata_list

    def load(self) -> bool:
        """
        :return: True if loaded or if it can be created without overwriting existing data. False otherwise.
        """
        calibration_map_filepath: str = self._map_filepath()
        if not os.path.exists(calibration_map_filepath):
            self._calibration_map = dict()
            return True
        elif not os.path.isfile(calibration_map_filepath):
            logger.critical(f"Calibration map file location {calibration_map_filepath} exists but is not a file.")
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.CRITICAL,
                message="Filepath location for calibration map exists but is not a file. "
                        "Most likely a directory exists at that location, "
                        "and it needs to be manually removed.")
            return False
        json_dict: dict = IOUtils.hjson_read(
            filepath=calibration_map_filepath,
            on_error_for_user=lambda msg: self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=msg),
            on_error_for_dev=logger.error)
        if not json_dict:
            logger.error(f"Failed to load calibration map from file {calibration_map_filepath}.")
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message="Failed to load calibration map from file.")
            return False
        calibration_map: IntrinsicCalibrator.DataMap
        try:
            calibration_map = IntrinsicCalibrator.DataMap(**json_dict)
        except ValidationError as e:
            logger.error(e)
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message="Failed to parse calibration map from file.")
            return False
        self._calibration_map = calibration_map.as_dict()
        return True

    def _map_filepath(self) -> str:
        return os.path.join(self._configuration.data_path, IntrinsicCalibrator.CALIBRATION_MAP_FILENAME)

    def _path_for_map_key(
        self,
        map_key: ImageResolution
    ) -> str:
        return os.path.join(self._configuration.data_path, str(map_key))

    def _result_filepath(
        self,
        map_key: ImageResolution,
        result_identifier: str
    ) -> str:
        key_path: str = self._path_for_map_key(map_key=map_key)
        return os.path.join(
            key_path,
            result_identifier + IntrinsicCalibrator.RESULT_FORMAT)

    def save(self) -> None:
        IOUtils.json_write(
            filepath=self._map_filepath(),
            json_dict=IntrinsicCalibrator.DataMap.from_dict(self._calibration_map).model_dump(),
            on_error_for_user=lambda msg: self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.ERROR,
                message=msg),
            on_error_for_dev=logger.error)

    # noinspection DuplicatedCode
    def update_image_metadata(
        self,
        image_identifier: str,
        image_state: ImageState,
        image_label: str | None
    ) -> None:
        found_count: int = 0
        for map_key in self._calibration_map:
            for image in self._calibration_map[map_key].image_metadata_list:
                if image.identifier == image_identifier:
                    image.state = image_state
                    if image_label is not None:
                        image.label = image_label
                    found_count += 1
                    break
        if found_count < 1:
            raise MCTIntrinsicCalibrationError(
                message=f"Image identifier {image_identifier} is not associated with any image.")
        elif found_count > 1:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.WARNING,
                message=f"Image identifier {image_identifier} is associated with multiple images.")
        self.save()

    # noinspection DuplicatedCode
    def update_result_metadata(
        self,
        result_identifier: str,
        result_state: ResultState,
        result_label: str | None = None
    ) -> None:
        found_count: int = 0
        matching_map_keys: set[ImageResolution] = set()  # Normally this shall be of size exactly 1
        for map_key in self._calibration_map:
            for result in self._calibration_map[map_key].result_metadata_list:
                if result.identifier == result_identifier:
                    result.state = result_state
                    if result_label is not None:
                        result.label = result_label
                    found_count += 1
                    matching_map_keys.add(map_key)
                    break

        # Some cleanup as applicable
        if result_state == _ResultState.ACTIVE:
            for map_key in matching_map_keys:
                # If size greater than 1, something is wrong... but nonetheless
                # we'll ensure there is only one active result per resolution
                for result in self._calibration_map[map_key].result_metadata_list:
                    if result.identifier != result_identifier and result.state == _ResultState.ACTIVE:
                        result.state = _ResultState.RETAIN

        if found_count < 1:
            raise MCTIntrinsicCalibrationError(
                message=f"Result identifier {result_identifier} is not associated with any result.")
        elif found_count > 1:
            self._status_message_source.enqueue_status_message(
                severity=SeverityLabel.WARNING,
                message=f"Result identifier {result_identifier} is associated with multiple results. "
                        "This suggests that the calibration map is in an inconsistent state. "
                        "It may be prudent to either manually correct it, or recreate it.")

        self.save()
