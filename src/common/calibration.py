from .image_processing import \
    ImageFormat, \
    ImageResolution, \
    ImageUtils
from .math import \
    IntrinsicParameters, \
    Matrix4x4
from .serialization import \
    IOUtils
from .status import \
    MCTError
import abc
import datetime
from enum import StrEnum
import logging
import numpy
import os
from pydantic import BaseModel, Field, ValidationError
from typing import Final, Optional
import uuid


logger = logging.getLogger(__name__)


class CalibrationErrorReason(StrEnum):
    INITIALIZATION: Final[str] = "initialization"
    INVALID_INPUT: Final[str] = "invalid_input"
    INVALID_STATE: Final[str] = "invalid_state"
    DATA_NOT_FOUND: Final[str] = "data_not_found"
    COMPUTATION_FAILURE: Final[str] = "computation_failure"


_PUBLIC_MESSAGE_KEY: Final[str] = "public_message"
_PRIVATE_MESSAGE_KEY: Final[str] = "private_message"


class MCTCalibrationError(MCTError):
    public_message: str | None
    private_message: str
    reason: CalibrationErrorReason

    def __init__(
        self,
        reason: CalibrationErrorReason,
        public_message: str | None = None,
        private_message: str | None = None,
        *args
    ):
        super().__init__(args)
        self.reason = reason
        self.public_message = public_message
        self.private_message = private_message
        if self.private_message is None and self.public_message is not None:
            self.private_message = self.private_message


_RESULT_FORMAT: Final[str] = ".json"


# =====================================================================================================================
# Internal structures applicable to both intrinsic and extrinsic calibrations
# =====================================================================================================================


class _Configuration(BaseModel):
    data_path: str = Field()


class _ImageState(StrEnum):
    IGNORE = "ignore"
    SELECT = "select"
    DELETE = "delete"  # stage for deletion


class _ImageMetadata(BaseModel):
    identifier: str = Field()
    filepath: str = Field()
    detector_label: str = Field()
    resolution: ImageResolution = Field()
    label: str = Field(default_factory=str)  # human-readable label
    timestamp_utc_iso8601: str = Field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc).isoformat())
    state: _ImageState = Field(default=_ImageState.SELECT)

    def is_selected(self):
        return self.state == _ImageState.SELECT


class _ResultState(StrEnum):
    ACTIVE = "active"  # will be stored AND marked for use. Only one result expected to be active per image resolution.
    RETAIN = "retain"  # store, but do not use
    DELETE = "delete"  # stage for deletion


class _ResultMetadata(BaseModel):
    identifier: str = Field()
    filepath: str = Field()
    resolution: ImageResolution | None = Field(default=None)  # Used in intrinsic, not currently used in extrinsic
    label: str = Field(default_factory=str)
    timestamp_utc_iso8601: str = Field(
        default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc).isoformat())
    image_identifiers: list[str] = Field(default_factory=list)
    state: _ResultState = Field(default=_ResultState.RETAIN)

    def timestamp_utc(self):
        return datetime.datetime.fromisoformat(self.timestamp_utc_iso8601)


class _DataLedger(BaseModel):
    image_metadata_list: list[_ImageMetadata] = Field(default_factory=list)
    result_metadata_list: list[_ResultMetadata] = Field(default_factory=list)


class AbstractCalibrator(abc.ABC):

    _data_path: str

    _DATA_LEDGER_FILENAME: Final[str] = "data_ledger.json"
    _data_ledger: _DataLedger
    _data_ledger_filepath: str

    def __init__(
        self,
        configuration: _Configuration
    ):
        self._data_path = configuration.data_path
        if not self._exists_on_filesystem(path=self._data_path, pathtype="path", create_path=True):
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INITIALIZATION,
                public_message="Data path does not exist and could not be created.",
                private_message=f"{self._data_path} does not exist and could not be created.")

        self._data_ledger_filepath = os.path.join(self._data_path, AbstractCalibrator._DATA_LEDGER_FILENAME)
        self._load_data_ledger()

    def _add_image(
        self,
        image: numpy.ndarray,
        metadata: _ImageMetadata,
    ) -> None:
        """
        Helper for saving images consistently across different types of calibrators
        Returns true if successful, False otherwise.
        """
        # Before making any changes to the data ledger, make sure folders exist
        if not self._exists_on_filesystem(path=os.path.dirname(metadata.filepath), pathtype="path", create_path=True):
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message="Failed to create storage location for input image.")
        # Also make sure that this file does not somehow already exist (highly unlikely)
        if os.path.exists(metadata.filepath):
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message="Image appears to already exist. This is never expected to occur. "
                               "Please try again, and if this error continues to occur then please report a bug.",
                private_message=f"Image {metadata.filepath} appears to already exist. This is never expected to occur.")
        image_bytes: bytes
        image_bytes = ImageUtils.image_to_bytes(image_data=image, image_format=ImageFormat.FORMAT_PNG)
        try:
            with (open(metadata.filepath, 'wb') as in_file):
                in_file.write(image_bytes)
        except IOError as e:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message="Failed to save image - see local log for more details.",
                private_message=f"Failed to save image to {metadata.filepath}, reason: {str(e)}.")
        self._data_ledger.image_metadata_list.append(metadata)
        self._save_data_ledger()

    def _add_result(
        self,
        result: dict,
        metadata: _ResultMetadata
    ) -> None:
        self._save_dict_to_filepath(
            filepath=metadata.filepath,
            json_dict=result,
            ignore_none=True)
        self._data_ledger.result_metadata_list.append(metadata)
        self._save_data_ledger()

    @staticmethod
    def _delete_file_if_exists(filepath: str):
        try:
            os.remove(filepath)
        except FileNotFoundError:
            pass
        except OSError as e:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message=f"Failed to remove a file from the calibrator due to an unexpected reason. "
                               f"See local log for details.",
                private_message=f"Failed to delete file {filepath} for reason: {str(e)}.")

    # noinspection DuplicatedCode
    def delete_staged(self) -> None:
        image_indices_to_delete: list = list()
        image_metadata: _ImageMetadata
        for image_index, image_metadata in enumerate(self._data_ledger.image_metadata_list):
            if image_metadata.state == _ImageState.DELETE:
                self._delete_file_if_exists(image_metadata.filepath)
                image_indices_to_delete.append(image_index)
        for i in reversed(image_indices_to_delete):
            del self._data_ledger.image_metadata_list[i]
        result_indices_to_delete: list = list()
        result_metadata: _ResultMetadata
        for result_index, result_metadata in enumerate(self._data_ledger.result_metadata_list):
            if result_metadata.state == _ResultState.DELETE:
                self._delete_file_if_exists(result_metadata.filepath)
                result_indices_to_delete.append(result_index)
        for i in reversed(result_indices_to_delete):
            del self._data_ledger.result_metadata_list[i]
        self._save_data_ledger()

    @staticmethod
    def _exists_on_filesystem(
        path: str,
        pathtype: IOUtils.PathType,
        create_path: bool = False
    ) -> bool:
        errors: dict[str, str] = dict()
        return_value: bool = IOUtils.exists(
            path=path,
            pathtype=pathtype,
            create_path=create_path,
            on_error_for_user=lambda msg: errors.__setitem__(_PUBLIC_MESSAGE_KEY, msg),
            on_error_for_dev=lambda msg: errors.__setitem__(_PRIVATE_MESSAGE_KEY, msg))
        if len(errors) > 0:
            logger.error(errors[_PRIVATE_MESSAGE_KEY])
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message=f"Error determining if a file exists on the file system; See local log for details.",
                private_message=errors[_PRIVATE_MESSAGE_KEY])
        return return_value

    def get_image_by_identifier(
        self,
        identifier: str
    ) -> str:
        return self._load_image(identifier=identifier)

    # noinspection DuplicatedCode
    def _get_result_metadata_by_identifier(
        self,
        identifier: str
    ) -> _ResultMetadata:
        match_count: int = 0
        matching_result_metadata: _ResultMetadata | None = None
        for result_metadata in self._data_ledger.result_metadata_list:
            if result_metadata.identifier == identifier:
                match_count += 1
                matching_result_metadata = result_metadata
                break
        if match_count < 1:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.DATA_NOT_FOUND,
                public_message=f"Identifier {identifier} is not associated with any result.")
        elif match_count > 1:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message=f"Identifier {identifier} is associated with multiple results.")
        return matching_result_metadata

    def list_image_metadata(self) -> list[_ImageMetadata]:
        return list(self._data_ledger.image_metadata_list)

    def list_result_metadata(self) -> list[_ResultMetadata]:
        return list(self._data_ledger.result_metadata_list)

    def _load_data_ledger(self) -> None:
        """
        :return: True if loaded or if it can be created without overwriting existing data. False otherwise.
        """
        json_dict: dict
        json_dict = self._load_dict_from_filepath(filepath=self._data_ledger_filepath)
        try:
            self._data_ledger = _DataLedger(**json_dict)
        except ValidationError as e:
            logger.error(e)
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message=f"Error loading the data ledger; See local log for details.",
                private_message=str(e))

    @staticmethod
    def _load_dict_from_filepath(
        filepath: str
    ) -> dict:
        """
        :return: dict containing existing data (or empty if no data exists)
        """
        if not os.path.exists(filepath):
            return dict()  # Not considered an error, just doesn't exist yet
        elif not os.path.isfile(filepath):
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message="A file failed to load. See local log for details.",
                private_message=f"JSON at {filepath} exists but is not a file. "
                        "Most likely a directory exists at that location, "
                        "and it needs to be manually removed.")
        errors: dict[str, str] = dict()
        json_dict: dict = IOUtils.hjson_read(
            filepath=filepath,
            on_error_for_user=lambda msg: errors.__setitem__(_PUBLIC_MESSAGE_KEY, msg),
            on_error_for_dev=lambda msg: errors.__setitem__(_PRIVATE_MESSAGE_KEY, msg))
        if not json_dict:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message=f"Error loading data; See local log for details.",
                private_message=errors[_PRIVATE_MESSAGE_KEY])
        return json_dict

    # noinspection DuplicatedCode
    def _load_image(
        self,
        identifier: str
    ) -> str:  # image in base64
        match_count: int = 0
        matching_metadata: _ImageMetadata | None = None
        for image_metadata in self._data_ledger.image_metadata_list:
            if image_metadata.identifier == identifier:
                match_count += 1
                matching_metadata = image_metadata
                break
        if match_count < 1:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.DATA_NOT_FOUND,
                private_message=f"Identifier {identifier} is not associated with any image.")
        elif match_count > 1:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                private_message=f"Identifier {identifier} is associated with multiple images.")

        if not os.path.exists(matching_metadata.filepath):
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                private_message=f"File does not exist for image {identifier}.")
        image_bytes: bytes
        try:
            with (open(matching_metadata.filepath, 'rb') as in_file):
                image_bytes = in_file.read()
        except OSError:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                private_message=f"Failed to open image {identifier}.")
        image_base64 = ImageUtils.bytes_to_base64(image_bytes=image_bytes)
        return image_base64

    def _load_result(
        self,
        identifier: str,
        result_type: type[BaseModel]
    ) -> ...:
        metadata: _ResultMetadata = self._get_result_metadata_by_identifier(identifier=identifier)
        return self._load_result_by_metadata(metadata=metadata, result_type=result_type)

    def _load_result_by_metadata(
        self,
        metadata: _ResultMetadata,
        result_type: type[BaseModel]
    ) -> ...:
        """
        Read the calibration result corresponding to the provided metadata.
        """
        json_dict: dict
        load_success: bool
        json_dict = self._load_dict_from_filepath(metadata.filepath)
        result: result_type = result_type(**json_dict)
        return result

    def _save_data_ledger(self) -> None:
        return self._save_dict_to_filepath(
            filepath=self._data_ledger_filepath,
            json_dict=self._data_ledger.model_dump())

    @staticmethod
    def _save_dict_to_filepath(
        filepath: str,
        json_dict: dict,
        ignore_none: bool = False
    ) -> None:
        """
        :param filepath: Where to write the file
        :param json_dict: What to write to the file
        :param ignore_none: See IOUtils.json_write
        """
        errors: dict[str, str] = dict()
        IOUtils.json_write(
            filepath=filepath,
            json_dict=json_dict,
            on_error_for_user=lambda msg: errors.__setitem__(_PUBLIC_MESSAGE_KEY, msg),
            on_error_for_dev=lambda msg: errors.__setitem__(_PRIVATE_MESSAGE_KEY, msg),
            ignore_none=ignore_none)
        if len(errors) > 0:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message="Error saving data; See local log for more details.",
                private_message=errors[_PRIVATE_MESSAGE_KEY])

    # noinspection DuplicatedCode
    def update_image_metadata(
        self,
        image_identifier: str,
        image_state: _ImageState,
        image_label: str | None
    ) -> None:
        match_count: int = 0
        matched_metadata: _ImageMetadata | None = None
        for metadata in self._data_ledger.image_metadata_list:
            if metadata.identifier == image_identifier:
                match_count += 1
                matched_metadata = metadata
        if match_count < 1:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.DATA_NOT_FOUND,
                private_message=f"Identifier {image_identifier} is not associated with any image.")
        elif match_count > 1:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                private_message=f"Identifier {image_identifier} is associated with multiple images. "
                                "This suggests that the data ledger is in an inconsistent state. "
                                "It may be prudent to either manually correct it, or recreate it.")
        matched_metadata.state = image_state
        if image_label is not None:
            matched_metadata.label = image_label
        self._save_data_ledger()

    # noinspection DuplicatedCode
    def update_result_metadata(
        self,
        result_identifier: str,
        result_state: _ResultState,
        result_label: str | None = None
    ) -> None:
        match_count: int = 0
        matched_metadata: _ResultMetadata | None = None
        for metadata in self._data_ledger.result_metadata_list:
            if metadata.identifier == result_identifier:
                match_count += 1
                matched_metadata = metadata
        if match_count < 1:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.DATA_NOT_FOUND,
                private_message=f"Identifier {result_identifier} is not associated with any result.")
        elif match_count > 1:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                private_message=f"Identifier {result_identifier} is associated with multiple results. "
                                "This suggests that the data ledger is in an inconsistent state. "
                                "Please manually correct it, or recreate it.")
        matched_metadata.state = result_state
        if result_label is not None:
            matched_metadata.label = result_label
        self._save_data_ledger()


# =====================================================================================================================
# Intrinsic calibration
# =====================================================================================================================


class IntrinsicCalibration(BaseModel):
    timestamp_utc: str = Field()
    image_resolution: ImageResolution = Field()
    calibrated_values: IntrinsicParameters = Field()
    supplemental_data: dict = Field()


class IntrinsicCalibrator(AbstractCalibrator, abc.ABC):
    Configuration: type[_Configuration] = _Configuration
    ImageState: type[_ImageState] = _ImageState
    ImageMetadata: type[_ImageMetadata] = _ImageMetadata
    ResultState: type[_ResultState] = _ResultState
    ResultMetadata: type[_ResultMetadata] = _ResultMetadata

    def __init__(
        self,
        configuration: Configuration,
    ):
        super().__init__(configuration=configuration)

    # noinspection DuplicatedCode
    def add_image(
        self,
        image_base64: str,
        detector_label: str = "",
        timestamp_utc_iso8601: str | None = None
    ) -> str:  # id of image
        if timestamp_utc_iso8601 is None:
            timestamp_utc_iso8601 = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        image: numpy.ndarray = ImageUtils.base64_to_image(input_base64=image_base64, color_mode="color")
        identifier: str = str(uuid.uuid4())
        resolution: ImageResolution = ImageResolution(x_px=image.shape[1], y_px=image.shape[0])
        filepath = os.path.join(self._data_path, str(resolution), identifier + ImageFormat.FORMAT_PNG)
        metadata: _ImageMetadata = IntrinsicCalibrator.ImageMetadata(
            identifier=identifier,
            filepath=filepath,
            detector_label=detector_label,
            resolution=resolution,
            timestamp_utc_iso8601=timestamp_utc_iso8601)
        self._add_image(
            image=image,
            metadata=metadata)
        return metadata.identifier

    def calculate(
        self,
        image_resolution: ImageResolution
    ) -> tuple[str, IntrinsicCalibration]:
        """
        :returns: a tuple containing a result identifier (GUID as string) and the IntrinsicCalibration structure
        """

        image_metadata_list: list[_ImageMetadata] = list()  # image metadata available for calibration
        for image_index, image_metadata in enumerate(self._data_ledger.image_metadata_list):
            if image_metadata.resolution != image_resolution:
                continue
            if image_metadata.state != _ImageState.SELECT:
                continue
            if not self._exists_on_filesystem(path=image_metadata.filepath, pathtype="filepath"):
                raise MCTCalibrationError(
                    reason=CalibrationErrorReason.INVALID_STATE,
                    public_message="An image failed to load. "
                                   "suggesting that the data ledger is in an inconsistent state. "
                                   "Please see the locaL log for details.",
                    private_message=f"Image {image_metadata.identifier} was not found. "
                                    "This suggests that the data ledger is in an inconsistent state. "
                                    "Please correct the data ledger.")
            image_metadata_list.append(image_metadata)

        if len(image_metadata_list) == 0:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.COMPUTATION_FAILURE,
                public_message=f"No images found for resolution {str(image_resolution)}.")

        intrinsic_calibration, image_metadata_list = self._calculate_implementation(
            image_resolution=image_resolution,
            image_metadata_list=image_metadata_list)

        result_identifier: str = str(uuid.uuid4())
        result_filepath = \
            os.path.join(self._data_path, str(image_resolution), result_identifier + _RESULT_FORMAT)
        result_metadata: IntrinsicCalibrator.ResultMetadata = IntrinsicCalibrator.ResultMetadata(
            identifier=result_identifier,
            filepath=result_filepath,
            resolution=image_resolution,
            image_identifiers=[image_metadata.identifier for image_metadata in image_metadata_list])
        self._add_result(
            result=intrinsic_calibration.model_dump(),
            metadata=result_metadata)

        # For now, assume that the user's intent is to set any new calibration to be the active one
        self.update_result_metadata(
            result_identifier=result_metadata.identifier,
            result_state=_ResultState.ACTIVE)

        return result_identifier, intrinsic_calibration

    @abc.abstractmethod
    def _calculate_implementation(
        self,
        image_resolution: ImageResolution,
        image_metadata_list: list[ImageMetadata]
    ) -> tuple[IntrinsicCalibration, list[ImageMetadata]]:  # metadata of images that were actually used in calibration
        pass

    def get_result(
        self,
        result_identifier: str
    ) -> IntrinsicCalibration:
        return self._load_result(
            identifier=result_identifier,
            result_type=IntrinsicCalibration)

    def get_result_active_by_image_resolution(
        self,
        image_resolution: ImageResolution,
    ) -> Optional[...]:
        match_count: int = 0
        matched_metadata: IntrinsicCalibrator.ResultMetadata | None = None
        for result_metadata in self._data_ledger.result_metadata_list:
            if result_metadata.state == _ResultState.ACTIVE and result_metadata.resolution == image_resolution:
                matched_metadata = result_metadata
                match_count += 1

        if match_count < 1:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.DATA_NOT_FOUND,
                public_message=f"No result metadata is active for resolution {str(image_resolution)}. "
                               "Please ensure one has been selected, then try again.")
        if match_count > 1:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message=f"Multiple result metadata are active for resolution {str(image_resolution)}. "
                               "To recover from this ambiguous state, explicitly set "
                               "one of the results as \"active\", which will reset others to \"retain\".")

        return self._load_result_by_metadata(
            metadata=matched_metadata,
            result_type=IntrinsicCalibration)

    def list_resolutions(self) -> list[ImageResolution]:
        resolutions_as_str: set[str] = {str(metadata.resolution) for metadata in self._data_ledger.image_metadata_list}
        resolutions: list[ImageResolution] = [ImageResolution.from_str(resolution) for resolution in resolutions_as_str]
        return resolutions

    def list_image_metadata_by_image_resolution(
        self,
        image_resolution: ImageResolution
    ) -> list[ImageMetadata]:
        image_metadata_list: list[IntrinsicCalibrator.ImageMetadata] = list()
        for metadata in self._data_ledger.image_metadata_list:
            if metadata.resolution == image_resolution:
                image_metadata_list.append(metadata)
        return image_metadata_list

    def list_result_metadata_by_image_resolution(
        self,
        image_resolution: ImageResolution
    ) -> list[ResultMetadata]:
        result_metadata_list: list[IntrinsicCalibrator.ResultMetadata] = list()
        for metadata in self._data_ledger.result_metadata_list:
            if metadata.resolution == image_resolution:
                result_metadata_list.append(metadata)
        return result_metadata_list

    def update_result_metadata(
        self,
        result_identifier: str,
        result_state: ResultState,
        result_label: str | None = None
    ) -> None:
        super().update_result_metadata(
            result_identifier=result_identifier,
            result_state=result_state,
            result_label=result_label)

        # Some cleanup as applicable
        if result_state == _ResultState.ACTIVE:
            matching_metadata: _ResultMetadata = self._get_result_metadata_by_identifier(identifier=result_identifier)
            for metadata in self._data_ledger.result_metadata_list:
                if metadata.resolution == matching_metadata.resolution and \
                   metadata.identifier != matching_metadata.identifier:
                        metadata.state = _ResultState.RETAIN  # Only one ACTIVE per resolution
            self._save_data_ledger()


# =====================================================================================================================
# Extrinsic calibration
# =====================================================================================================================


class ExtrinsicCalibrationDetectorResult(BaseModel):
    detector_label: str = Field()
    detector_to_reference: Matrix4x4 = Field()


class ExtrinsicCalibration(BaseModel):
    timestamp_utc: str = Field()
    calibrated_values: list[ExtrinsicCalibrationDetectorResult] = Field()
    supplemental_data: dict = Field()


class ExtrinsicCalibrator(AbstractCalibrator, abc.ABC):
    Configuration: type[_Configuration] = _Configuration
    ImageState: type[_ImageState] = _ImageState
    ImageMetadata: type[_ImageMetadata] = _ImageMetadata
    ResultState: type[_ResultState] = _ResultState
    ResultMetadata: type[_ResultMetadata] = _ResultMetadata

    detector_intrinsics_by_label: dict[str, IntrinsicParameters]

    def __init__(
        self,
        configuration: Configuration | dict
    ):
        if isinstance(configuration, dict):
            configuration = ExtrinsicCalibrator.Configuration(**configuration)
        self.detector_intrinsics_by_label = dict()
        super().__init__(configuration=configuration)

    # noinspection DuplicatedCode
    def add_image(
        self,
        image_base64: str,
        detector_label: str,
        timestamp_utc_iso8601: str
    ) -> str:  # id of image
        image: numpy.ndarray = ImageUtils.base64_to_image(input_base64=image_base64, color_mode="color")
        identifier: str = str(uuid.uuid4())
        resolution: ImageResolution = ImageResolution(x_px=image.shape[1], y_px=image.shape[0])
        filepath = os.path.join(self._data_path, str(resolution), identifier + ImageFormat.FORMAT_PNG)
        metadata: _ImageMetadata = ExtrinsicCalibrator.ImageMetadata(
            identifier=identifier,
            filepath=filepath,
            detector_label=detector_label,
            resolution=resolution,
            timestamp_utc_iso8601=timestamp_utc_iso8601)
        self._add_image(
            image=image,
            metadata=metadata)
        return metadata.identifier

    def calculate(
        self
    ) -> tuple[str, ExtrinsicCalibration]:
        """
        :returns: a tuple containing a result identifier (GUID as string) and the ExtrinsicCalibration structure
        """

        # if detector_labels != list(set(detector_labels)):
        #     raise MCTIntrinsicCalibrationError(message=f"Detector labels must not contain duplicated elements.")
        # if len(detector_labels) != len(detector_intrinsics):
        #     raise MCTIntrinsicCalibrationError(message=f"Expected detector labels and intrinsics to be of same size.")
        # detector_intrinsics_by_label: dict[str, IntrinsicParameters] = dict(zip(detector_labels, detector_intrinsics))

        image_metadata_list: list[_ImageMetadata] = list()  # image metadata available for calibration
        for image_index, image_metadata in enumerate(self._data_ledger.image_metadata_list):
            if image_metadata.state != _ImageState.SELECT:
                continue
            if not self._exists_on_filesystem(path=image_metadata.filepath, pathtype="filepath"):
                raise MCTCalibrationError(
                    reason=CalibrationErrorReason.INVALID_STATE,
                    public_message="An image failed to load. "
                                   "suggesting that the data ledger is in an inconsistent state. "
                                   "Please see the locaL log for details.",
                    private_message=f"Image {image_metadata.identifier} was not found. "
                                    "This suggests that the data ledger is in an inconsistent state. "
                                    "Please correct the data ledger.")
            image_metadata_list.append(image_metadata)

        # This is a check to make sure that there are no duplicates over any (timestamp, detector_label)
        identifiers: list[tuple[str, str]] = [
            (metadata.timestamp_utc_iso8601, metadata.detector_label)
            for metadata in image_metadata_list]
        if len(identifiers) != len(set(identifiers)):
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message="Duplicate data were detected over (timestamp, detector_label).")

        if len(image_metadata_list) == 0:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.COMPUTATION_FAILURE,
                public_message=f"No images found for calibration.")

        extrinsic_calibration, image_metadata_list = self._calculate_implementation(
            image_metadata_list=image_metadata_list)

        result_identifier: str = str(uuid.uuid4())
        result_filepath = os.path.join(self._data_path, result_identifier + _RESULT_FORMAT)
        result_metadata: ExtrinsicCalibrator.ResultMetadata = ExtrinsicCalibrator.ResultMetadata(
            identifier=result_identifier,
            filepath=result_filepath,
            image_identifiers=[image_metadata.identifier for image_metadata in image_metadata_list])
        self._add_result(
            result=extrinsic_calibration.model_dump(),
            metadata=result_metadata)

        # For now, assume that the user's intent is to set any new calibration to be the active one
        self.update_result_metadata(
            result_identifier=result_metadata.identifier,
            result_state=_ResultState.ACTIVE)

        return result_identifier, extrinsic_calibration

    def intrinsic_parameters_update(
        self,
        detector_label: str,
        intrinsic_parameters: IntrinsicParameters
    ) -> None:
        self.detector_intrinsics_by_label[detector_label] = intrinsic_parameters

    @abc.abstractmethod
    def _calculate_implementation(
        self,
        image_metadata_list: list[ImageMetadata]
    ):
        pass

    def get_result(
        self,
        result_identifier: str
    ) -> ExtrinsicCalibration:
        return self._load_result(
            identifier=result_identifier,
            result_type=ExtrinsicCalibration)

    def get_result_active(
        self
    ) -> Optional[ExtrinsicCalibration]:
        match_count: int = 0
        matched_metadata: IntrinsicCalibrator.ResultMetadata | None = None
        for result_metadata in self._data_ledger.result_metadata_list:
            if result_metadata.state == _ResultState.ACTIVE:
                matched_metadata = result_metadata
                match_count += 1

        if match_count < 1:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.DATA_NOT_FOUND,
                public_message=f"No result metadata is active. "
                               "Please ensure one has been selected, then try again.")
        if match_count > 1:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.INVALID_STATE,
                public_message=f"Multiple result metadata are active. "
                               "To recover from this ambiguous state, explicitly set "
                               "one of the results as \"active\", which will reset others to \"retain\".")

        return self._load_result_by_metadata(
            metadata=matched_metadata,
            result_type=IntrinsicCalibration)
