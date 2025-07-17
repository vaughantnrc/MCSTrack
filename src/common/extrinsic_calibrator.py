from src.common import \
    StatusMessageSource
import abc
import datetime
from enum import StrEnum
import logging
from pydantic import BaseModel, Field
from typing import Final


logger = logging.getLogger(__name__)


class _ImageState(StrEnum):
    IGNORE: Final[int] = "ignore"
    SELECT: Final[int] = "select"
    DELETE: Final[int] = "delete"  # stage for deletion


class _ImageMetadata(BaseModel):
    identifier: str = Field()
    label: str = Field(default_factory=str)  # human-readable label
    timestamp_utc: str = Field(default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc).isoformat())
    state: _ImageState = Field(default=_ImageState.SELECT)


class ExtrinsicCalibrator(abc.ABC):

    _image_filepaths: dict[tuple[str, str], str]  # (detector_id, frame_id) -> image_filepath
    _status_message_source: StatusMessageSource

    DATA_FILENAME: Final[str] = "extrinsic_calibration_data.json"

    # data:
    #   per detector:
    #     initial_frame transform to reference_target
    #     final transform to reference_target
    #     per frame:
    #       image
    #       (marker_id,2d_points)s
    #   final (frame_id,marker_id,3d_points)s
    #
    # input data:
    #   per detector:
    #     per frame:
    #       PNG: image
    #
    # output data:
    #   per detector:
    #     JSON: transform to reference_target
    #     JSON: Additional stats, inc. reference_target definition
