from src.common.structures import ImageResolution
import datetime
from enum import StrEnum
from pydantic import BaseModel, Field
from typing import Final


class CalibrationImageState(StrEnum):
    IGNORE: Final[int] = "ignore"
    SELECT: Final[int] = "select"
    DELETE: Final[int] = "delete"  # stage for deletion


class CalibrationImageMetadata(BaseModel):
    identifier: str = Field()
    label: str = Field(default_factory=str)
    timestamp_utc: str = Field(default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc).isoformat())
    state: CalibrationImageState = Field(default=CalibrationImageState.SELECT)


class CalibrationResultState(StrEnum):
    # indicate to use this calibration (as opposed to simply storing it)
    # normally there shall only ever be one ACTIVE calibration for a given image resolution
    ACTIVE: Final[str] = "active"

    # store the calibration, but don't mark it for use
    RETAIN: Final[str] = "retain"

    # stage for deletion
    DELETE: Final[str] = "delete"


class CalibrationResultMetadata(BaseModel):
    identifier: str = Field()
    label: str = Field(default_factory=str)
    timestamp_utc_iso8601: str = Field(default_factory=lambda: datetime.datetime.now(tz=datetime.timezone.utc).isoformat())
    image_identifiers: list[str] = Field(default_factory=list)
    state: CalibrationResultState = Field(default=CalibrationResultState.RETAIN)

    def timestamp_utc(self):
        return datetime.datetime.fromisoformat(self.timestamp_utc_iso8601)


class CalibrationMapValue(BaseModel):
    image_metadata_list: list[CalibrationImageMetadata] = Field(default_factory=list)
    result_metadata_list: list[CalibrationResultMetadata] = Field(default_factory=list)


class CalibrationMapEntry(BaseModel):
    key: ImageResolution = Field()
    value: CalibrationMapValue = Field()


class CalibrationMap(BaseModel):
    entries: list[CalibrationMapEntry] = Field(default_factory=list)

    def as_dict(self) -> dict[ImageResolution, CalibrationMapValue]:
        return_value: dict[ImageResolution, CalibrationMapValue] = dict()
        for entry in self.entries:
            if entry.key not in return_value:
                return_value[entry.key] = CalibrationMapValue()
            for image_metadata in entry.value.image_metadata_list:
                return_value[entry.key].image_metadata_list.append(image_metadata)
            for result_metadata in entry.value.result_metadata_list:
                return_value[entry.key].result_metadata_list.append(result_metadata)
        return return_value

    @staticmethod
    def from_dict(in_dict: dict[ImageResolution, CalibrationMapValue]):
        entries: list[CalibrationMapEntry] = list()
        for key in in_dict.keys():
            entries.append(CalibrationMapEntry(key=key, value=in_dict[key]))
        return CalibrationMap(entries=entries)
