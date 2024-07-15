from .calibration_map_entry import CalibrationMapEntry
from .calibration_map_value import CalibrationMapValue
from src.common.structures.detector_resolution import DetectorResolution
from pydantic import BaseModel, Field


class CalibrationMap(BaseModel):
    entries: list[CalibrationMapEntry] = Field(default_factory=list)

    def as_dict(self) -> dict[DetectorResolution, CalibrationMapValue]:
        return_value: dict[DetectorResolution, CalibrationMapValue] = dict()
        for entry in self.entries:
            if entry.key not in return_value:
                return_value[entry.key] = CalibrationMapValue()
            for image_metadata in entry.value.image_metadata_list:
                return_value[entry.key].image_metadata_list.append(image_metadata)
            for result_metadata in entry.value.result_metadata_list:
                return_value[entry.key].result_metadata_list.append(result_metadata)
        return return_value

    @staticmethod
    def from_dict(in_dict: dict[DetectorResolution, CalibrationMapValue]):
        entries: list[CalibrationMapEntry] = list()
        for key in in_dict.keys():
            entries.append(CalibrationMapEntry(key=key, value=in_dict[key]))
        return CalibrationMap(entries=entries)
