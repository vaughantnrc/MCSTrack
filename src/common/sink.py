from .math import Pose
from .mct_component import DetectorFrame, MixerFrame
from .status import MCTError
import abc
import datetime
import io
import os
from pydantic import BaseModel, Field
from typing import Any


class SinkConfiguration(BaseModel):
    implementation: str = Field()
    configuration: dict[str, Any] = Field()


class SinkException(MCTError):
    message: str

    def __init__(self, message: str, *args, **kwargs):
        super().__init__(args, kwargs)
        self.message = message


class BaseSink(abc.ABC):

    @abc.abstractmethod
    def __init__(
        self,
        **kwargs
    ):
        """
        :param configuration: dict
        """
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def open(self):
        pass

    @abc.abstractmethod
    def handle_detector_frame(
        self,
        frame: DetectorFrame
    ) -> None:
        pass

    @abc.abstractmethod
    def handle_mixer_frame(
        self,
        frame: MixerFrame
    ) -> None:
        pass

    @staticmethod
    @abc.abstractmethod
    def implementation_str():
        pass


class CSVPoseSinkConfiguration(BaseModel):
    prepend_datetime: bool = Field(default=True)
    path: str = Field(default_factory=str)
    filename: str = Field()


class CSVPoseSink(BaseSink):

    _configuration: CSVPoseSinkConfiguration
    _file: io.TextIOBase | None

    def __init__(
        self,
        configuration: dict
    ):
        self._configuration = CSVPoseSinkConfiguration(**configuration)
        self._file = None

    def open(self):
        if self._file is not None:
            self._file.close()
        filename: str = f"{self._configuration.filename}"
        if self._configuration.prepend_datetime:
            datetime_str: str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{datetime_str}_{filename}"
        filepath: str = os.path.join(self._configuration.path, filename)
        _file = open(filepath, 'w')

    def close(self):
        if self._file is not None:
            self._file.close()

    def handle_detector_frame(
        self,
        frame: DetectorFrame
    ) -> None:
        pass

    def handle_mixer_frame(
        self,
        frame: MixerFrame
    ) -> None:
        header_line: str = "frame_timestamp,target_id,target_timestamp,TX,TY,TZ,R11,R12,R13,R21,R22,R23,31,R32,R33"
        output_lines: list[str] = [header_line]
        frame_timestamp_utc_iso8601 = frame.timestamp_utc_iso8601
        poses: list[Pose] = frame.detector_poses + frame.target_poses
        for pose in poses:
            target_timestamp_utc_iso8601: str = pose.solver_timestamp_utc_iso8601
            translation: list[float] = pose.object_to_reference_matrix.get_translation()
            rotation_matrix: list[list[float]] = pose.object_to_reference_matrix.get_rotation_as_matrix()
            output_lines.append(
                f"{frame_timestamp_utc_iso8601},{pose.target_id},{target_timestamp_utc_iso8601},"
                f"{translation[0]},{translation[1]},{translation[2]},"
                f"{rotation_matrix[0][0]},{rotation_matrix[0][1]},{rotation_matrix[0][2]},"
                f"{rotation_matrix[1][0]},{rotation_matrix[1][1]},{rotation_matrix[1][2]},"
                f"{rotation_matrix[2][0]},{rotation_matrix[2][1]},{rotation_matrix[2][2]}")
        self._file.writelines(output_lines)

    @staticmethod
    def implementation_str():
        return "pose_csv"
