from .image import Annotation, ImageResolution
from .linear_algebra import Pose
import abc
import datetime
import numpy
from pydantic import BaseModel, Field, PrivateAttr


class DetectorFrame(BaseModel):
    annotations: list[Annotation] = Field(default_factory=list)
    timestamp_utc_iso8601: str = Field()
    image_resolution: ImageResolution = Field()

    @property
    def annotations_identified(self):
        return [annotation for annotation in self.annotations if annotation.label != Annotation.UNIDENTIFIED_LABEL]

    @property
    def annotations_unidentified(self):
        return [annotation for annotation in self.annotations if annotation.label == Annotation.UNIDENTIFIED_LABEL]

    @property
    def timestamp_utc(self):
        return datetime.datetime.fromisoformat(self.timestamp_utc_iso8601)


class PoseSolverFrame(BaseModel):
    detector_poses: list[Pose] | None = Field()
    target_poses: list[Pose] | None = Field()
    timestamp_utc_iso8601: str = Field()

    def timestamp_utc(self):
        return datetime.datetime.fromisoformat(self.timestamp_utc_iso8601)


# --------------------------------------------------------------------------------
#  Everything below should be assessed for either migration or deletion
# --------------------------------------------------------------------------------


class Marker(BaseModel):
    marker_id: str = Field()
    marker_size: float | None = Field(default=None)
    points: list[list[float]] | None = Field(default=None)

    def get_points_internal(self) -> list[list[float]]:
        # Use the TargetBase.get_points() instead.
        if self.points is None:
            if self.marker_size is None:
                raise RuntimeError("TargetMarker defined with neither marker_size nor points.")
            half_width = self.marker_size / 2.0
            self.points = [
                [-half_width, half_width, 0.0],
                [half_width, half_width, 0.0],
                [half_width, -half_width, 0.0],
                [-half_width, -half_width, 0.0]]
        return self.points


class TargetBase(BaseModel, abc.ABC):
    label: str = Field()

    @abc.abstractmethod
    def get_marker_ids(self) -> list[str]: ...

    @abc.abstractmethod
    def get_points_for_marker_id(self, marker_id: str) -> list[list[float]]: ...

    @abc.abstractmethod
    def get_points(self) -> list[list[float]]: ...


class TargetMarker(TargetBase, Marker):
    def get_marker_ids(self) -> list[str]:
        return [self.marker_id]

    def get_points(self) -> list[list[float]]:
        return self.get_points_internal()

    def get_points_for_marker_id(self, marker_id: str) -> list[list[float]]:
        if marker_id != self.marker_id:
            raise IndexError(f"marker_id {marker_id} is not in target {self.label}")
        return self.get_points_internal()


class TargetBoard(TargetBase):
    markers: list[Marker] = Field()
    _marker_dict: None | dict[str, Marker] = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._marker_dict = None

    def get_marker_ids(self) -> list[str]:
        return [marker.marker_id for marker in self.markers]

    def get_points(self) -> list[list[float]]:
        points = list()
        for marker in self.markers:
            points += marker.get_points_internal()
        return points

    def get_points_for_marker_id(self, marker_id: str) -> list[list[float]]:
        if self._marker_dict is None:
            self._marker_dict = dict()
            for marker in self.markers:
                self._marker_dict[marker.marker_id] = marker
        if marker_id not in self._marker_dict:
            raise IndexError(f"marker_id {marker_id} is not in target {self.label}")
        return self._marker_dict[marker_id].points
