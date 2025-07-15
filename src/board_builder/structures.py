from src.common.util import MathUtils
from src.common.structures import Matrix4x4, Pose
import abc
import datetime
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
from scipy.spatial.transform import Rotation as R


# TODO: Merge into a similar structure in common
class MarkerCorners:
    detector_label: str
    marker_id: int
    points: list[list[float]]
    timestamp: datetime.datetime

    def __init__(
        self,
        detector_label: str,
        marker_id: int,
        points: list[list[float]],
        timestamp: datetime.datetime
    ):
        self.detector_label = detector_label
        self.marker_id = marker_id
        self.points = points
        self.timestamp = timestamp


class MarkerRaySet(BaseModel):
    marker_id: int = Field()
    image_points: list[list[float]] = Field()  # image positions of marker corners. Size 4.
    image_timestamp: datetime.datetime = Field()
    ray_origin_reference: list[float] = Field()  # Shared origin for all rays (same detector)
    ray_directions_reference: list[list[float]] = Field()  # Size 4 (one for each image point)
    detector_label: str = Field()
    detector_to_reference_matrix: Matrix4x4 = Field()

    @staticmethod
    def age_seconds(
        marker_ray_set,
        query_timestamp: datetime.datetime
    ):
        return (query_timestamp - marker_ray_set.image_timestamp).total_seconds()

    @staticmethod
    def newest_timestamp_in_list(marker_ray_set_list: list) -> datetime.datetime:
        return_value = datetime.datetime.now()
        for ray_set in marker_ray_set_list:
            if ray_set.image_timestamp > return_value:
                return_value = ray_set.image_timestamp
        return return_value

    @staticmethod
    def oldest_timestamp_in_list(marker_ray_set_list: list) -> datetime.datetime:
        return_value = datetime.datetime.utcfromtimestamp(0)
        for ray_set in marker_ray_set_list:
            if ray_set.image_timestamp > return_value:
                return_value = ray_set.image_timestamp
        return return_value


class MatrixNode:
    def __init__(self, node_id: str):
        self.id = node_id
        self.neighbours = []
        self.weights = {}

    def add_neighbour(self, neighbour_node, weight: int):
        self.neighbours.append(neighbour_node)
        self.weights[neighbour_node.id] = weight


# TODO: Merge/replace this with pose under common data structures
class PoseData(BaseModel):
    target_id: str = Field()
    object_to_reference_matrix: Matrix4x4 = Field()
    ray_sets: list[MarkerRaySet]

    def newest_timestamp(self) -> datetime.datetime:
        return MarkerRaySet.newest_timestamp_in_list(self.ray_sets)

    def oldest_timestamp(self) -> datetime.datetime:
        return MarkerRaySet.oldest_timestamp_in_list(self.ray_sets)

    @staticmethod
    def age_seconds(
        pose,
        query_timestamp: datetime.datetime
    ) -> float:
        return (query_timestamp - pose.oldest_timestamp()).total_seconds()


class PoseLocation:

    _id: str
    _timestamp: str
    _TMatrix: np.ndarray
    _RMAT_list: list
    _TVEC_list: list

    def __init__(self, object_id):
        self._id = object_id
        self._timestamp = str(datetime.datetime.now())

        self._TMatrix = np.eye(4)
        self._RMAT_list = []  # Rotation matrix
        self._TVEC_list = []  # Translation vector

        self.frame_count = 0

    def add_matrix(self, transformation_matrix: Matrix4x4, timestamp: str):
        self._timestamp = timestamp

        self._RMAT_list.append(transformation_matrix[:3, :3])
        self._TVEC_list.append(transformation_matrix[:3, 3])

        avg_translation = np.mean(self._TVEC_list, axis=0)

        quaternions = [R.from_matrix(rot).as_quat(canonical=True) for rot in self._RMAT_list]
        quaternions = [[float(quaternion[i]) for i in range(0, 4)] for quaternion in quaternions]
        avg_quat = MathUtils.average_quaternion(quaternions)
        avg_rotation = R.from_quat(avg_quat).as_matrix()

        self._TMatrix[:3, :3] = avg_rotation
        self._TMatrix[:3, 3] = avg_translation

    def get_matrix(self):
        return self._TMatrix

    def get_average_pose(self):
        pose = Pose(
            target_id=self._id,
            object_to_reference_matrix=Matrix4x4.from_numpy_array(self._TMatrix),
            solver_timestamp_utc_iso8601=self._timestamp
        )
        return pose

    def get_median_pose(self):
        if not self._RMAT_list or not self._TVEC_list:
            raise ValueError("No matrices available to compute the median.")

        rmat_array = np.array(self._RMAT_list)
        tvec_array = np.array(self._TVEC_list)

        median_rmat = np.median(rmat_array, axis=0)
        median_tvec = np.median(tvec_array, axis=0)

        median_transformation_matrix = np.eye(4)
        median_transformation_matrix[:3, :3] = median_rmat
        median_transformation_matrix[:3, 3] = median_tvec

        pose = Pose(
            target_id=self._id,
            object_to_reference_matrix=Matrix4x4.from_numpy_array(median_transformation_matrix),
            solver_timestamp_utc_iso8601=self._timestamp
        )

        return pose


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

