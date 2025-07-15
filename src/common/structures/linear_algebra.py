import numpy
from pydantic import BaseModel, Field
from typing import ClassVar, Final


_DEFAULT_EPSILON: Final[float] = 0.0001


class IterativeClosestPointParameters(BaseModel):
    # ICP will stop after this many iterations
    termination_iteration_count: int = Field()

    # ICP will stop if distance *and* angle difference from one iteration to the next
    # is smaller than these
    termination_delta_translation: float = Field()
    termination_delta_rotation_radians: float = Field()

    # ICP will stop if overall point-to-point distance (between source and target)
    # mean *or* root-mean-square is less than specified
    termination_mean_point_distance: float = Field()
    termination_rms_point_distance: float = Field()  # root-mean-square


class Landmark(BaseModel):

    # These can denote that multiple landmarks are related if they share the same
    # "base label" (the part before the first and only occurrence of this character).
    RELATION_CHARACTER: ClassVar[str] = "$"

    """
    A distinct point in 3D space.
    Coordinates are in the unit of the user's choosing.
    """
    feature_label: str = Field()
    x: float = Field()
    y: float = Field()
    z: float = Field()

    def as_float_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def base_feature_label(self) -> str:
        """
        Part of the label before the RELATION_CHARACTER.
        """
        if self.RELATION_CHARACTER not in self.feature_label:
            return self.feature_label
        return self.feature_label[0:self.feature_label.index(self.RELATION_CHARACTER)]


class Matrix4x4(BaseModel):

    @staticmethod
    def _identity_values() -> list[float]:
        return \
            [1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0]
    values: list[float] = Field(default_factory=_identity_values)

    def as_numpy_array(self):
        a = self.values
        return numpy.asarray(
            [[a[0],  a[1],  a[2],  a[3]],
             [a[4],  a[5],  a[6],  a[7]],
             [a[8],  a[9],  a[10], a[11]],
             [a[12], a[13], a[14], a[15]]])

    def __getitem__(self, idx: tuple[int, int]) -> float:
        if isinstance(idx, tuple):
            return self.values[(idx[0]*4) + idx[1]]
        else:
            raise ValueError("Unexpected index. Expected tuple index [row,col].")

    def __mul__(self, other) -> 'Matrix4x4':
        if not isinstance(other, Matrix4x4):
            raise ValueError
        result_numpy_array = numpy.matmul(self.as_numpy_array(), other.as_numpy_array())
        return Matrix4x4(values=list(result_numpy_array.flatten()))

    def get_translation(self) -> list[float]:
        """
        Return a vector of [x,y,z] representing translation.
        """
        a = self.values
        return [a[3], a[7], a[11]]

    def inverse(self) -> 'Matrix4x4':
        inv_numpy_array = numpy.linalg.inv(self.as_numpy_array())
        return Matrix4x4.from_numpy_array(inv_numpy_array)

    @staticmethod
    def from_raw_values(
        v00, v01, v02, v03,
        v10, v11, v12, v13,
        v20, v21, v22, v23,
        v30, v31, v32, v33
    ) -> 'Matrix4x4':
        return Matrix4x4(values=[
            v00, v01, v02, v03,
            v10, v11, v12, v13,
            v20, v21, v22, v23,
            v30, v31, v32, v33])

    @staticmethod
    def from_list(
        value_list: list[float]
    ) -> 'Matrix4x4':
        if len(value_list) != 16:
            raise ValueError(f"Expected a list of 16 float. Got {str(value_list)}.")
        return Matrix4x4(values=list(value_list))

    @staticmethod
    def from_numpy_array(
        value_array: numpy.ndarray
    ) -> 'Matrix4x4':
        if len(value_array) != 4:
            raise ValueError(f"Expected input array to have 4 rows. Got {len(value_array)}.")
        for i in range(0, len(value_array)):
            if len(value_array[i]) != 4:
                raise ValueError(f"Expected input row {i} to have 4 col. Got {len(value_array[i])}.")
        return Matrix4x4(values=list(value_array.flatten()))


class Pose(BaseModel):
    target_id: str = Field()
    object_to_reference_matrix: Matrix4x4 = Field()
    solver_timestamp_utc_iso8601: str = Field()


class Ray:
    source_point: list[float]
    direction: list[float]

    def __init__(
        self,
        source_point: list[float],
        direction: list[float],
        epsilon: float = _DEFAULT_EPSILON
    ):
        direction_norm = numpy.linalg.norm(direction)
        if direction_norm < epsilon:
            raise ValueError("Direction cannot be zero.")
        self.source_point = source_point
        self.direction = direction


class Target(BaseModel):
    """
    A trackable object.
    """
    label: str
    landmarks: list[Landmark]

    def get_landmark_point(
        self,
        feature_label: str
    ) -> list[float]:
        for landmark in self.landmarks:
            if landmark.feature_label == feature_label:
                return landmark.as_float_list()
        raise ValueError
