import numpy
from pydantic import BaseModel, Field


class _Marker(BaseModel):
    marker_id: int = Field()
    marker_size: float | None = Field(default=None)
    points: list[list[float]] | None = Field(default=None)

    # TODO: During validation, make sure either marker_size or points is defined, but not both.

    def get_marker_size(self) -> float:
        if self.marker_size is None:
            if self.points is None or len(self.points) < 2:
                raise RuntimeError("TargetMarker defined with neither marker_size nor enough points.")
            marker_size_sum: float = 0.0
            for point_index in range(0, len(self.points)):
                point_a: numpy.ndarray = numpy.asarray(self.points[point_index])
                point_b: numpy.ndarray = numpy.asarray(self.points[point_index-1])
                vector: numpy.ndarray = point_a - point_b
                marker_size_sum += numpy.linalg.norm(vector)
            self.marker_size = marker_size_sum / len(self.points)
        return self.marker_size

    def get_points(self) -> list[list[float]]:
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


class TargetBase(BaseModel):
    target_id: str = Field()


class TargetMarker(TargetBase, _Marker):
    pass  # Contains target_id as well as marker fields


class TargetBoard(TargetBase):
    markers: list[_Marker] = Field()
