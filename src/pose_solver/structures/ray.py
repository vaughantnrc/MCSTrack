import numpy
from typing import Final, List


EPSILON: Final[float] = 0.0001


class Ray:
    source_point: List[float]
    direction: List[float]

    def __init__(
        self,
        source_point: List[float],
        direction: List[float]
    ):
        direction_norm = numpy.linalg.norm(direction)
        if direction_norm < EPSILON:
            raise ValueError("Direction cannot be zero.")
        self.source_point = source_point
        self.direction = direction
