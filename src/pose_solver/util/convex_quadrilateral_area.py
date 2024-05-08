import numpy
from typing import Final

EPSILON: Final[float] = 0.0001


# General approach:
# Given points a, b, c, and d shown below,
# and calculating points e and f shown below,
# add areas defined by right triangles bea, ceb, dfc, and afd
# b..................c
# . ..              ...
#  .  ...        ...  .
#  .     ..   .f.      .
#   .      .e.  ...    .
#   .   ...        ..   .
#    ...             ... .
#    a...................d
def convex_quadrilateral_area(
    points: list[list[float]]  # 2D points in clockwise order
) -> float:
    point_a = numpy.array(points[0], dtype="float32")
    point_b = numpy.array(points[1], dtype="float32")
    point_c = numpy.array(points[2], dtype="float32")
    point_d = numpy.array(points[3], dtype="float32")

    vector_ac = point_c - point_a
    vector_ac_norm = numpy.linalg.norm(vector_ac)
    vector_bd = point_d - point_b
    vector_bd_norm = numpy.linalg.norm(vector_bd)
    if vector_ac_norm <= EPSILON or vector_bd_norm <= EPSILON:
        return 0.0
    width_vector = vector_ac / numpy.linalg.norm(vector_ac)
    height_vector = numpy.array([width_vector[1], -width_vector[0]], dtype="float32")  # rotated 90 degrees

    sum_of_areas: float = 0.0
    point_pairs: list[tuple[numpy.ndarray, numpy.ndarray]] = [
        (point_a, point_b),
        (point_b, point_c),
        (point_c, point_d),
        (point_d, point_a)]
    for point_pair in point_pairs:
        line_vector = point_pair[1] - point_pair[0]
        width = numpy.dot(line_vector, width_vector)
        height = numpy.dot(line_vector, height_vector)
        sum_of_areas += numpy.abs(width * height / 2.0)

    return sum_of_areas


def test():
    points = [
        [1.0, 3.0],
        [2.0, 5.0],
        [5.0, 3.0],
        [3.0, 2.0]]
    area = 6.0
    assert abs(convex_quadrilateral_area(points) - area) <= EPSILON
    assert abs(convex_quadrilateral_area([points[3]] + points[0:3]) - area) <= EPSILON
    print("Success")


if __name__ == "__main__":
    test()
