# Solution based on: Arun et al. Least square fitting of two 3D point sets (1987)
# https://stackoverflow.com/questions/66923224/rigid-registration-of-two-point-clouds-with-known-correspondence
# Use mirroring solution proposed by Oomori et al.
# Oomori et al. Point cloud matching using singular value decomposition. (2016)
import numpy


def register_corresponding_points(
    point_set_from: list[list[float]],
    point_set_to: list[list[float]]
) -> numpy.array:  # 4x4 transformation matrix, indexed by [row,col]
    assert len(point_set_from) == len(point_set_to)
    point_count = len(point_set_from)
    sums_from = numpy.array([0, 0, 0], dtype="float32")
    sums_to = numpy.array([0, 0, 0], dtype="float32")
    for point_index in range(0, point_count):
        sums_from += numpy.array(point_set_from[point_index])
        sums_to += numpy.array(point_set_to[point_index])
    centroid_from = (sums_from / point_count).reshape((3, 1))  # as column
    centroid_to = (sums_to / point_count).reshape((3, 1))  # as column
    points_from = numpy.array(point_set_from).transpose()  # as columns
    points_to = numpy.array(point_set_to).transpose()  # as columns
    # centroid_from = numpy.mean(points_from, axis=0)
    # centroid_to = numpy.mean(points_to, axis=0).reshape((3, 1))
    centered_points_from = points_from - centroid_from
    centered_points_to = points_to - centroid_to
    covariance = numpy.matmul(centered_points_from, centered_points_to.transpose())
    u, _, vh = numpy.linalg.svd(covariance)
    s = numpy.identity(3, dtype="float32")  # s will be the Oomori mirror fix
    s[2, 2] = numpy.linalg.det(numpy.matmul(u, vh))
    rotation = numpy.matmul(u, numpy.matmul(s, vh)).transpose()
    translation = centroid_to - numpy.matmul(rotation, centroid_from)
    matrix = numpy.identity(4, dtype="float32")
    matrix[0:3, 0:3] = rotation
    matrix[0:3, 3] = translation[0:3].reshape(3)
    return matrix
