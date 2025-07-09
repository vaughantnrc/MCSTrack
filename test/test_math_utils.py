from src.common import MathUtils
from src.common.structures import \
    IterativeClosestPointParameters, \
    Ray
import datetime
import numpy
from scipy.spatial.transform import Rotation
from typing import Final
from unittest import TestCase


EPSILON: Final[float] = 0.0001


class TestMathUtils(TestCase):

    def assertRotationCloseToIdentity(
        self,
        matrix: numpy.ndarray,
        tolerance: float = 0.1
    ) -> None:
        self.assertAlmostEqual(float(matrix[0, 0]), 1.0, delta=tolerance)
        self.assertAlmostEqual(float(matrix[0, 1]), 0.0, delta=tolerance)
        self.assertAlmostEqual(float(matrix[0, 2]), 0.0, delta=tolerance)
        self.assertAlmostEqual(float(matrix[1, 0]), 0.0, delta=tolerance)
        self.assertAlmostEqual(float(matrix[1, 1]), 1.0, delta=tolerance)
        self.assertAlmostEqual(float(matrix[1, 2]), 0.0, delta=tolerance)
        self.assertAlmostEqual(float(matrix[2, 0]), 0.0, delta=tolerance)
        self.assertAlmostEqual(float(matrix[2, 1]), 0.0, delta=tolerance)
        self.assertAlmostEqual(float(matrix[2, 2]), 1.0, delta=tolerance)

    def test_convex_quadrilateral_area(self):
        points = [
            [1.0, 3.0],
            [2.0, 5.0],
            [5.0, 3.0],
            [3.0, 2.0]]
        area = 6.0
        self.assertAlmostEqual(abs(MathUtils.convex_quadrilateral_area(points)), area, delta=EPSILON)
        self.assertAlmostEqual(abs(MathUtils.convex_quadrilateral_area([points[3]] + points[0:3])), area, delta=EPSILON)

    def test_iterative_closest_point(self) -> None:
        # Transformation is approximately
        source_known_points = [
            [2.0, 0.0, 2.0],
            [2.0, 2.0, 2.0],
            [2.0, 2.0, 0.0],
            [2.0, 0.0, 0.0]]
        source_ray_points = [
            [0.0, 2.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0]]
        target_known_points = [
            [1.0, 1.0, -2.0],
            [-1.0, 1.0, -2.0],
            [-1.0, -1.0, -2.0],
            [1.0, -1.0, -2.0]]
        origin = [0.0, 0.0, 1.0]
        sqrt3 = numpy.sqrt(3.0)
        target_rays = [
            Ray(source_point=origin, direction=[-sqrt3, sqrt3, -sqrt3]),
            Ray(source_point=origin, direction=[sqrt3, sqrt3, -sqrt3]),
            Ray(source_point=origin, direction=[sqrt3, -sqrt3, -sqrt3]),
            Ray(source_point=origin, direction=[-sqrt3, -sqrt3, -sqrt3])]
        begin_datetime = datetime.datetime.utcnow()
        icp_parameters = IterativeClosestPointParameters(
            termination_iteration_count=100,
            termination_delta_translation=0.001,
            termination_delta_rotation_radians=0.001,
            termination_mean_point_distance=0.0001,
            termination_rms_point_distance=0.0001)
        icp_output = MathUtils.iterative_closest_point_for_points_and_rays(
            source_known_points=source_known_points,
            target_known_points=target_known_points,
            source_ray_points=source_ray_points,
            target_rays=target_rays,
            parameters=icp_parameters)
        source_to_target_matrix = icp_output.source_to_target_matrix.as_numpy_array()
        end_datetime = datetime.datetime.utcnow()
        duration = (end_datetime - begin_datetime)
        duration_seconds = duration.seconds + (duration.microseconds / 1000000.0)
        message = str()
        for source_point in source_known_points:
            source_4d = [source_point[0], source_point[1], source_point[2], 1]
            target_4d = list(numpy.matmul(source_to_target_matrix, source_4d))
            target_point = [target_4d[0], target_4d[1], target_4d[2]]
            message = message + str(target_point) + "\n"
        message += "Algorithm took " + "{:.6f}".format(duration_seconds) + " seconds " + \
            "and took " + str(icp_output.iteration_count) + " iterations."
        print(message)

        # TODO: Comparisons, self.assertXXXXX()

    def test_register_corresponding_points_identity_3_points(self):
        point_set_from = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]]
        point_set_to = point_set_from
        matrix = MathUtils.register_corresponding_points(point_set_from, point_set_to)
        self.assertRotationCloseToIdentity(matrix)
        self.assertAlmostEqual(matrix[0, 3], 0.0)
        self.assertAlmostEqual(matrix[1, 3], 0.0)
        self.assertAlmostEqual(matrix[2, 3], 0.0)
        self.assertEqual(matrix[3, 3], 1.0)
        self.assertEqual(matrix[3, 0], 0.0)
        self.assertEqual(matrix[3, 1], 0.0)
        self.assertEqual(matrix[3, 2], 0.0)

    def test_register_corresponding_points_identity_4_points(self):
        point_set_from = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]]
        point_set_to = point_set_from
        matrix = MathUtils.register_corresponding_points(point_set_from, point_set_to)
        self.assertRotationCloseToIdentity(matrix)
        self.assertAlmostEqual(matrix[0, 3], 0.0)
        self.assertAlmostEqual(matrix[1, 3], 0.0)
        self.assertAlmostEqual(matrix[2, 3], 0.0)
        self.assertEqual(matrix[3, 3], 1.0)
        self.assertEqual(matrix[3, 0], 0.0)
        self.assertEqual(matrix[3, 1], 0.0)
        self.assertEqual(matrix[3, 2], 0.0)

    def test_register_corresponding_points_translation(self):
        point_set_from = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]]
        translation = [10, 0, 10]
        point_set_to = list(point_set_from + numpy.stack(translation))
        matrix = MathUtils.register_corresponding_points(point_set_from, point_set_to)
        self.assertRotationCloseToIdentity(matrix)
        self.assertAlmostEqual(matrix[0, 3], translation[0])
        self.assertAlmostEqual(matrix[1, 3], translation[1])
        self.assertAlmostEqual(matrix[2, 3], translation[2])
        self.assertEqual(matrix[3, 3], 1.0)
        self.assertEqual(matrix[3, 0], 0.0)
        self.assertEqual(matrix[3, 1], 0.0)
        self.assertEqual(matrix[3, 2], 0.0)

    def test_register_corresponding_points_rotation(self):
        point_set_from = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]]
        ground_truth = Rotation.as_matrix(Rotation.from_euler(seq="x", angles=90, degrees=True))
        point_set_to = list(numpy.matmul(ground_truth, numpy.asarray(point_set_from).T).T)
        result = MathUtils.register_corresponding_points(point_set_from, point_set_to)
        for i in range(0, 3):
            for j in range(0, 3):
                self.assertAlmostEqual(result[i][j], ground_truth[i][j])
        self.assertAlmostEqual(result[0, 3], 0.0)
        self.assertAlmostEqual(result[1, 3], 0.0)
        self.assertAlmostEqual(result[2, 3], 0.0)
        self.assertEqual(result[3, 3], 1.0)
        self.assertEqual(result[3, 0], 0.0)
        self.assertEqual(result[3, 1], 0.0)
        self.assertEqual(result[3, 2], 0.0)

    def test_register_corresponding_points_rotation_and_translation(self):
        point_set_from = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]]
        ground_truth = Rotation.as_matrix(Rotation.from_euler(seq="z", angles=-90, degrees=True))
        point_set_to = list(numpy.matmul(ground_truth, numpy.asarray(point_set_from).T).T)
        translation = [10, -20, 30]
        point_set_to = list(point_set_to + numpy.stack(translation))
        result = MathUtils.register_corresponding_points(point_set_from, point_set_to)
        for i in range(0, 3):
            for j in range(0, 3):
                self.assertAlmostEqual(result[i][j], ground_truth[i][j])
        self.assertAlmostEqual(result[0, 3], translation[0])
        self.assertAlmostEqual(result[1, 3], translation[1])
        self.assertAlmostEqual(result[2, 3], translation[2])
        self.assertEqual(result[3, 3], 1.0)
        self.assertEqual(result[3, 0], 0.0)
        self.assertEqual(result[3, 1], 0.0)
        self.assertEqual(result[3, 2], 0.0)

    def test_register_corresponding_points_too_few_points(self):
        point_set_from = [[0.0, 0.0, 0.0]]
        point_set_to = point_set_from
        try:
            MathUtils.register_corresponding_points(point_set_from, point_set_to)
        except ValueError:
            pass

    def test_register_corresponding_points_inequal_point_set_lengths(self):
        point_set_from = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]]
        point_set_to = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]
        try:
            MathUtils.register_corresponding_points(point_set_from, point_set_to)
        except ValueError:
            pass

    def test_register_corresponding_points_collinear(self):
        point_set_from = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]]
        point_set_to = point_set_from
        try:
            MathUtils.register_corresponding_points(point_set_from, point_set_to, collinearity_do_check=True)
        except ValueError:
            pass

    def test_register_corresponding_points_singular(self):
        point_set_from = [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]]
        point_set_to = point_set_from
        try:
            MathUtils.register_corresponding_points(point_set_from, point_set_to, collinearity_do_check=True)
        except ValueError:
            pass

