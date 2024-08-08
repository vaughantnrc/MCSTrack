from ..structures import \
    IntrinsicParameters, \
    Matrix4x4, \
    TargetBase
import cv2
import numpy
from scipy.spatial.transform import Rotation
from typing import TypeVar


XPointKey = TypeVar("XPointKey")


class MathUtils:
    """
    static class for reused math-related functions.
    """

    def __init__(self):
        raise RuntimeError("This class is not meant to be initialized.")

    @staticmethod
    def convert_detector_points_to_vectors(
        points: list[list[float]],  # [point_index][x/y/z]
        detector_intrinsics: IntrinsicParameters,
        detector_to_reference_matrix: Matrix4x4
    ) -> list[list[float]]:
        """
        Given a detector's matrix transform and its intrinsic properties,
        convert pixel coordinates to ray directions (with origin at the detector).
        """
        distorted_points: numpy.ndarray = numpy.asarray(points)
        distorted_points = numpy.reshape(
            a=distorted_points,
            newshape=(1, len(points), 2))
        undistorted_points: numpy.ndarray = cv2.undistortPoints(
            src=distorted_points,
            cameraMatrix=numpy.asarray(detector_intrinsics.get_matrix()),
            distCoeffs=numpy.asarray(detector_intrinsics.get_distortion_coefficients()))
        rays: list[list[float]] = list()
        origin_point_detector = [0, 0, 0, 1]  # origin
        detector_to_reference: numpy.ndarray = detector_to_reference_matrix.as_numpy_array()
        for undistorted_point in undistorted_points:
            target_point_image = [undistorted_point[0, 0], undistorted_point[0, 1], 1, 1]  # reverse perspective
            target_point_detector = MathUtils.image_to_opengl_vector(target_point_image)
            ray_direction_detector = numpy.subtract(target_point_detector, origin_point_detector)
            ray_direction_detector = ray_direction_detector / numpy.linalg.norm(ray_direction_detector)  # normalize
            ray_direction_reference = numpy.matmul(detector_to_reference, ray_direction_detector)
            rays.append(list(ray_direction_reference[0:3]))
        return rays

    @staticmethod
    def convert_detector_corners_to_vectors(
        corners_by_marker_id: dict[str, list[list[float]]],  # [marker_id][point_index][x/y]
        detector_intrinsics: IntrinsicParameters,
        detector_to_reference_matrix: Matrix4x4
    ) -> dict[str, list[list[float]]]:  # [marker_id][point_index][x/y/z]
        """
        Given a detector's matrix transform and its intrinsic properties,
        convert pixel coordinates to ray directions (with origin at the detector).
        """
        ray_vectors_by_marker_id: dict[str, list[list[float]]] = dict()
        corners: list[list[float]]
        marker_id: str
        for marker_id in corners_by_marker_id.keys():
            corners = corners_by_marker_id[marker_id]
            rays: list[list[float]] = MathUtils.convert_detector_points_to_vectors(
                points=corners,
                detector_intrinsics=detector_intrinsics,
                detector_to_reference_matrix=detector_to_reference_matrix)
            ray_vectors_by_marker_id[marker_id] = rays
        return ray_vectors_by_marker_id

    @staticmethod
    def estimate_matrix_transform_to_detector(
        target: TargetBase,
        corners_by_marker_id: dict[str, list[list[float]]],  # [marker_id][point_index][x/y/z]
        detector_intrinsics: IntrinsicParameters
    ) -> Matrix4x4:
        target_points: list[list[float]] = list()    # ordered points [point_index][x/y/z]
        detector_points: list[list[float]] = list()  # ordered points [point_index][x/y]
        for marker_id in target.get_marker_ids():
            if marker_id in corners_by_marker_id:
                target_points += target.get_points_for_marker_id(marker_id=marker_id)
                detector_points += corners_by_marker_id[marker_id]
        rotation_vector: numpy.ndarray
        translation_vector: numpy.ndarray
        _, rotation_vector, translation_vector = cv2.solvePnP(
            objectPoints=numpy.asarray(target_points),
            imagePoints=numpy.asarray(detector_points),
            cameraMatrix=numpy.asarray(detector_intrinsics.get_matrix()),
            distCoeffs=numpy.asarray(detector_intrinsics.get_distortion_coefficients()))
        rotation_vector = rotation_vector.flatten()
        translation_vector = translation_vector.flatten()
        object_to_camera_matrix = numpy.identity(4, dtype="float32")
        object_to_camera_matrix[0:3, 0:3] = Rotation.from_rotvec(rotation_vector).as_matrix()
        object_to_camera_matrix[0:3, 3] = translation_vector[0:3]
        object_to_detector_matrix = MathUtils.image_to_opengl_transformation_matrix(object_to_camera_matrix)
        return Matrix4x4.from_numpy_array(object_to_detector_matrix)

    @staticmethod
    def image_to_opengl_transformation_matrix(
        transformation_matrix_image: numpy.ndarray
    ) -> numpy.ndarray:
        transformation_matrix_detector = numpy.array(transformation_matrix_image)
        transformation_matrix_detector[1:3, :] = -transformation_matrix_detector[1:3, :]
        return transformation_matrix_detector
        # Equivalent to:
        # transformation_matrix_180 = numpy.identity(4, dtype="float")
        # transformation_matrix_180[1, 1] *= -1
        # transformation_matrix_180[2, 2] *= -1
        # return numpy.matmul(transformation_matrix_180, transformation_matrix_image)

    @staticmethod
    def image_to_opengl_vector(
        vector_image: numpy.ndarray | list[float]
    ) -> numpy.ndarray:
        vector_detector = numpy.array(vector_image)
        vector_detector[1:3] = -vector_detector[1:3]
        return vector_detector
        # Equivalent to:
        # transformation_matrix_180 = numpy.identity(4, dtype="float")
        # transformation_matrix_180[1, 1] *= -1
        # transformation_matrix_180[2, 2] *= -1
        # return numpy.matmul(transformation_matrix_180, vector_image)
