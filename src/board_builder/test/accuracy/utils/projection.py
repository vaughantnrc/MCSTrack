# The math used is consistent with the OpenCV documentation: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from src.common.structures import Matrix4x4, IntrinsicParameters


def projection(
    world_point,
    detector_to_reference_matrix: Matrix4x4,
    detector_intrinsics: IntrinsicParameters
) -> list[float]:
    """
    Projects 3D points onto the 2D detector frame
    """

    object_points: np.ndarray = np.asarray([world_point])
    reference_to_detector_matrix: Matrix4x4 = detector_to_reference_matrix.inverse()
    reference_to_detector_matrix_3x3: np.ndarray = reference_to_detector_matrix.as_numpy_array()[0:3, 0:3]
    rvec: np.ndarray = Rotation.from_matrix(reference_to_detector_matrix_3x3).as_rotvec(degrees=False)
    tvec: np.ndarray = np.asarray(reference_to_detector_matrix.get_translation())
    camera_matrix: np.ndarray = np.asarray(
        [[detector_intrinsics.focal_length_x_px, 0.0, detector_intrinsics.optical_center_x_px],
         [0.0, detector_intrinsics.focal_length_y_px, detector_intrinsics.optical_center_y_px],
         [0.0, 0.0, 1.0]])
    dist_coeffs: np.ndarray = np.asarray(detector_intrinsics.get_distortion_coefficients())
    image_points, _ = cv2.projectPoints(
        objectPoints=object_points,
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs)

    return [image_points[0][0][0], image_points[0][0][1]]

    # print(image_points)
    #
    # cx, cy = detector_intrinsics.optical_center_x_px, detector_intrinsics.optical_center_y_px
    #
    # # Convert world point to camera space
    # point_relative = np.asarray(world_point) - np.asarray(detector_to_reference_matrix.get_translation())
    #
    # point_relative_normalized = point_relative / np.linalg.norm(point_relative)
    #
    # # Project point onto camera axes
    # point_detector_normalized = detector_to_reference_matrix.inverse().as_numpy_array() @ np.asarray([
    #     point_relative_normalized[0],
    #     point_relative_normalized[1],
    #     point_relative_normalized[2],
    #     0.0])
    # x = point_detector_normalized[0]
    # y = point_detector_normalized[1]
    # z = point_detector_normalized[2]
    #
    # # Check if the point is in front of the camera
    # if z <= 0:
    #     return None  # Point is behind the camera
    #
    # # Project to image plane
    # pixel_x = cx + detector_intrinsics.focal_length_x_px * x / z
    # pixel_y = cy + detector_intrinsics.focal_length_y_px * y / z
    #
    # print(f"{pixel_x} {pixel_y}")

