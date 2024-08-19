# The math used is consistent with the OpenCV documentation: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
import numpy as np
from src.common.structures import Matrix4x4


def projection(
    world_point,
    detector_to_reference_matrix: Matrix4x4,
    detector_intrinsics
):
    """
    Projects 3D points onto the 2D detector frame
    """
    cx, cy = detector_intrinsics.optical_center_x_px, detector_intrinsics.optical_center_y_px

    # Convert world point to camera space
    point_relative = np.asarray(world_point) - np.asarray(detector_to_reference_matrix.get_translation())

    point_relative_normalized = point_relative / np.linalg.norm(point_relative)

    # Project point onto camera axes
    point_detector_normalized = detector_to_reference_matrix.inverse().as_numpy_array() @ np.asarray([
        point_relative_normalized[0],
        point_relative_normalized[1],
        point_relative_normalized[2],
        0.0])
    x = point_detector_normalized[0]
    y = point_detector_normalized[1]
    z = point_detector_normalized[2]

    # Check if the point is in front of the camera
    if z <= 0:
        return None  # Point is behind the camera

    # Project to image plane
    pixel_x = cx + detector_intrinsics.focal_length_x_px * x / z
    pixel_y = cy - detector_intrinsics.focal_length_y_px * y / z

    return [pixel_x, pixel_y]
