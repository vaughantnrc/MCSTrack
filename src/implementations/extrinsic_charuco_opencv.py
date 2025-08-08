from .common_aruco_opencv import ArucoOpenCVCommon
from src.common import \
    ExtrinsicCalibration, \
    ExtrinsicCalibrator, \
    IntrinsicParameters
import cv2
import cv2.aruco


class CharucoOpenCVIntrinsicCalibrator(ExtrinsicCalibrator):
    def _calculate_implementation(
        self,
        detector_intrinsics_by_label: dict[str, IntrinsicParameters],
        image_metadata_list: list[ExtrinsicCalibrator.ImageMetadata]
    ) -> tuple[ExtrinsicCalibration, list[ExtrinsicCalibrator.ImageMetadata]]:
        aruco_detector_parameters: ... = cv2.aruco.DetectorParameters()

        charuco_spec = ArucoOpenCVCommon.CharucoBoard()
        charuco_board: cv2.aruco.CharucoBoard = charuco_spec.create_board()

        raise NotImplementedError()

        # data:
        #   per detector:
        #     initial_frame transform to reference_target
        #     final transform to reference_target
        #     per frame:
        #       image
        #       (marker_id,2d_points)s
        #   final (frame_id,marker_id,3d_points)s
        #
        # input data:
        #   per detector:
        #     per frame:
        #       PNG: image
        #
        # output data:
        #   per detector:
        #     JSON: transform to reference_target
        #     JSON: Additional stats, inc. reference_target definition

        # Constraint: Reference board must be visible to all cameras for first frame_id (frame_0)
        # - Estimate camera position relative to frame_0
        #   MathUtils.estimate_matrix_transform_to_detector()
        # - Convert points to rays for all (camera_id, frame_id) using frame_0 as basis
        #   MathUtils.convert_detector_corners_to_vectors()
        # - For each (frame_id, point_id), intersect N rays to get 3D points. All 3D Points = working_points.
        #   MathUtils.closest_intersection_between_n_lines()
        # - Refine camera positions based on working_points via PnP
        #   MathUtils.estimate_matrix_transform_to_detector()
        # Iterate max times or until convergence:
        #  - Convert points to rays for all (camera_id, frame_id), using working_points as basis
        #  - For each (frame_id, point_id), intersect N rays to get 3D points. All 3D Points = working_points.
        #  - Refine camera positions based on working_points via PnP
