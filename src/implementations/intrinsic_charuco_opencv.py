from .common_aruco_opencv import ArucoOpenCVCommon
from src.common import \
    CalibrationErrorReason, \
    ImageResolution, \
    IntrinsicCalibration, \
    IntrinsicCalibrator, \
    IntrinsicParameters, \
    MCTCalibrationError
import cv2
import cv2.aruco
import datetime
import numpy


class CharucoOpenCVIntrinsicCalibrator(IntrinsicCalibrator):
    def _calculate_implementation(
        self,
        image_resolution: ImageResolution,
        image_metadata_list: list[IntrinsicCalibrator.ImageMetadata]
    ) -> tuple[IntrinsicCalibration, list[IntrinsicCalibrator.ImageMetadata]]:
        aruco_detector_parameters: ... = cv2.aruco.DetectorParameters()

        # mismatched_keys: list[str] = ArucoOpenCVAnnotator.assign_key_value_list_to_aruco_detection_parameters(
        #     detection_parameters=aruco_detector_parameters,
        #     key_value_list=marker_parameters)
        # if len(mismatched_keys) > 0:
        #     raise MCTIntrinsicCalibrationError(
        #         message=f"The following parameters could not be applied due to key mismatch: {str(mismatched_keys)}")

        charuco_spec = ArucoOpenCVCommon.CharucoBoard()
        charuco_board: cv2.aruco.CharucoBoard = charuco_spec.create_board()

        all_charuco_corners = list()
        all_charuco_ids = list()
        used_image_metadata: list[IntrinsicCalibrator.ImageMetadata] = list()
        for metadata in image_metadata_list:
            image_rgb = cv2.imread(metadata.filepath)
            image_greyscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            (marker_corners, marker_ids, _) = cv2.aruco.detectMarkers(
                image=image_greyscale,
                dictionary=charuco_spec.aruco_dictionary(),
                parameters=aruco_detector_parameters)
            if len(marker_corners) <= 0:
                continue
            used_image_metadata.append(metadata)
            # Note:
            # Marker corners are the corners of the markers, whereas
            # ChArUco corners are the corners of the chessboard.
            # ChArUco calibration function works with the corners of the chessboard.
            _, frame_charuco_corners, frame_charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=marker_corners,
                markerIds=marker_ids,
                image=image_greyscale,
                board=charuco_board,
            )
            # Algorithm requires a minimum of 4 markers
            if frame_charuco_corners is not None and len(frame_charuco_corners) >= 4:
                all_charuco_corners.append(frame_charuco_corners)
                all_charuco_ids.append(frame_charuco_ids)

        if len(all_charuco_corners) <= 0:
            raise MCTCalibrationError(
                reason=CalibrationErrorReason.COMPUTATION_FAILURE,
                public_message="The input images did not contain visible markers.")

        # outputs to be stored in these containers
        calibration_result = cv2.aruco.calibrateCameraCharucoExtended(
            charucoCorners=all_charuco_corners,
            charucoIds=all_charuco_ids,
            board=charuco_board,
            imageSize=numpy.array(charuco_spec.size_mm(), dtype="int32"),  # Exception if float
            cameraMatrix=numpy.identity(3, dtype='f'),
            distCoeffs=numpy.zeros(5, dtype='f'))

        charuco_overall_reprojection_error = calibration_result[0]
        charuco_camera_matrix = calibration_result[1]
        charuco_distortion_coefficients = calibration_result[2]
        charuco_rotation_vectors = calibration_result[3]
        charuco_translation_vectors = calibration_result[4]
        charuco_intrinsic_stdevs = calibration_result[5]
        charuco_extrinsic_stdevs = calibration_result[6]
        charuco_reprojection_errors = calibration_result[7]

        supplemental_data: dict = {
            "reprojection_error": charuco_overall_reprojection_error,
            "calibrated_stdevs": [value[0] for value in charuco_intrinsic_stdevs],
            # "marker_parameters": marker_parameters,
            "frame_results": [{
                "image_identifier": used_image_metadata[i].identifier,
                "translation": [
                    charuco_translation_vectors[i][0, 0],
                    charuco_translation_vectors[i][1, 0],
                    charuco_translation_vectors[i][2, 0]],
                "rotation": [
                    charuco_rotation_vectors[i][0, 0],
                    charuco_rotation_vectors[i][1, 0],
                    charuco_rotation_vectors[i][2, 0]],
                "translation_stdev": [
                    charuco_extrinsic_stdevs[i * 6 + 3, 0],
                    charuco_extrinsic_stdevs[i * 6 + 4, 0],
                    charuco_extrinsic_stdevs[i * 6 + 5, 0]],
                "rotation_stdev": [
                    charuco_extrinsic_stdevs[i * 6 + 0, 0],
                    charuco_extrinsic_stdevs[i * 6 + 1, 0],
                    charuco_extrinsic_stdevs[i * 6 + 2, 0]],
                "reprojection_error": charuco_reprojection_errors[i, 0]}
                for i in range(0, len(charuco_reprojection_errors))]}

        # TODO: Assertion on size of distortion coefficients being 5?
        #  Note: OpenCV documentation specifies the order of distortion coefficients
        #  https://docs.opencv.org/4.x/d9/d6a/group__aruco.html#ga366993d29fdddd995fba8c2e6ca811ea
        #  So far I have not seen calibration return a number of coefficients other than 5.
        #  Note too that there is an unchecked expectation that radial distortion be monotonic.

        intrinsic_calibration: IntrinsicCalibration = IntrinsicCalibration(
            timestamp_utc=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            image_resolution=image_resolution,
            calibrated_values=IntrinsicParameters(
                focal_length_x_px=float(charuco_camera_matrix[0, 0]),
                focal_length_y_px=float(charuco_camera_matrix[1, 1]),
                optical_center_x_px=float(charuco_camera_matrix[0, 2]),
                optical_center_y_px=float(charuco_camera_matrix[1, 2]),
                radial_distortion_coefficients=[
                    float(charuco_distortion_coefficients[0, 0]),
                    float(charuco_distortion_coefficients[1, 0]),
                    float(charuco_distortion_coefficients[4, 0])],
                tangential_distortion_coefficients=[
                    float(charuco_distortion_coefficients[2, 0]),
                    float(charuco_distortion_coefficients[3, 0])]),
            supplemental_data=supplemental_data)

        return intrinsic_calibration, used_image_metadata
