from src.common import \
    ExtrinsicCalibration, \
    ExtrinsicCalibrationDetectorResult, \
    ImageResolution, \
    ImageUtils, \
    IntrinsicParameters, \
    KeyValueSimpleAny, \
    KeyValueSimpleString
from src.implementations.common_aruco_opencv import \
    ArucoOpenCVCommon
from src.implementations.extrinsic_charuco_opencv import \
    CharucoOpenCVExtrinsicCalibrator
import cv2
import datetime
import numpy
import os
import re
from scipy.spatial.transform import Rotation
from tempfile import TemporaryDirectory
from typing import Final
import unittest


IMAGE_CONTENT_PATTERN: Final[re.Pattern] = re.compile(r"C([a-zA-Z0-9]+)_F([a-zA-Z0-9]+).png")
IMAGE_CONTENT_MATCH_INDEX_CAMERA: Final[int] = 1
IMAGE_CONTENT_MATCH_INDEX_FRAME: Final[int] = 2
IMAGE_RESOLUTION: Final[ImageResolution] = ImageResolution(x_px=1920, y_px=1080)
MARKER_DETECTION_PARAMETERS: list[KeyValueSimpleAny] = [
    KeyValueSimpleString(
        key=ArucoOpenCVCommon.KEY_CORNER_REFINEMENT_METHOD,
        value=ArucoOpenCVCommon.CORNER_REFINEMENT_METHOD_SUBPIX)]
THRESHOLD_TRANSLATION_IN_PLANE_MM: Final[float] = 10
THRESHOLD_TRANSLATION_OUT_OF_PLANE_MM: Final[float] = 25
THRESHOLD_ROTATION_DEG: Final[float] = 1


class TestPoseSolver(unittest.TestCase):
    def test(self):
        # Organize ourselves with respect to the input data
        image_location: str = os.path.join("images", "simulated", "ideal")
        image_contents: list[str] = os.listdir(image_location)
        image_filepaths_by_camera_frame: dict[str, dict[str, str]] = dict()  # Access as: x[CameraID][FrameID]
        image_filepaths_by_frame_camera: dict[str, dict[str, str]] = dict()  # Access as: x[FrameID][CameraID]
        timestamps_iso8601_by_frame: dict[str, str] = dict()  # Access as: x[FrameID]
        reference_time: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
        image_count: int = 0
        for image_content in image_contents:
            if image_content == "about.txt":
                continue

            image_filepath: str = os.path.join(image_location, image_content)
            if not os.path.isfile(image_filepath):
                continue

            match: re.Match = IMAGE_CONTENT_PATTERN.match(image_content)
            if match is None:
                self.fail(
                    f"The input filename {image_content} did not match the expected pattern. "
                    "Were files moved or added?")

            camera_id: str = match.group(IMAGE_CONTENT_MATCH_INDEX_CAMERA)
            frame_id: str = match.group(IMAGE_CONTENT_MATCH_INDEX_FRAME)
            if camera_id not in image_filepaths_by_camera_frame:
                image_filepaths_by_camera_frame[camera_id] = dict()
            image_filepaths_by_camera_frame[camera_id][frame_id] = image_filepath
            if frame_id not in image_filepaths_by_frame_camera:
                image_filepaths_by_frame_camera[frame_id] = dict()
                timestamps_iso8601_by_frame[frame_id] = (
                    reference_time
                    - datetime.timedelta(hours=1)
                    + datetime.timedelta(seconds=image_count)).isoformat()
            image_filepaths_by_frame_camera[frame_id][camera_id] = image_filepath
            image_count += 1

        # All cameras have the same imaging parameters.
        # These were calculated by hand assuming lenses without any distortions
        intrinsic_parameters: IntrinsicParameters = IntrinsicParameters(
            focal_length_x_px=3582.76878,
            focal_length_y_px=3640.38430,
            optical_center_x_px=960.0,
            optical_center_y_px=540.0,
            radial_distortion_coefficients=[0, 0, 0],
            tangential_distortion_coefficients=[0, 0])

        intrinsics_by_camera: dict[str, IntrinsicParameters] = dict()  # Access as x[camera_id]
        for camera_id in image_filepaths_by_camera_frame.keys():
            intrinsics_by_camera[camera_id] = intrinsic_parameters

        extrinsic_calibrator: CharucoOpenCVExtrinsicCalibrator
        extrinsic_calibration: ExtrinsicCalibration
        with TemporaryDirectory() as temppath:
            extrinsic_calibrator: CharucoOpenCVExtrinsicCalibrator = CharucoOpenCVExtrinsicCalibrator(
                configuration=CharucoOpenCVExtrinsicCalibrator.Configuration(data_path=temppath))
            for frame_id, image_filepaths_by_camera_id in image_filepaths_by_frame_camera.items():
                for camera_id, image_filepath in image_filepaths_by_camera_id.items():
                    image: numpy.ndarray = cv2.imread(image_filepath)
                    image_base64: str = ImageUtils.image_to_base64(image)
                    extrinsic_calibrator.add_image(
                        image_base64=image_base64,
                        detector_label=camera_id,
                        timestamp_utc_iso8601=timestamps_iso8601_by_frame[frame_id])
            for camera_id, intrinsic_parameters in intrinsics_by_camera.items():
                extrinsic_calibrator.intrinsic_parameters_update(
                    detector_label=camera_id,
                    intrinsic_parameters=intrinsic_parameters)
            _, extrinsic_calibration = extrinsic_calibrator.calculate()

        # label, translation, rotation (as quaternion)
        ground_truth_detector_poses: dict[str, tuple[list[float], list[float]]] = {
            "01": ([-866.025, 0., 500.], [ 0.353553, -0.353553, -0.612372, 0.612372]),
            "02": ([-612.372, -612.372, 500.], [ 0.46194, -0.191342, -0.331414, 0.800103]),
            "03": ([0., -866.025, 500.], [ 0.5, 0., 0., 0.866025]),
            "04": ([612.372, -612.372, 500.], [ 0.46194, 0.191342, 0.331414, 0.800103]),
            "05": ([866.025, 0., 500.], [ 0.353553, 0.353553, 0.612372, 0.612372]),
            "06": ([-707.107, 0., 707.107], [ 0.270598, -0.270598, -0.653281, 0.653282]),
            "07": ([-500., -500., 707.107], [ 0.353553, -0.146447, -0.353553, 0.853553]),
            "08": ([0., -707.107, 707.107], [ 0.382683, 0., 0., 0.92388]),
            "09": ([500., -500., 707.107], [ 0.353553, 0.146447, 0.353553, 0.853553]),
            "10": ([707.107, 0., 707.107], [ 0.270598, 0.270598, 0.653282, 0.653281])}

        calibrated_value: ExtrinsicCalibrationDetectorResult
        for calibrated_value in extrinsic_calibration.calibrated_values:
            expected_translation: list[float]
            expected_rotation_quaternion: list[float]
            expected_translation, expected_rotation_quaternion = \
                ground_truth_detector_poses[calibrated_value.detector_label]
            obtained_translation: list[float] = calibrated_value.detector_to_reference.get_translation()
            obtained_rotation_quaternion: list[float] = \
                calibrated_value.detector_to_reference.get_rotation_as_quaternion()

            translation_difference_vector: numpy.ndarray = \
                numpy.asarray(expected_translation) - numpy.asarray(obtained_translation)
            # noinspection PyArgumentList
            image_plane_normal: numpy.ndarray = numpy.matmul(
                Rotation.from_quat(expected_rotation_quaternion).as_matrix(),
                numpy.asarray([0, 0, 1]))
            translation_difference_in_plane_mm: float = numpy.dot(
                translation_difference_vector,
                image_plane_normal)
            self.assertLess(translation_difference_in_plane_mm, THRESHOLD_TRANSLATION_IN_PLANE_MM)

            # Pythagorean theorem to get the translation component out of plane
            translation_difference_out_of_plane_mm: float = numpy.sqrt(
                translation_difference_in_plane_mm ** 2 + numpy.linalg.norm(translation_difference_vector) ** 2)
            self.assertLess(translation_difference_out_of_plane_mm, THRESHOLD_TRANSLATION_OUT_OF_PLANE_MM)

            # noinspection PyArgumentList
            rotation_difference_deg: float = numpy.linalg.norm(
                Rotation.from_matrix(numpy.matmul(
                    Rotation.from_quat(expected_rotation_quaternion).as_matrix(),
                    numpy.linalg.inv(Rotation.from_quat(obtained_rotation_quaternion).as_matrix())
                )).as_rotvec(degrees=True))
            self.assertLess(rotation_difference_deg, THRESHOLD_ROTATION_DEG)

            # print(
            #     f"{calibrated_value.detector_label}: \n"
            #     f"  Expected translation: {expected_translation}\n"
            #     f"  Obtained translation: {obtained_translation}\n"
            #     f"  Expected rotation: {expected_rotation_quaternion}\n"
            #     f"  Obtained rotation: {obtained_rotation_quaternion}\n"
            #     f"  Translation Diff IP: {translation_difference_in_plane_mm}\n"
            #     f"  Translation Diff OOP: {translation_difference_out_of_plane_mm}\n"
            #     f"  Rotation Diff Deg: {rotation_difference_deg}")

        # These are from the Blender file, copied by hand.
        # They are not currently used in the test, but they are here for possible later use.
        # Note that Blender represents quaternions in WXYZ, and here it is XYZW.
        # The numbers have not been verified, and as such it is possible that there may be some errors.
        _ground_truth_frame_poses: dict[str, tuple[list[float], list[float]]] = {
            "A0": ([0., 0., 10.], [ 0., 0., 0., 1.]),
            "B1": ([125., 100., 30.], [ 0.065263, -0.113039, 0.495722, 0.858616]),
            "B2": ([-125., 100., 30.], [ 0.113039, -0.065263, 0.858616, 0.495722]),
            "B3": ([125., -100., 30.], [ 0.065263, 0.113039, 0.495722, -0.858616]),
            "B4": ([-125., -100., 30.], [ 0.113039, 0.065263, 0.858616, -0.495722]),
            "C1": ([250., 250., 30.], [ 0.065263, -0.113039, 0.495722, 0.858616]),
            "C2": ([-250., 250., 30.], [ 0.113039, -0.065263, 0.858616, 0.495722]),
            "C3": ([250., -250., 30.], [ 0.065263, 0.113039, 0.495722, -0.858616]),
            "C4": ([-250., -250., 30.], [ 0.113039, 0.065263, 0.858616, -0.495722]),
            "D1": ([-25., -25., 80.], [ 0.176704, -0.4266, 0.339444, 0.819491]),
            "D2": ([25., -25., 80.], [ 0.4266, -0.176703, 0.819491, 0.339444]),
            "D3": ([-25., 25., 80.], [ -0.176704, -0.4266, -0.339444, 0.819491]),
            "D4": ([25., 25., 80.], [ 0.4266, 0.176704, 0.819491, -0.339445]),
            "E1": ([-175., 5., 105.], [ 0.06027, -0.457798, 0.115778, 0.879422]),
            "E2": ([0., -175., 105.], [ 0.326506, -0.326506, 0.627211, 0.627211]),
            "E3": ([175., -5., 105.], [ 0.457798, -0.06027, 0.879422, 0.115778])}


if __name__ == "__main__":
    a = TestPoseSolver()
    a.test()
