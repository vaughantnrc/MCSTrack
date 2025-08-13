from src.common import \
    ExtrinsicCalibration, \
    ExtrinsicCalibrationDetectorResult, \
    ImageResolution, \
    ImageUtils, \
    IntrinsicParameters, \
    IntrinsicCalibrator, \
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
            _, extrinsic_calibration = extrinsic_calibrator.calculate(detector_intrinsics_by_label=intrinsics_by_camera)

        calibrated_value: ExtrinsicCalibrationDetectorResult
        for calibrated_value in extrinsic_calibration.calibrated_values:
            print(
                f"Detector {calibrated_value.detector_label}:\n"
                f"  Translation: {calibrated_value.detector_to_reference.get_translation()}\n"
                f"  Rotation: {calibrated_value.detector_to_reference.get_rotation_as_quaternion(canonical=True)}")


if __name__ == "__main__":
    a = TestPoseSolver()
    a.test()
