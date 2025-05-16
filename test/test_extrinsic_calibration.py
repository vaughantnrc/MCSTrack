import cv2
import numpy
import os
import re
from src.common import ImageCoding, StatusMessageSource
from src.common.structures import \
    CORNER_REFINEMENT_METHOD_SUBPIX, \
    ImageResolution, \
    IntrinsicCalibration, \
    KeyValueSimpleAny, \
    KeyValueSimpleString, \
    MarkerSnapshot
from src.detector import Calibrator
from src.detector.implementations.marker_aruco_opencv import ArucoOpenCVMarker
from src.detector.structures import CalibratorConfiguration
from src.detector.util import KEY_CORNER_REFINEMENT_METHOD
from tempfile import TemporaryDirectory
from typing import Final
import unittest


IMAGE_CONTENT_PATTERN: Final[re.Pattern] = re.compile(r"C([a-zA-Z0-9]+)_F([a-zA-Z0-9]+).png")
IMAGE_CONTENT_MATCH_INDEX_CAMERA: Final[int] = 1
IMAGE_CONTENT_MATCH_INDEX_FRAME: Final[int] = 2
IMAGE_RESOLUTION: Final[ImageResolution] = ImageResolution(x_px=1920, y_px=1080)
MARKER_DETECTION_PARAMETERS: list[KeyValueSimpleAny] = [
    KeyValueSimpleString(
        key=KEY_CORNER_REFINEMENT_METHOD,
        value=CORNER_REFINEMENT_METHOD_SUBPIX)]


class TestPoseSolver(unittest.TestCase):
    def test(self):
        status_message_source: StatusMessageSource = StatusMessageSource(
            source_label="test",
            send_to_logger=True)  # Python built-in logger

        # Organize ourselves with respect to the input data
        image_location: str = os.path.join("images", "simulated", "ideal")
        image_contents: list[str] = os.listdir(image_location)
        image_filepaths: dict[str, dict[str, str]] = dict()  # Access as: images[CameraID][FrameID]
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
            if camera_id not in image_filepaths:
                image_filepaths[camera_id] = dict()
            image_filepaths[camera_id][frame_id] = image_filepath
        image_count: int = sum(len(image_filepaths[camera_id]) for camera_id in image_filepaths.keys())
        message = f"Found {image_count} image files."
        status_message_source.enqueue_status_message(severity="info", message=message)

        # All cameras have the same imaging parameters.
        # To simplify our lives and ensure a reasonable result,
        # we'll calibrate all of the cameras with the same set of input images.
        # We'll use all images from the A# and B# sets of frames.
        calibration_result: IntrinsicCalibration | None = None
        with TemporaryDirectory() as temppath:
            calibrator: Calibrator = Calibrator(
                configuration=CalibratorConfiguration(data_path=temppath),
                status_message_source=status_message_source)
            for camera_id, image_filepaths_by_frame_id in image_filepaths.items():
                for frame_id, image_filepath in image_filepaths_by_frame_id.items():
                    if not frame_id.startswith("A") and not frame_id.startswith("B"):
                        continue
                    image: numpy.ndarray = cv2.imread(image_filepath)
                    image_base64: str = ImageCoding.image_to_base64(image)
                    calibrator.add_image(image_base64)
            _, calibration_result = calibrator.calculate(
                image_resolution=IMAGE_RESOLUTION,
                marker_parameters=MARKER_DETECTION_PARAMETERS)

        print(calibration_result.model_dump())
        return

        marker: ArucoOpenCVMarker = ArucoOpenCVMarker(
            configuration={"method": "aruco_opencv"},
            status_message_source=status_message_source)
        marker.set_parameters(parameters=MARKER_DETECTION_PARAMETERS)
        image_marker_snapshots: dict[str, dict[str, list[MarkerSnapshot]]] = dict()
        detection_count: int = 0
        for camera_id, image_filepaths_by_frame_id in image_filepaths.items():
            for frame_id, image_filepath in image_filepaths_by_frame_id.items():
                if camera_id not in image_marker_snapshots:
                    image_marker_snapshots[camera_id] = dict()
                image: numpy.ndarray = cv2.imread(image_filepath)
                marker.update(image)
                marker_snapshots: list[MarkerSnapshot] = marker.get_markers_detected()
                image_marker_snapshots[camera_id][frame_id] = marker_snapshots
                detection_count += len(marker_snapshots)
        message = f"{detection_count} detections."
        status_message_source.enqueue_status_message(severity="info", message=message)
        print(message)
