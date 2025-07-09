from ..exceptions import MCTDetectorRuntimeError
from ..interfaces import AbstractMarker
from ..structures import \
    MarkerConfiguration, \
    MarkerStatus
from ..util import \
    assign_aruco_detection_parameters_to_key_value_list, \
    assign_key_value_list_to_aruco_detection_parameters
from src.common import StatusMessageSource
from src.common.structures import \
    KeyValueMetaAny, \
    KeyValueSimpleAny, \
    MarkerCornerImagePoint, \
    MarkerSnapshot
import cv2.aruco
import datetime
import logging
import numpy
from typing import Any


logger = logging.getLogger(__name__)


# Look at https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
# for documentation on individual parameters


class ArucoOpenCVMarker(AbstractMarker):

    _marker_dictionary: Any | None  # created by OpenCV, type cv2.aruco.Dictionary
    _marker_parameters: Any  # created by OpenCV, type cv2.aruco.DetectorParameters
    _marker_label_reverse_dictionary: dict[int, str]
    _marker_detected_snapshots: list[MarkerSnapshot]
    _marker_rejected_snapshots: list[MarkerSnapshot]
    _marker_timestamp_utc: datetime.datetime

    def __init__(
        self,
        configuration: MarkerConfiguration,
        status_message_source: StatusMessageSource
    ):
        super().__init__(
            configuration=configuration,
            status_message_source=status_message_source)

        self._marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self._marker_parameters = cv2.aruco.DetectorParameters()
        self._marker_label_reverse_dictionary = dict()
        self._marker_detected_snapshots = list()  # Markers that are determined to be valid, and are identified
        self._marker_rejected_snapshots = list()  # Things that looked at first like markers but got later filtered out
        self._marker_timestamp_utc = datetime.datetime.min
        self.set_status(MarkerStatus.RUNNING)  # Always running

    def get_changed_timestamp(self) -> datetime.datetime:
        return self._marker_timestamp_utc

    def get_markers_detected(self) -> list[MarkerSnapshot]:
        return self._marker_detected_snapshots

    def get_markers_rejected(self) -> list[MarkerSnapshot]:
        return self._marker_rejected_snapshots

    def get_parameters(self) -> list[KeyValueMetaAny]:
        return assign_aruco_detection_parameters_to_key_value_list(self._marker_parameters)

    @staticmethod
    def get_type_identifier() -> str:
        return "aruco_opencv"

    @staticmethod
    def _marker_corner_image_point_list_from_embedded_list(
        corner_image_points_px: list[list[float]]
    ) -> list[MarkerCornerImagePoint]:
        corner_image_point_list: list[MarkerCornerImagePoint] = list()
        assert len(corner_image_points_px) == 4
        for corner_image_point_px in corner_image_points_px:
            corner_image_point_list.append(MarkerCornerImagePoint(
                x_px=corner_image_point_px[0],
                y_px=corner_image_point_px[1]))
        return corner_image_point_list

    # noinspection DuplicatedCode
    def set_parameters(
        self,
        parameters: list[KeyValueSimpleAny]
    ) -> None:
        mismatched_keys: list[str] = assign_key_value_list_to_aruco_detection_parameters(
            detection_parameters=self._marker_parameters,
            key_value_list=parameters)
        if len(mismatched_keys) > 0:
            raise MCTDetectorRuntimeError(
                message=f"The following parameters could not be applied due to key mismatch: {str(mismatched_keys)}")

    def update(
        self,
        image: numpy.ndarray
    ) -> None:
        if self._marker_dictionary is None:
            message: str = "No marker dictionary has been set."
            self.add_status_message(severity="error", message=message)
            self.set_status(MarkerStatus.FAILURE)
            return

        image_greyscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        (detected_corner_points_raw, detected_dictionary_indices, rejected_corner_points_raw) = cv2.aruco.detectMarkers(
            image=image_greyscale,
            dictionary=self._marker_dictionary,
            parameters=self._marker_parameters)

        self._marker_detected_snapshots = list()
        # note: detected_indices is (inconsistently) None sometimes if no markers are detected
        if detected_dictionary_indices is not None and len(detected_dictionary_indices) > 0:
            detected_marker_count = detected_dictionary_indices.size
            # Shape of some output was previously observed to (also) be inconsistent... make it consistent here:
            detected_corner_points_px = numpy.array(detected_corner_points_raw).reshape((detected_marker_count, 4, 2))
            detected_dictionary_indices = list(detected_dictionary_indices.reshape(detected_marker_count))
            for detected_marker_index, detected_marker_id in enumerate(detected_dictionary_indices):
                if False:  # TODO: Re-enable
                    if detected_marker_id not in self._marker_label_reverse_dictionary:
                        message: str = \
                            f"Found a marker with index {detected_marker_id} "\
                            "but it does not appear in the dictionary."
                        self.add_status_message(severity="error", message=message)
                        self.set_status(MarkerStatus.FAILURE)
                        return
                    marker_label: str = self._marker_label_reverse_dictionary[detected_marker_id]
                else:
                    marker_label: str = str(detected_marker_id)
                corner_image_points_px = detected_corner_points_px[detected_marker_index]
                corner_image_points: list[MarkerCornerImagePoint] = \
                    self._marker_corner_image_point_list_from_embedded_list(
                        corner_image_points_px=corner_image_points_px.tolist())
                self._marker_detected_snapshots.append(MarkerSnapshot(
                    label=marker_label,
                    corner_image_points=corner_image_points))

        self._marker_rejected_snapshots = list()
        if rejected_corner_points_raw:
            rejected_corner_points_px = numpy.array(rejected_corner_points_raw).reshape((-1, 4, 2))
            for rejected_marker_index in range(rejected_corner_points_px.shape[0]):
                corner_image_points_px = rejected_corner_points_px[rejected_marker_index]
                corner_image_points: list[MarkerCornerImagePoint] = \
                    self._marker_corner_image_point_list_from_embedded_list(
                        corner_image_points_px=corner_image_points_px.tolist())
                self._marker_rejected_snapshots.append(MarkerSnapshot(
                    label=f"unknown",
                    corner_image_points=corner_image_points))

        self._marker_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)
