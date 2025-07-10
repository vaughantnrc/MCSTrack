from .common_aruco_opencv import ArucoOpenCVCommon
from src.common import \
    Annotator, \
    MCTAnnotatorRuntimeError, \
    StatusMessageSource
from src.common.structures import \
    Annotation, \
    KeyValueMetaAny, \
    KeyValueSimpleAny, \
    RELATION_CHARACTER
import cv2.aruco
import datetime
import logging
import numpy
from typing import Optional


logger = logging.getLogger(__name__)


class ArucoOpenCVAnnotator(Annotator):

    _aruco_dictionary: Optional  # created by OpenCV, type cv2.aruco.Dictionary
    _aruco_parameters: ...  # created by OpenCV, type cv2.aruco.DetectorParameters
    _snapshots_identified: list[Annotation]  # Markers that are determined to be valid, and are identified
    _snapshots_unidentified: list[Annotation]  # Looked at first like markers but got filtered out
    _update_timestamp_utc: datetime.datetime

    def __init__(
        self,
        configuration: Annotator.Configuration,
        status_message_source: StatusMessageSource
    ):
        super().__init__(
            configuration=configuration,
            status_message_source=status_message_source)

        self._aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self._aruco_parameters = cv2.aruco.DetectorParameters()
        self._snapshots_identified = list()
        self._snapshots_unidentified = list()
        self._update_timestamp_utc = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        self.set_status(Annotator.Status.RUNNING)  # Always running

    def get_changed_timestamp(self) -> datetime.datetime:
        return self._update_timestamp_utc

    def get_markers_detected(self) -> list[Annotation]:
        return self._snapshots_identified

    def get_markers_rejected(self) -> list[Annotation]:
        return self._snapshots_unidentified

    def get_parameters(self) -> list[KeyValueMetaAny]:
        return ArucoOpenCVCommon.assign_aruco_detection_parameters_to_key_value_list(self._aruco_parameters)

    @staticmethod
    def get_type_identifier() -> str:
        return "aruco_opencv"

    # noinspection DuplicatedCode
    def set_parameters(
        self,
        parameters: list[KeyValueSimpleAny]
    ) -> None:
        mismatched_keys: list[str] = ArucoOpenCVCommon.assign_key_value_list_to_aruco_detection_parameters(
            detection_parameters=self._aruco_parameters,
            key_value_list=parameters)
        if len(mismatched_keys) > 0:
            raise MCTAnnotatorRuntimeError(
                message=f"The following parameters could not be applied due to key mismatch: {str(mismatched_keys)}")

    def update(
        self,
        image: numpy.ndarray
    ) -> None:
        if self._aruco_dictionary is None:
            message: str = "No ArUco dictionary has been set."
            self.add_status_message(severity="error", message=message)
            self.set_status(Annotator.Status.FAILURE)
            return

        image_greyscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        (detected_corner_points_raw, detected_dictionary_indices, rejected_corner_points_raw) = cv2.aruco.detectMarkers(
            image=image_greyscale,
            dictionary=self._aruco_dictionary,
            parameters=self._aruco_parameters)

        self._snapshots_identified = list()
        # note: detected_indices is (inconsistently) None sometimes if nothing is detected
        if detected_dictionary_indices is not None and len(detected_dictionary_indices) > 0:
            detected_count = detected_dictionary_indices.size
            # Shape of some output was previously observed to (also) be inconsistent... make it consistent here:
            detected_corner_points_px = numpy.array(detected_corner_points_raw).reshape((detected_count, 4, 2))
            detected_dictionary_indices = list(detected_dictionary_indices.reshape(detected_count))
            for detected_index, detected_id in enumerate(detected_dictionary_indices):
                for corner_index in range(4):
                    detected_label: str = f"{detected_id}{RELATION_CHARACTER}{corner_index}"
                    self._snapshots_identified.append(Annotation(
                        label=detected_label,
                        x_px=detected_corner_points_px[detected_index][corner_index][0],
                        y_px=detected_corner_points_px[detected_index][corner_index][1]))

        self._snapshots_unidentified = list()
        if rejected_corner_points_raw:
            rejected_corner_points_px = numpy.array(rejected_corner_points_raw).reshape((-1, 4, 2))
            for rejected_index in range(rejected_corner_points_px.shape[0]):
                for corner_index in range(4):
                    self._snapshots_unidentified.append(Annotation(
                        label=Annotation.UNIDENTIFIED_LABEL,
                        x_px=rejected_corner_points_px[rejected_index][corner_index][0],
                        y_px=rejected_corner_points_px[rejected_index][corner_index][1]))

        self._update_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)
