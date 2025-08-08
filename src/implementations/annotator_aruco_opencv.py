from .common_aruco_opencv import ArucoOpenCVCommon
from src.common import \
    Annotator, \
    Annotation, \
    KeyValueMetaAny, \
    KeyValueSimpleAny, \
    MCTAnnotatorRuntimeError, \
    SeverityLabel, \
    StatusMessageSource
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
            self.add_status_message(
                severity=SeverityLabel.ERROR,
                message=message)
            self.set_status(Annotator.Status.FAILURE)
            return
        image_greyscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self._snapshots_identified, self._snapshots_unidentified = ArucoOpenCVCommon.annotations_from_greyscale_image(
            aruco_detector_parameters=self._aruco_parameters,
            aruco_dictionary=self._aruco_dictionary,
            image_greyscale=image_greyscale)
        self._update_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)
