from ..exceptions import MCTDetectorRuntimeError
from ..interfaces import AbstractMarker
from ..structures import \
    MarkerConfiguration, \
    MarkerStatus
from src.common import StatusMessageSource
from src.common.structures import \
    CornerRefinementMethod, \
    CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT, \
    CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT, \
    KeyValueMetaAny, \
    KeyValueMetaBool, \
    KeyValueMetaEnum, \
    KeyValueMetaFloat, \
    KeyValueMetaInt, \
    KeyValueSimpleAbstract, \
    KeyValueSimpleAny, \
    KeyValueSimpleBool, \
    KeyValueSimpleFloat, \
    KeyValueSimpleInt, \
    KeyValueSimpleString, \
    MarkerCornerImagePoint, \
    MarkerSnapshot
import cv2.aruco
import datetime
import logging
import numpy
from typing import Any, Final, get_args


logger = logging.getLogger(__name__)


# Look at https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
# for documentation on individual parameters

# Adaptive Thresholding
KEY_ADAPTIVE_THRESH_WIN_SIZE_MIN: Final[str] = "adaptiveThreshWinSizeMin"
KEY_ADAPTIVE_THRESH_WIN_SIZE_MAX: Final[str] = "adaptiveThreshWinSizeMax"
KEY_ADAPTIVE_THRESH_WIN_SIZE_STEP: Final[str] = "adaptiveThreshWinSizeStep"
KEY_ADAPTIVE_THRESH_CONSTANT: Final[str] = "adaptiveThreshConstant"
# Contour Filtering
KEY_MIN_MARKER_PERIMETER_RATE: Final[str] = "minMarkerPerimeterRate"  # Marker size ratio
KEY_MAX_MARKER_PERIMETER_RATE: Final[str] = "maxMarkerPerimeterRate"
KEY_POLYGONAL_APPROX_ACCURACY_RATE: Final[str] = "polygonalApproxAccuracyRate"  # Square tolerance ratio
KEY_MIN_CORNER_DISTANCE_RATE: Final[str] = "minCornerDistanceRate"  # Corner separation ratio
KEY_MIN_MARKER_DISTANCE_RATE: Final[str] = "minMarkerDistanceRate"  # Marker separation ratio
KEY_MIN_DISTANCE_TO_BORDER: Final[str] = "minDistanceToBorder"  # Border distance in pixels
# Bits Extraction
KEY_MARKER_BORDER_BITS: Final[str] = "markerBorderBits"  # Border width (px)
KEY_MIN_OTSU_STDDEV: Final[str] = "minOtsuStdDev"  # Minimum brightness stdev
KEY_PERSPECTIVE_REMOVE_PIXEL_PER_CELL: Final[str] = "perspectiveRemovePixelPerCell"  # Bit Sampling Rate
KEY_PERSPECTIVE_REMOVE_IGNORED_MARGIN_PER_CELL: Final[str] = "perspectiveRemoveIgnoredMarginPerCell"  # Bit Margin Ratio
# Marker Identification
KEY_MAX_ERRONEOUS_BITS_IN_BORDER_RATE: Final[str] = "maxErroneousBitsInBorderRate"  # Border Error Rate
KEY_ERROR_CORRECTION_RATE: Final[str] = "errorCorrectionRate"  # Error Correction Rat
KEY_DETECT_INVERTED_MARKER: Final[str] = "detectInvertedMarker"
KEY_CORNER_REFINEMENT_METHOD: Final[str] = "cornerRefinementMethod"
KEY_CORNER_REFINEMENT_WIN_SIZE: Final[str] = "cornerRefinementWinSize"
KEY_CORNER_REFINEMENT_MAX_ITERATIONS: Final[str] = "cornerRefinementMaxIterations"
KEY_CORNER_REFINEMENT_MIN_ACCURACY: Final[str] = "cornerRefinementMinAccuracy"
# April Tag Only
KEY_APRIL_TAG_CRITICAL_RAD: Final[str] = "aprilTagCriticalRad"
KEY_APRIL_TAG_DEGLITCH: Final[str] = "aprilTagDeglitch"
KEY_APRIL_TAG_MAX_LINE_FIT_MSE: Final[str] = "aprilTagMaxLineFitMse"
KEY_APRIL_TAG_MAX_N_MAXIMA: Final[str] = "aprilTagMaxNmaxima"
KEY_APRIL_TAG_MIN_CLUSTER_PIXELS: Final[str] = "aprilTagMinClusterPixels"
KEY_APRIL_TAG_MIN_WHITE_BLACK_DIFF: Final[str] = "aprilTagMinWhiteBlackDiff"
KEY_APRIL_TAG_QUAD_DECIMATE: Final[str] = "aprilTagQuadDecimate"
KEY_APRIL_TAG_QUAD_SIGMA: Final[str] = "aprilTagQuadSigma"
# ArUco 3
KEY_USE_ARUCO_3_DETECTION: Final[str] = "useAruco3Detection"
KEY_MIN_MARKER_LENGTH_RATIO_ORIGINAL_IMG: Final[str] = "minMarkerLengthRatioOriginalImg"
KEY_MIN_SIDE_LENGTH_CANONICAL_IMG: Final[str] = "minSideLengthCanonicalImg"


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
        self._marker_parameters = cv2.aruco.DetectorParameters_create()
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
        return_value: list[KeyValueMetaAny] = list()

        return_value.append(KeyValueMetaInt(
            key=KEY_ADAPTIVE_THRESH_WIN_SIZE_MIN,
            value=self._marker_parameters.adaptiveThreshWinSizeMin,
            range_minimum=1,
            range_maximum=99))

        return_value.append(KeyValueMetaInt(
            key=KEY_ADAPTIVE_THRESH_WIN_SIZE_MAX,
            value=self._marker_parameters.adaptiveThreshWinSizeMax,
            range_minimum=1,
            range_maximum=99))

        return_value.append(KeyValueMetaInt(
            key=KEY_ADAPTIVE_THRESH_WIN_SIZE_STEP,
            value=self._marker_parameters.adaptiveThreshWinSizeStep,
            range_minimum=1,
            range_maximum=99,
            range_step=2))

        return_value.append(KeyValueMetaFloat(
            key=KEY_ADAPTIVE_THRESH_CONSTANT,
            value=self._marker_parameters.adaptiveThreshConstant,
            range_minimum=-255.0,
            range_maximum=255.0,
            range_step=1.0))

        return_value.append(KeyValueMetaFloat(
            key=KEY_MIN_MARKER_PERIMETER_RATE,
            value=self._marker_parameters.minMarkerPerimeterRate,
            range_minimum=0,
            range_maximum=8.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_MAX_MARKER_PERIMETER_RATE,
            value=self._marker_parameters.maxMarkerPerimeterRate,
            range_minimum=0.0,
            range_maximum=8.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_POLYGONAL_APPROX_ACCURACY_RATE,
            value=self._marker_parameters.polygonalApproxAccuracyRate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_MIN_CORNER_DISTANCE_RATE,
            value=self._marker_parameters.minCornerDistanceRate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_MIN_MARKER_DISTANCE_RATE,
            value=self._marker_parameters.minMarkerDistanceRate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaInt(
            key=KEY_MIN_DISTANCE_TO_BORDER,
            value=self._marker_parameters.minDistanceToBorder,
            range_minimum=0,
            range_maximum=512))

        return_value.append(KeyValueMetaInt(
            key=KEY_MARKER_BORDER_BITS,
            value=self._marker_parameters.markerBorderBits,
            range_minimum=1,
            range_maximum=9))

        return_value.append(KeyValueMetaFloat(
            key=KEY_MIN_OTSU_STDDEV,
            value=self._marker_parameters.minOtsuStdDev,
            range_minimum=0.0,
            range_maximum=256.0,
            range_step=1.0))

        return_value.append(KeyValueMetaInt(
            key=KEY_PERSPECTIVE_REMOVE_PIXEL_PER_CELL,
            value=self._marker_parameters.perspectiveRemovePixelPerCell,
            range_minimum=1,
            range_maximum=20))

        return_value.append(KeyValueMetaFloat(
            key=KEY_PERSPECTIVE_REMOVE_IGNORED_MARGIN_PER_CELL,
            value=self._marker_parameters.perspectiveRemoveIgnoredMarginPerCell,
            range_minimum=0.0,
            range_maximum=0.5,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_MAX_ERRONEOUS_BITS_IN_BORDER_RATE,
            value=self._marker_parameters.maxErroneousBitsInBorderRate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_ERROR_CORRECTION_RATE,
            value=self._marker_parameters.errorCorrectionRate,
            range_minimum=-0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaBool(
            key=KEY_DETECT_INVERTED_MARKER,
            value=self._marker_parameters.detectInvertedMarker))

        if self._marker_parameters.cornerRefinementMethod not in CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT:
            raise MCTDetectorRuntimeError(
                message=f"Corner refinement method appears to be set to an invalid value: "
                        f"{self._marker_parameters.corner_refinement_method}.")
        corner_refinement_method_text: CornerRefinementMethod = \
            CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT[self._marker_parameters.cornerRefinementMethod]
        return_value.append(KeyValueMetaEnum(
            key=KEY_CORNER_REFINEMENT_METHOD,
            value=corner_refinement_method_text,
            allowable_values=get_args(CornerRefinementMethod)))

        return_value.append(KeyValueMetaInt(
            key=KEY_CORNER_REFINEMENT_WIN_SIZE,
            value=self._marker_parameters.cornerRefinementWinSize,
            range_minimum=1,
            range_maximum=9))

        return_value.append(KeyValueMetaInt(
            key=KEY_CORNER_REFINEMENT_MAX_ITERATIONS,
            value=self._marker_parameters.cornerRefinementMaxIterations,
            range_minimum=1,
            range_maximum=100))

        return_value.append(KeyValueMetaFloat(
            key=KEY_CORNER_REFINEMENT_MIN_ACCURACY,
            value=self._marker_parameters.cornerRefinementMinAccuracy,
            range_minimum=0.0,
            range_maximum=5.0,
            range_step=0.1))

        return_value.append(KeyValueMetaFloat(
            key=KEY_APRIL_TAG_CRITICAL_RAD,
            value=self._marker_parameters.aprilTagCriticalRad,
            range_minimum=-0.0,
            range_maximum=numpy.pi,
            range_step=numpy.pi / 20.0))

        return_value.append(KeyValueMetaBool(
            key=KEY_APRIL_TAG_DEGLITCH,
            value=self._marker_parameters.aprilTagDeglitch))

        return_value.append(KeyValueMetaFloat(
            key=KEY_APRIL_TAG_MAX_LINE_FIT_MSE,
            value=self._marker_parameters.aprilTagMaxLineFitMse,
            range_minimum=0.0,
            range_maximum=512.0,
            range_step=0.01))

        return_value.append(KeyValueMetaInt(
            key=KEY_APRIL_TAG_MAX_N_MAXIMA,
            value=self._marker_parameters.aprilTagMaxNmaxima,
            range_minimum=1,
            range_maximum=100))

        return_value.append(KeyValueMetaInt(
            key=KEY_APRIL_TAG_MIN_CLUSTER_PIXELS,
            value=self._marker_parameters.aprilTagMinClusterPixels,
            range_minimum=0,
            range_maximum=512))

        return_value.append(KeyValueMetaInt(
            key=KEY_APRIL_TAG_MIN_WHITE_BLACK_DIFF,
            value=self._marker_parameters.aprilTagMinWhiteBlackDiff,
            range_minimum=0,
            range_maximum=256))

        return_value.append(KeyValueMetaFloat(
            key=KEY_APRIL_TAG_QUAD_DECIMATE,
            value=self._marker_parameters.aprilTagQuadDecimate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_APRIL_TAG_QUAD_SIGMA,
            value=self._marker_parameters.aprilTagQuadSigma,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        # Note: a relatively recent addition to OpenCV, may not be available in some python versions
        if hasattr(self._marker_parameters, "useAruco3Detection"):
            return_value.append(KeyValueMetaBool(
                key=KEY_USE_ARUCO_3_DETECTION,
                value=self._marker_parameters.useAruco3Detection))

            return_value.append(KeyValueMetaFloat(
                key=KEY_MIN_MARKER_LENGTH_RATIO_ORIGINAL_IMG,
                value=self._marker_parameters.minMarkerLengthRatioOriginalImg,
                range_minimum=0.0,
                range_maximum=1.0,
                range_step=0.01))

            return_value.append(KeyValueMetaInt(
                key=KEY_MIN_SIDE_LENGTH_CANONICAL_IMG,
                value=self._marker_parameters.minSideLengthCanonicalImg,
                range_minimum=1,
                range_maximum=512))

        return return_value

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
        mismatched_keys: list[str] = list()
        key_value: KeyValueSimpleAbstract
        for key_value in parameters:
            if key_value.key == KEY_ADAPTIVE_THRESH_WIN_SIZE_MIN:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.adaptiveThreshWinSizeMin = key_value.value
            elif key_value.key == KEY_ADAPTIVE_THRESH_WIN_SIZE_MAX:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.adaptiveThreshWinSizeMax = key_value.value
            elif key_value.key == KEY_ADAPTIVE_THRESH_WIN_SIZE_STEP:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.adaptiveThreshWinSizeStep = key_value.value
            elif key_value.key == KEY_ADAPTIVE_THRESH_CONSTANT:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.adaptiveThreshConstant = key_value.value
            elif key_value.key == KEY_MIN_MARKER_PERIMETER_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.minMarkerPerimeterRate = key_value.value
            elif key_value.key == KEY_MAX_MARKER_PERIMETER_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.maxMarkerPerimeterRate = key_value.value
            elif key_value.key == KEY_POLYGONAL_APPROX_ACCURACY_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.polygonalApproxAccuracyRate = key_value.value
            elif key_value.key == KEY_MIN_CORNER_DISTANCE_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.minCornerDistanceRate = key_value.value
            elif key_value.key == KEY_MIN_MARKER_DISTANCE_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.minMarkerDistanceRate = key_value.value
            elif key_value.key == KEY_MIN_DISTANCE_TO_BORDER:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.minDistanceToBorder = key_value.value
            elif key_value.key == KEY_MARKER_BORDER_BITS:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.markerBorderBits = key_value.value
            elif key_value.key == KEY_MIN_OTSU_STDDEV:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.minOtsuStdDev = key_value.value
            elif key_value.key == KEY_PERSPECTIVE_REMOVE_PIXEL_PER_CELL:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.perspectiveRemovePixelPerCell = key_value.value
            elif key_value.key == KEY_PERSPECTIVE_REMOVE_IGNORED_MARGIN_PER_CELL:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.perspectiveRemoveIgnoredMarginPerCell = key_value.value
            elif key_value.key == KEY_MAX_ERRONEOUS_BITS_IN_BORDER_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.maxErroneousBitsInBorderRate = key_value.value
            elif key_value.key == KEY_ERROR_CORRECTION_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.errorCorrectionRate = key_value.value
            elif key_value.key == KEY_DETECT_INVERTED_MARKER:
                if not isinstance(key_value, KeyValueSimpleBool):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.detectInvertedMarker = key_value.value
            elif key_value.key == KEY_CORNER_REFINEMENT_METHOD:
                if not isinstance(key_value, KeyValueSimpleString):
                    mismatched_keys.append(key_value.key)
                    continue
                corner_refinement_method: str = key_value.value
                if corner_refinement_method in CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT:
                    # noinspection PyTypeChecker
                    self._marker_parameters.cornerRefinementMethod = \
                        CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT[corner_refinement_method]
                else:
                    raise MCTDetectorRuntimeError(
                        message=f"Failed to find corner refinement method {corner_refinement_method}.")
            elif key_value.key == KEY_CORNER_REFINEMENT_WIN_SIZE:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.cornerRefinementWinSize = key_value.value
            elif key_value.key == KEY_CORNER_REFINEMENT_MAX_ITERATIONS:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.cornerRefinementMaxIterations = key_value.value
            elif key_value.key == KEY_CORNER_REFINEMENT_MIN_ACCURACY:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.cornerRefinementMinAccuracy = key_value.value
            elif key_value.key == KEY_APRIL_TAG_CRITICAL_RAD:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.aprilTagCriticalRad = key_value.value
            elif key_value.key == KEY_APRIL_TAG_DEGLITCH:
                if not isinstance(key_value, KeyValueSimpleBool):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.aprilTagDeglitch = int(key_value.value)
            elif key_value.key == KEY_APRIL_TAG_MAX_LINE_FIT_MSE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.aprilTagMaxLineFitMse = key_value.value
            elif key_value.key == KEY_APRIL_TAG_MAX_N_MAXIMA:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.aprilTagMaxNmaxima = key_value.value
            elif key_value.key == KEY_APRIL_TAG_MIN_CLUSTER_PIXELS:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.aprilTagMinClusterPixels = key_value.value
            elif key_value.key == KEY_APRIL_TAG_MIN_WHITE_BLACK_DIFF:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.aprilTagMinWhiteBlackDiff = key_value.value
            elif key_value.key == KEY_APRIL_TAG_QUAD_DECIMATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.aprilTagQuadDecimate = key_value.value
            elif key_value.key == KEY_APRIL_TAG_QUAD_SIGMA:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.aprilTagQuadSigma = key_value.value
            elif key_value.key == KEY_USE_ARUCO_3_DETECTION:
                if not isinstance(key_value, KeyValueSimpleBool):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.useAruco3Detection = key_value.value
            elif key_value.key == KEY_MIN_MARKER_LENGTH_RATIO_ORIGINAL_IMG:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.minMarkerLengthRatioOriginalImg = key_value.value
            elif key_value.key == KEY_MIN_SIDE_LENGTH_CANONICAL_IMG:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                self._marker_parameters.minSideLengthCanonicalImg = key_value.value
            else:
                mismatched_keys.append(key_value.key)

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

        self._marker_timestamp_utc = datetime.datetime.utcnow()
