from src.common import \
    Annotator, \
    MCTAnnotatorRuntimeError, \
    StatusMessageSource
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


class ArucoOpenCVAnnotator(Annotator):

    _aruco_dictionary: Any | None  # created by OpenCV, type cv2.aruco.Dictionary
    _aruco_parameters: Any  # created by OpenCV, type cv2.aruco.DetectorParameters
    _snapshots_identified: list[MarkerSnapshot]  # Markers that are determined to be valid, and are identified
    _snapshots_unidentified: list[MarkerSnapshot]  # Looked at first like markers but got filtered out
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

    @staticmethod
    def assign_aruco_detection_parameters_to_key_value_list(
            detection_parameters: ...  # cv2.aruco.DetectionParameters
    ) -> list[KeyValueMetaAny]:

        return_value: list[KeyValueMetaAny] = list()

        return_value.append(KeyValueMetaInt(
            key=KEY_ADAPTIVE_THRESH_WIN_SIZE_MIN,
            value=detection_parameters.adaptiveThreshWinSizeMin,
            range_minimum=1,
            range_maximum=99))

        return_value.append(KeyValueMetaInt(
            key=KEY_ADAPTIVE_THRESH_WIN_SIZE_MAX,
            value=detection_parameters.adaptiveThreshWinSizeMax,
            range_minimum=1,
            range_maximum=99))

        return_value.append(KeyValueMetaInt(
            key=KEY_ADAPTIVE_THRESH_WIN_SIZE_STEP,
            value=detection_parameters.adaptiveThreshWinSizeStep,
            range_minimum=1,
            range_maximum=99,
            range_step=2))

        return_value.append(KeyValueMetaFloat(
            key=KEY_ADAPTIVE_THRESH_CONSTANT,
            value=detection_parameters.adaptiveThreshConstant,
            range_minimum=-255.0,
            range_maximum=255.0,
            range_step=1.0))

        return_value.append(KeyValueMetaFloat(
            key=KEY_MIN_MARKER_PERIMETER_RATE,
            value=detection_parameters.minMarkerPerimeterRate,
            range_minimum=0,
            range_maximum=8.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_MAX_MARKER_PERIMETER_RATE,
            value=detection_parameters.maxMarkerPerimeterRate,
            range_minimum=0.0,
            range_maximum=8.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_POLYGONAL_APPROX_ACCURACY_RATE,
            value=detection_parameters.polygonalApproxAccuracyRate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_MIN_CORNER_DISTANCE_RATE,
            value=detection_parameters.minCornerDistanceRate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_MIN_MARKER_DISTANCE_RATE,
            value=detection_parameters.minMarkerDistanceRate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaInt(
            key=KEY_MIN_DISTANCE_TO_BORDER,
            value=detection_parameters.minDistanceToBorder,
            range_minimum=0,
            range_maximum=512))

        return_value.append(KeyValueMetaInt(
            key=KEY_MARKER_BORDER_BITS,
            value=detection_parameters.markerBorderBits,
            range_minimum=1,
            range_maximum=9))

        return_value.append(KeyValueMetaFloat(
            key=KEY_MIN_OTSU_STDDEV,
            value=detection_parameters.minOtsuStdDev,
            range_minimum=0.0,
            range_maximum=256.0,
            range_step=1.0))

        return_value.append(KeyValueMetaInt(
            key=KEY_PERSPECTIVE_REMOVE_PIXEL_PER_CELL,
            value=detection_parameters.perspectiveRemovePixelPerCell,
            range_minimum=1,
            range_maximum=20))

        return_value.append(KeyValueMetaFloat(
            key=KEY_PERSPECTIVE_REMOVE_IGNORED_MARGIN_PER_CELL,
            value=detection_parameters.perspectiveRemoveIgnoredMarginPerCell,
            range_minimum=0.0,
            range_maximum=0.5,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_MAX_ERRONEOUS_BITS_IN_BORDER_RATE,
            value=detection_parameters.maxErroneousBitsInBorderRate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_ERROR_CORRECTION_RATE,
            value=detection_parameters.errorCorrectionRate,
            range_minimum=-0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaBool(
            key=KEY_DETECT_INVERTED_MARKER,
            value=detection_parameters.detectInvertedMarker))

        if detection_parameters.cornerRefinementMethod not in CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT:
            message: str = f"Corner refinement method appears to be set to an invalid value: " \
                           f"{detection_parameters.corner_refinement_method}."
            logger.error(message)
            raise MCTAnnotatorRuntimeError(message=message)
        corner_refinement_method_text: CornerRefinementMethod = \
            CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT[detection_parameters.cornerRefinementMethod]
        return_value.append(KeyValueMetaEnum(
            key=KEY_CORNER_REFINEMENT_METHOD,
            value=corner_refinement_method_text,
            allowable_values=get_args(CornerRefinementMethod)))

        return_value.append(KeyValueMetaInt(
            key=KEY_CORNER_REFINEMENT_WIN_SIZE,
            value=detection_parameters.cornerRefinementWinSize,
            range_minimum=1,
            range_maximum=9))

        return_value.append(KeyValueMetaInt(
            key=KEY_CORNER_REFINEMENT_MAX_ITERATIONS,
            value=detection_parameters.cornerRefinementMaxIterations,
            range_minimum=1,
            range_maximum=100))

        return_value.append(KeyValueMetaFloat(
            key=KEY_CORNER_REFINEMENT_MIN_ACCURACY,
            value=detection_parameters.cornerRefinementMinAccuracy,
            range_minimum=0.0,
            range_maximum=5.0,
            range_step=0.1))

        return_value.append(KeyValueMetaFloat(
            key=KEY_APRIL_TAG_CRITICAL_RAD,
            value=detection_parameters.aprilTagCriticalRad,
            range_minimum=-0.0,
            range_maximum=numpy.pi,
            range_step=numpy.pi / 20.0))

        return_value.append(KeyValueMetaBool(
            key=KEY_APRIL_TAG_DEGLITCH,
            value=detection_parameters.aprilTagDeglitch))

        return_value.append(KeyValueMetaFloat(
            key=KEY_APRIL_TAG_MAX_LINE_FIT_MSE,
            value=detection_parameters.aprilTagMaxLineFitMse,
            range_minimum=0.0,
            range_maximum=512.0,
            range_step=0.01))

        return_value.append(KeyValueMetaInt(
            key=KEY_APRIL_TAG_MAX_N_MAXIMA,
            value=detection_parameters.aprilTagMaxNmaxima,
            range_minimum=1,
            range_maximum=100))

        return_value.append(KeyValueMetaInt(
            key=KEY_APRIL_TAG_MIN_CLUSTER_PIXELS,
            value=detection_parameters.aprilTagMinClusterPixels,
            range_minimum=0,
            range_maximum=512))

        return_value.append(KeyValueMetaInt(
            key=KEY_APRIL_TAG_MIN_WHITE_BLACK_DIFF,
            value=detection_parameters.aprilTagMinWhiteBlackDiff,
            range_minimum=0,
            range_maximum=256))

        return_value.append(KeyValueMetaFloat(
            key=KEY_APRIL_TAG_QUAD_DECIMATE,
            value=detection_parameters.aprilTagQuadDecimate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=KEY_APRIL_TAG_QUAD_SIGMA,
            value=detection_parameters.aprilTagQuadSigma,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        # Note: a relatively recent addition to OpenCV, may not be available in some python versions
        if hasattr(detection_parameters, "useAruco3Detection"):
            return_value.append(KeyValueMetaBool(
                key=KEY_USE_ARUCO_3_DETECTION,
                value=detection_parameters.useAruco3Detection))

            return_value.append(KeyValueMetaFloat(
                key=KEY_MIN_MARKER_LENGTH_RATIO_ORIGINAL_IMG,
                value=detection_parameters.minMarkerLengthRatioOriginalImg,
                range_minimum=0.0,
                range_maximum=1.0,
                range_step=0.01))

            return_value.append(KeyValueMetaInt(
                key=KEY_MIN_SIDE_LENGTH_CANONICAL_IMG,
                value=detection_parameters.minSideLengthCanonicalImg,
                range_minimum=1,
                range_maximum=512))

        return return_value

    @staticmethod
    def assign_key_value_list_to_aruco_detection_parameters(
            detection_parameters: ...,  # cv2.aruco.DetectionParameters
            key_value_list: list[KeyValueSimpleAny]
    ) -> list[str]:
        """
        Returns list of mismatched keys
        """
        mismatched_keys: list[str] = list()
        key_value: KeyValueSimpleAbstract
        for key_value in key_value_list:
            if key_value.key == KEY_ADAPTIVE_THRESH_WIN_SIZE_MIN:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.adaptiveThreshWinSizeMin = key_value.value
            elif key_value.key == KEY_ADAPTIVE_THRESH_WIN_SIZE_MAX:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.adaptiveThreshWinSizeMax = key_value.value
            elif key_value.key == KEY_ADAPTIVE_THRESH_WIN_SIZE_STEP:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.adaptiveThreshWinSizeStep = key_value.value
            elif key_value.key == KEY_ADAPTIVE_THRESH_CONSTANT:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.adaptiveThreshConstant = key_value.value
            elif key_value.key == KEY_MIN_MARKER_PERIMETER_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minMarkerPerimeterRate = key_value.value
            elif key_value.key == KEY_MAX_MARKER_PERIMETER_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.maxMarkerPerimeterRate = key_value.value
            elif key_value.key == KEY_POLYGONAL_APPROX_ACCURACY_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.polygonalApproxAccuracyRate = key_value.value
            elif key_value.key == KEY_MIN_CORNER_DISTANCE_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minCornerDistanceRate = key_value.value
            elif key_value.key == KEY_MIN_MARKER_DISTANCE_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minMarkerDistanceRate = key_value.value
            elif key_value.key == KEY_MIN_DISTANCE_TO_BORDER:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minDistanceToBorder = key_value.value
            elif key_value.key == KEY_MARKER_BORDER_BITS:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.markerBorderBits = key_value.value
            elif key_value.key == KEY_MIN_OTSU_STDDEV:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minOtsuStdDev = key_value.value
            elif key_value.key == KEY_PERSPECTIVE_REMOVE_PIXEL_PER_CELL:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.perspectiveRemovePixelPerCell = key_value.value
            elif key_value.key == KEY_PERSPECTIVE_REMOVE_IGNORED_MARGIN_PER_CELL:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.perspectiveRemoveIgnoredMarginPerCell = key_value.value
            elif key_value.key == KEY_MAX_ERRONEOUS_BITS_IN_BORDER_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.maxErroneousBitsInBorderRate = key_value.value
            elif key_value.key == KEY_ERROR_CORRECTION_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.errorCorrectionRate = key_value.value
            elif key_value.key == KEY_DETECT_INVERTED_MARKER:
                if not isinstance(key_value, KeyValueSimpleBool):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.detectInvertedMarker = key_value.value
            elif key_value.key == KEY_CORNER_REFINEMENT_METHOD:
                if not isinstance(key_value, KeyValueSimpleString):
                    mismatched_keys.append(key_value.key)
                    continue
                corner_refinement_method: str = key_value.value
                if corner_refinement_method in CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT:
                    # noinspection PyTypeChecker
                    detection_parameters.cornerRefinementMethod = \
                        CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT[corner_refinement_method]
                else:
                    raise MCTAnnotatorRuntimeError(
                        message=f"Failed to find corner refinement method {corner_refinement_method}.")
            elif key_value.key == KEY_CORNER_REFINEMENT_WIN_SIZE:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.cornerRefinementWinSize = key_value.value
            elif key_value.key == KEY_CORNER_REFINEMENT_MAX_ITERATIONS:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.cornerRefinementMaxIterations = key_value.value
            elif key_value.key == KEY_CORNER_REFINEMENT_MIN_ACCURACY:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.cornerRefinementMinAccuracy = key_value.value
            elif key_value.key == KEY_APRIL_TAG_CRITICAL_RAD:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagCriticalRad = key_value.value
            elif key_value.key == KEY_APRIL_TAG_DEGLITCH:
                if not isinstance(key_value, KeyValueSimpleBool):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagDeglitch = int(key_value.value)
            elif key_value.key == KEY_APRIL_TAG_MAX_LINE_FIT_MSE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagMaxLineFitMse = key_value.value
            elif key_value.key == KEY_APRIL_TAG_MAX_N_MAXIMA:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagMaxNmaxima = key_value.value
            elif key_value.key == KEY_APRIL_TAG_MIN_CLUSTER_PIXELS:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagMinClusterPixels = key_value.value
            elif key_value.key == KEY_APRIL_TAG_MIN_WHITE_BLACK_DIFF:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagMinWhiteBlackDiff = key_value.value
            elif key_value.key == KEY_APRIL_TAG_QUAD_DECIMATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagQuadDecimate = key_value.value
            elif key_value.key == KEY_APRIL_TAG_QUAD_SIGMA:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagQuadSigma = key_value.value
            elif key_value.key == KEY_USE_ARUCO_3_DETECTION:
                if not isinstance(key_value, KeyValueSimpleBool):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.useAruco3Detection = key_value.value
            elif key_value.key == KEY_MIN_MARKER_LENGTH_RATIO_ORIGINAL_IMG:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minMarkerLengthRatioOriginalImg = key_value.value
            elif key_value.key == KEY_MIN_SIDE_LENGTH_CANONICAL_IMG:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minSideLengthCanonicalImg = key_value.value
            else:
                mismatched_keys.append(key_value.key)
        return mismatched_keys

    def get_changed_timestamp(self) -> datetime.datetime:
        return self._update_timestamp_utc

    def get_markers_detected(self) -> list[MarkerSnapshot]:
        return self._snapshots_identified

    def get_markers_rejected(self) -> list[MarkerSnapshot]:
        return self._snapshots_unidentified

    def get_parameters(self) -> list[KeyValueMetaAny]:
        return self.assign_aruco_detection_parameters_to_key_value_list(self._aruco_parameters)

    @staticmethod
    def get_type_identifier() -> str:
        return "aruco_opencv"

    @staticmethod
    def _corner_image_point_list_from_embedded_list(
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
        mismatched_keys: list[str] = self.assign_key_value_list_to_aruco_detection_parameters(
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
                detected_label: str = str(detected_id)
                corner_image_points_px = detected_corner_points_px[detected_index]
                corner_image_points: list[MarkerCornerImagePoint] = \
                    self._corner_image_point_list_from_embedded_list(
                        corner_image_points_px=corner_image_points_px.tolist())
                self._snapshots_identified.append(MarkerSnapshot(
                    label=detected_label,
                    corner_image_points=corner_image_points))

        self._snapshots_unidentified = list()
        if rejected_corner_points_raw:
            rejected_corner_points_px = numpy.array(rejected_corner_points_raw).reshape((-1, 4, 2))
            for rejected_index in range(rejected_corner_points_px.shape[0]):
                corner_image_points_px = rejected_corner_points_px[rejected_index]
                corner_image_points: list[MarkerCornerImagePoint] = \
                    self._corner_image_point_list_from_embedded_list(
                        corner_image_points_px=corner_image_points_px.tolist())
                self._snapshots_unidentified.append(MarkerSnapshot(
                    label=f"unknown",
                    corner_image_points=corner_image_points))

        self._update_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)
