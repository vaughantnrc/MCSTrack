from src.common import \
    Annotation, \
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
    Landmark, \
    MathUtils, \
    MCTSerializationError, \
    Target
import cv2.aruco
import logging
import numpy
from typing import Final, get_args, Literal


logger = logging.getLogger(__name__)


# Look at https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
# for documentation on individual parameters

# Adaptive Thresholding
_KEY_ADAPTIVE_THRESH_WIN_SIZE_MIN: Final[str] = "adaptiveThreshWinSizeMin"
_KEY_ADAPTIVE_THRESH_WIN_SIZE_MAX: Final[str] = "adaptiveThreshWinSizeMax"
_KEY_ADAPTIVE_THRESH_WIN_SIZE_STEP: Final[str] = "adaptiveThreshWinSizeStep"
_KEY_ADAPTIVE_THRESH_CONSTANT: Final[str] = "adaptiveThreshConstant"
# Contour Filtering
_KEY_MIN_MARKER_PERIMETER_RATE: Final[str] = "minMarkerPerimeterRate"  # Marker size ratio
_KEY_MAX_MARKER_PERIMETER_RATE: Final[str] = "maxMarkerPerimeterRate"
_KEY_POLYGONAL_APPROX_ACCURACY_RATE: Final[str] = "polygonalApproxAccuracyRate"  # Square tolerance ratio
_KEY_MIN_CORNER_DISTANCE_RATE: Final[str] = "minCornerDistanceRate"  # Corner separation ratio
_KEY_MIN_MARKER_DISTANCE_RATE: Final[str] = "minMarkerDistanceRate"  # Marker separation ratio
_KEY_MIN_DISTANCE_TO_BORDER: Final[str] = "minDistanceToBorder"  # Border distance in pixels
# Bits Extraction
_KEY_MARKER_BORDER_BITS: Final[str] = "markerBorderBits"  # Border width (px)
_KEY_MIN_OTSU_STDDEV: Final[str] = "minOtsuStdDev"  # Minimum brightness stdev
_KEY_PERSPECTIVE_REMOVE_PIXEL_PER_CELL: Final[str] = "perspectiveRemovePixelPerCell"  # Bit Sampling Rate
_KEY_PERSPECTIVE_REMOVE_IGNORED_MARGIN_PER_CELL: Final[str] = "perspectiveRemoveIgnoredMarginPerCell"  # Bit Margin Ratio
# Marker Identification
_KEY_MAX_ERRONEOUS_BITS_IN_BORDER_RATE: Final[str] = "maxErroneousBitsInBorderRate"  # Border Error Rate
_KEY_ERROR_CORRECTION_RATE: Final[str] = "errorCorrectionRate"  # Error Correction Rat
_KEY_DETECT_INVERTED_MARKER: Final[str] = "detectInvertedMarker"
_KEY_CORNER_REFINEMENT_METHOD: Final[str] = "cornerRefinementMethod"
_KEY_CORNER_REFINEMENT_WIN_SIZE: Final[str] = "cornerRefinementWinSize"
_KEY_CORNER_REFINEMENT_MAX_ITERATIONS: Final[str] = "cornerRefinementMaxIterations"
_KEY_CORNER_REFINEMENT_MIN_ACCURACY: Final[str] = "cornerRefinementMinAccuracy"
# April Tag Only
_KEY_APRIL_TAG_CRITICAL_RAD: Final[str] = "aprilTagCriticalRad"
_KEY_APRIL_TAG_DEGLITCH: Final[str] = "aprilTagDeglitch"
_KEY_APRIL_TAG_MAX_LINE_FIT_MSE: Final[str] = "aprilTagMaxLineFitMse"
_KEY_APRIL_TAG_MAX_N_MAXIMA: Final[str] = "aprilTagMaxNmaxima"
_KEY_APRIL_TAG_MIN_CLUSTER_PIXELS: Final[str] = "aprilTagMinClusterPixels"
_KEY_APRIL_TAG_MIN_WHITE_BLACK_DIFF: Final[str] = "aprilTagMinWhiteBlackDiff"
_KEY_APRIL_TAG_QUAD_DECIMATE: Final[str] = "aprilTagQuadDecimate"
_KEY_APRIL_TAG_QUAD_SIGMA: Final[str] = "aprilTagQuadSigma"
# ArUco 3
_KEY_USE_ARUCO_3_DETECTION: Final[str] = "useAruco3Detection"
_KEY_MIN_MARKER_LENGTH_RATIO_ORIGINAL_IMG: Final[str] = "minMarkerLengthRatioOriginalImg"
_KEY_MIN_SIDE_LENGTH_CANONICAL_IMG: Final[str] = "minSideLengthCanonicalImg"


class ArucoOpenCVCommon:
    """
    A "class" to group related static functions and constants, like in a namespace.
    The class itself is not meant to be instantiated.
    """

    def __init__(self):
        raise RuntimeError(f"{__class__.__name__} is not meant to be instantiated.")


    KEY_ADAPTIVE_THRESH_WIN_SIZE_MIN: Final[str] = _KEY_ADAPTIVE_THRESH_WIN_SIZE_MIN
    KEY_ADAPTIVE_THRESH_WIN_SIZE_MAX: Final[str] = _KEY_ADAPTIVE_THRESH_WIN_SIZE_MAX
    KEY_ADAPTIVE_THRESH_WIN_SIZE_STEP: Final[str] = _KEY_ADAPTIVE_THRESH_WIN_SIZE_STEP
    KEY_ADAPTIVE_THRESH_CONSTANT: Final[str] = _KEY_ADAPTIVE_THRESH_CONSTANT
    KEY_MIN_MARKER_PERIMETER_RATE: Final[str] = _KEY_MIN_MARKER_PERIMETER_RATE
    KEY_MAX_MARKER_PERIMETER_RATE: Final[str] = _KEY_MAX_MARKER_PERIMETER_RATE
    KEY_POLYGONAL_APPROX_ACCURACY_RATE: Final[str] = _KEY_POLYGONAL_APPROX_ACCURACY_RATE
    KEY_MIN_CORNER_DISTANCE_RATE: Final[str] = _KEY_MIN_CORNER_DISTANCE_RATE
    KEY_MIN_MARKER_DISTANCE_RATE: Final[str] = _KEY_MIN_MARKER_DISTANCE_RATE
    KEY_MIN_DISTANCE_TO_BORDER: Final[str] = _KEY_MIN_DISTANCE_TO_BORDER
    KEY_MARKER_BORDER_BITS: Final[str] = _KEY_MARKER_BORDER_BITS
    KEY_MIN_OTSU_STDDEV: Final[str] = _KEY_MIN_OTSU_STDDEV
    KEY_PERSPECTIVE_REMOVE_PIXEL_PER_CELL: Final[str] = _KEY_PERSPECTIVE_REMOVE_PIXEL_PER_CELL
    KEY_PERSPECTIVE_REMOVE_IGNORED_MARGIN_PER_CELL: Final[str] = _KEY_PERSPECTIVE_REMOVE_IGNORED_MARGIN_PER_CELL
    KEY_MAX_ERRONEOUS_BITS_IN_BORDER_RATE: Final[str] = _KEY_MAX_ERRONEOUS_BITS_IN_BORDER_RATE
    KEY_ERROR_CORRECTION_RATE: Final[str] = _KEY_ERROR_CORRECTION_RATE
    KEY_DETECT_INVERTED_MARKER: Final[str] = _KEY_DETECT_INVERTED_MARKER
    KEY_CORNER_REFINEMENT_METHOD: Final[str] = _KEY_CORNER_REFINEMENT_METHOD
    KEY_CORNER_REFINEMENT_WIN_SIZE: Final[str] = _KEY_CORNER_REFINEMENT_WIN_SIZE
    KEY_CORNER_REFINEMENT_MAX_ITERATIONS: Final[str] = _KEY_CORNER_REFINEMENT_MAX_ITERATIONS
    KEY_CORNER_REFINEMENT_MIN_ACCURACY: Final[str] = _KEY_CORNER_REFINEMENT_MIN_ACCURACY
    KEY_APRIL_TAG_CRITICAL_RAD: Final[str] = _KEY_APRIL_TAG_CRITICAL_RAD
    KEY_APRIL_TAG_DEGLITCH: Final[str] = _KEY_APRIL_TAG_DEGLITCH
    KEY_APRIL_TAG_MAX_LINE_FIT_MSE: Final[str] = _KEY_APRIL_TAG_MAX_LINE_FIT_MSE
    KEY_APRIL_TAG_MAX_N_MAXIMA: Final[str] = _KEY_APRIL_TAG_MAX_N_MAXIMA
    KEY_APRIL_TAG_MIN_CLUSTER_PIXELS: Final[str] = _KEY_APRIL_TAG_MIN_CLUSTER_PIXELS
    KEY_APRIL_TAG_MIN_WHITE_BLACK_DIFF: Final[str] = _KEY_APRIL_TAG_MIN_WHITE_BLACK_DIFF
    KEY_APRIL_TAG_QUAD_DECIMATE: Final[str] = _KEY_APRIL_TAG_QUAD_DECIMATE
    KEY_APRIL_TAG_QUAD_SIGMA: Final[str] = _KEY_APRIL_TAG_QUAD_SIGMA
    KEY_USE_ARUCO_3_DETECTION: Final[str] = _KEY_USE_ARUCO_3_DETECTION
    KEY_MIN_MARKER_LENGTH_RATIO_ORIGINAL_IMG: Final[str] = _KEY_MIN_MARKER_LENGTH_RATIO_ORIGINAL_IMG
    KEY_MIN_SIDE_LENGTH_CANONICAL_IMG: Final[str] = _KEY_MIN_SIDE_LENGTH_CANONICAL_IMG

    CornerRefinementMethod = Literal["NONE", "SUBPIX", "CONTOUR", "APRILTAG"]
    CORNER_REFINEMENT_METHOD_NONE: Final[str] = 'NONE'
    CORNER_REFINEMENT_METHOD_SUBPIX: Final[str] = 'SUBPIX'
    CORNER_REFINEMENT_METHOD_CONTOUR: Final[str] = 'CONTOUR'
    CORNER_REFINEMENT_METHOD_APRILTAG: Final[str] = 'APRILTAG'

    CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT: dict[CornerRefinementMethod, int] = {
        "NONE": cv2.aruco.CORNER_REFINE_NONE,
        "SUBPIX": cv2.aruco.CORNER_REFINE_SUBPIX,
        "CONTOUR": cv2.aruco.CORNER_REFINE_CONTOUR,
        "APRILTAG": cv2.aruco.CORNER_REFINE_APRILTAG}

    CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT: dict[int, CornerRefinementMethod] = {
        cv2.aruco.CORNER_REFINE_NONE: "NONE",
        cv2.aruco.CORNER_REFINE_SUBPIX: "SUBPIX",
        cv2.aruco.CORNER_REFINE_CONTOUR: "CONTOUR",
        cv2.aruco.CORNER_REFINE_APRILTAG: "APRILTAG"}

    class CharucoBoard:
        dictionary_name: str
        square_count_x: int
        square_count_y: int
        square_size_px: int
        marker_size_px: int
        px_per_mm: float

        def __init__(self):
            self.dictionary_name = "DICT_4X4_100"
            self.square_count_x = 8
            self.square_count_y = 10
            self.square_size_px = 800
            self.marker_size_px = 400
            self.px_per_mm = 40

        def aruco_dictionary(self) -> cv2.aruco.Dictionary:
            if self.dictionary_name != "DICT_4X4_100":
                raise NotImplementedError("Only DICT_4X4_100 is currently implemented")
            aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
            return aruco_dictionary

        def as_target(
            self,
            target_label: str
        ) -> Target:
            """
            Note that the coordinates assume the same axes as get_marker_center_points,
            but the origin is in the center of the board, not the bottom-left corner.
            """
            corner_points: list[list[float]] = self.get_marker_corner_points()
            marker_count: int = len(corner_points) // 4
            landmarks: list[Landmark] = list()
            for marker_index in range(0, marker_count):
                for corner_index in range(0, 4):
                    landmarks.append(Landmark(
                        feature_label=f"{marker_index}{Landmark.RELATION_CHARACTER}{corner_index}",
                        x=corner_points[marker_index*4+corner_index][0],
                        y=corner_points[marker_index*4+corner_index][1],
                        z=corner_points[marker_index*4+corner_index][2]))
            return Target(
                label=target_label,
                landmarks=landmarks)

        def size_px(self) -> tuple[float, float]:
            board_size_x_px = self.square_count_x * self.square_size_px
            board_size_y_px = self.square_count_y * self.square_size_px
            return board_size_x_px, board_size_y_px

        def size_mm(self) -> tuple[float, float]:
            board_size_x_mm = self.square_count_x * self.square_size_px / self.px_per_mm
            board_size_y_mm = self.square_count_y * self.square_size_px / self.px_per_mm
            return board_size_x_mm, board_size_y_mm

        def create_board(self) -> ...:  # type cv2.aruco.CharucoBoard
            charuco_board = cv2.aruco.CharucoBoard(
                size=(self.square_count_x, self.square_count_y),
                squareLength=self.square_size_px,
                markerLength=self.marker_size_px,
                dictionary=self.aruco_dictionary())
            return charuco_board

        def get_marker_center_points(self) -> list[list[float]]:
            """
            Note that the coordinates assume (based on portrait orientation):
            origin: at bottom-left of board
            x-axis: goes right
            y-axis: goes up the page
            z-axis: comes out of the image and toward the viewer
            """
            points = []
            for y in range(self.square_count_y):
                for x in range(self.square_count_x):
                    if (x + y) % 2 == 1:  # Only add the points for the white squares
                        point_x = (x + 0.5) * self.square_size_px / self.px_per_mm
                        point_y = (self.square_count_y - y - 0.5) * self.square_size_px / self.px_per_mm
                        points.append([point_x, point_y, 0.0])
            return points

        def get_marker_corner_points(self) -> list[list[float]]:
            """
            Note that the coordinates assume the same axes as get_marker_center_points,
            but the origin is in the center of the board, not the bottom-left corner.
            """
            points = []
            marker_size_mm: float = self.marker_size_px / self.px_per_mm
            square_size_mm: float = self.square_size_px / self.px_per_mm
            for y_sq in range(self.square_count_y):
                for x_sq in range(self.square_count_x):
                    if (x_sq + y_sq) % 2 == 1:  # Only add the points for the white squares
                        x_sq_centered: float = x_sq - (self.square_count_x / 2.0)
                        y_sq_centered: float = y_sq - (self.square_count_y / 2.0)
                        for corner_index in range(0, 4):
                            x_mm: float = (x_sq_centered + 0.5) * square_size_mm
                            if corner_index == 0 or corner_index == 3:
                                x_mm -= (marker_size_mm / 2.0)
                            else:
                                x_mm += (marker_size_mm / 2.0)
                            y_mm: float = (-(y_sq_centered + 0.5)) * square_size_mm
                            if corner_index == 0 or corner_index == 1:
                                y_mm += (marker_size_mm / 2.0)
                            else:
                                y_mm -= (marker_size_mm / 2.0)
                            z_mm: float = 0.0
                            points.append([x_mm, y_mm, z_mm])
            return points

        def get_marker_ids(self) -> list[int]:
            num_markers = self.square_count_x * self.square_count_y // 2
            return list(range(num_markers))

    @staticmethod
    def annotations_from_greyscale_image(
        aruco_detector_parameters: cv2.aruco.DetectorParameters,
        aruco_dictionary: cv2.aruco.Dictionary,
        image_greyscale: numpy.ndarray
    ) -> tuple[list[Annotation], list[Annotation]]:
        (detected_corner_points_raw, detected_dictionary_indices, rejected_corner_points_raw) = cv2.aruco.detectMarkers(
            image=image_greyscale,
            dictionary=aruco_dictionary,
            parameters=aruco_detector_parameters)

        detected_annotations: list[Annotation] = list()
        # note: detected_indices is (inconsistently) None sometimes if nothing is detected
        if detected_dictionary_indices is not None and len(detected_dictionary_indices) > 0:
            detected_count = detected_dictionary_indices.size
            # Shape of some output was previously observed to (also) be inconsistent... make it consistent here:
            detected_corner_points_px = numpy.array(detected_corner_points_raw).reshape((detected_count, 4, 2))
            detected_dictionary_indices = list(detected_dictionary_indices.reshape(detected_count))
            for detected_index, detected_id in enumerate(detected_dictionary_indices):
                for corner_index in range(4):
                    detected_label: str = f"{detected_id}{Annotation.RELATION_CHARACTER}{corner_index}"
                    detected_annotations.append(Annotation(
                        feature_label=detected_label,
                        x_px=float(detected_corner_points_px[detected_index][corner_index][0]),
                        y_px=float(detected_corner_points_px[detected_index][corner_index][1])))

        rejected_annotations: list[Annotation] = list()
        if rejected_corner_points_raw:
            rejected_corner_points_px = numpy.array(rejected_corner_points_raw).reshape((-1, 4, 2))
            for rejected_index in range(rejected_corner_points_px.shape[0]):
                for corner_index in range(4):
                    rejected_annotations.append(Annotation(
                        feature_label=Annotation.UNIDENTIFIED_LABEL,
                        x_px=float(rejected_corner_points_px[rejected_index][corner_index][0]),
                        y_px=float(rejected_corner_points_px[rejected_index][corner_index][1])))

        return detected_annotations, rejected_annotations

    @staticmethod
    def assign_aruco_detection_parameters_to_key_value_list(
        detection_parameters: ...  # cv2.aruco.DetectionParameters
    ) -> list[KeyValueMetaAny]:

        return_value: list[KeyValueMetaAny] = list()

        return_value.append(KeyValueMetaInt(
            key=_KEY_ADAPTIVE_THRESH_WIN_SIZE_MIN,
            value=detection_parameters.adaptiveThreshWinSizeMin,
            range_minimum=1,
            range_maximum=99))

        return_value.append(KeyValueMetaInt(
            key=_KEY_ADAPTIVE_THRESH_WIN_SIZE_MAX,
            value=detection_parameters.adaptiveThreshWinSizeMax,
            range_minimum=1,
            range_maximum=99))

        return_value.append(KeyValueMetaInt(
            key=_KEY_ADAPTIVE_THRESH_WIN_SIZE_STEP,
            value=detection_parameters.adaptiveThreshWinSizeStep,
            range_minimum=1,
            range_maximum=99,
            range_step=2))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_ADAPTIVE_THRESH_CONSTANT,
            value=detection_parameters.adaptiveThreshConstant,
            range_minimum=-255.0,
            range_maximum=255.0,
            range_step=1.0))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_MIN_MARKER_PERIMETER_RATE,
            value=detection_parameters.minMarkerPerimeterRate,
            range_minimum=0,
            range_maximum=8.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_MAX_MARKER_PERIMETER_RATE,
            value=detection_parameters.maxMarkerPerimeterRate,
            range_minimum=0.0,
            range_maximum=8.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_POLYGONAL_APPROX_ACCURACY_RATE,
            value=detection_parameters.polygonalApproxAccuracyRate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_MIN_CORNER_DISTANCE_RATE,
            value=detection_parameters.minCornerDistanceRate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_MIN_MARKER_DISTANCE_RATE,
            value=detection_parameters.minMarkerDistanceRate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaInt(
            key=_KEY_MIN_DISTANCE_TO_BORDER,
            value=detection_parameters.minDistanceToBorder,
            range_minimum=0,
            range_maximum=512))

        return_value.append(KeyValueMetaInt(
            key=_KEY_MARKER_BORDER_BITS,
            value=detection_parameters.markerBorderBits,
            range_minimum=1,
            range_maximum=9))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_MIN_OTSU_STDDEV,
            value=detection_parameters.minOtsuStdDev,
            range_minimum=0.0,
            range_maximum=256.0,
            range_step=1.0))

        return_value.append(KeyValueMetaInt(
            key=_KEY_PERSPECTIVE_REMOVE_PIXEL_PER_CELL,
            value=detection_parameters.perspectiveRemovePixelPerCell,
            range_minimum=1,
            range_maximum=20))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_PERSPECTIVE_REMOVE_IGNORED_MARGIN_PER_CELL,
            value=detection_parameters.perspectiveRemoveIgnoredMarginPerCell,
            range_minimum=0.0,
            range_maximum=0.5,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_MAX_ERRONEOUS_BITS_IN_BORDER_RATE,
            value=detection_parameters.maxErroneousBitsInBorderRate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_ERROR_CORRECTION_RATE,
            value=detection_parameters.errorCorrectionRate,
            range_minimum=-0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaBool(
            key=_KEY_DETECT_INVERTED_MARKER,
            value=detection_parameters.detectInvertedMarker))

        if detection_parameters.cornerRefinementMethod not in \
           ArucoOpenCVCommon.CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT:
            message: str = f"Corner refinement method appears to be set to an invalid value: " \
                           f"{detection_parameters.corner_refinement_method}."
            logger.error(message)
            raise MCTSerializationError(message=message)
        corner_refinement_method_text: ArucoOpenCVCommon.CornerRefinementMethod = \
            ArucoOpenCVCommon.CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT[
                detection_parameters.cornerRefinementMethod]
        return_value.append(KeyValueMetaEnum(
            key=_KEY_CORNER_REFINEMENT_METHOD,
            value=corner_refinement_method_text,
            allowable_values=list(get_args(ArucoOpenCVCommon.CornerRefinementMethod))))

        return_value.append(KeyValueMetaInt(
            key=_KEY_CORNER_REFINEMENT_WIN_SIZE,
            value=detection_parameters.cornerRefinementWinSize,
            range_minimum=1,
            range_maximum=9))

        return_value.append(KeyValueMetaInt(
            key=_KEY_CORNER_REFINEMENT_MAX_ITERATIONS,
            value=detection_parameters.cornerRefinementMaxIterations,
            range_minimum=1,
            range_maximum=100))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_CORNER_REFINEMENT_MIN_ACCURACY,
            value=detection_parameters.cornerRefinementMinAccuracy,
            range_minimum=0.0,
            range_maximum=5.0,
            range_step=0.1))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_APRIL_TAG_CRITICAL_RAD,
            value=detection_parameters.aprilTagCriticalRad,
            range_minimum=-0.0,
            range_maximum=numpy.pi,
            range_step=numpy.pi / 20.0))

        return_value.append(KeyValueMetaBool(
            key=_KEY_APRIL_TAG_DEGLITCH,
            value=detection_parameters.aprilTagDeglitch))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_APRIL_TAG_MAX_LINE_FIT_MSE,
            value=detection_parameters.aprilTagMaxLineFitMse,
            range_minimum=0.0,
            range_maximum=512.0,
            range_step=0.01))

        return_value.append(KeyValueMetaInt(
            key=_KEY_APRIL_TAG_MAX_N_MAXIMA,
            value=detection_parameters.aprilTagMaxNmaxima,
            range_minimum=1,
            range_maximum=100))

        return_value.append(KeyValueMetaInt(
            key=_KEY_APRIL_TAG_MIN_CLUSTER_PIXELS,
            value=detection_parameters.aprilTagMinClusterPixels,
            range_minimum=0,
            range_maximum=512))

        return_value.append(KeyValueMetaInt(
            key=_KEY_APRIL_TAG_MIN_WHITE_BLACK_DIFF,
            value=detection_parameters.aprilTagMinWhiteBlackDiff,
            range_minimum=0,
            range_maximum=256))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_APRIL_TAG_QUAD_DECIMATE,
            value=detection_parameters.aprilTagQuadDecimate,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        return_value.append(KeyValueMetaFloat(
            key=_KEY_APRIL_TAG_QUAD_SIGMA,
            value=detection_parameters.aprilTagQuadSigma,
            range_minimum=0.0,
            range_maximum=1.0,
            range_step=0.01))

        # Note: a relatively recent addition to OpenCV, may not be available in some python versions
        if hasattr(detection_parameters, "useAruco3Detection"):
            return_value.append(KeyValueMetaBool(
                key=_KEY_USE_ARUCO_3_DETECTION,
                value=detection_parameters.useAruco3Detection))

            return_value.append(KeyValueMetaFloat(
                key=_KEY_MIN_MARKER_LENGTH_RATIO_ORIGINAL_IMG,
                value=detection_parameters.minMarkerLengthRatioOriginalImg,
                range_minimum=0.0,
                range_maximum=1.0,
                range_step=0.01))

            return_value.append(KeyValueMetaInt(
                key=_KEY_MIN_SIDE_LENGTH_CANONICAL_IMG,
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
            if key_value.key == _KEY_ADAPTIVE_THRESH_WIN_SIZE_MIN:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.adaptiveThreshWinSizeMin = key_value.value
            elif key_value.key == _KEY_ADAPTIVE_THRESH_WIN_SIZE_MAX:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.adaptiveThreshWinSizeMax = key_value.value
            elif key_value.key == _KEY_ADAPTIVE_THRESH_WIN_SIZE_STEP:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.adaptiveThreshWinSizeStep = key_value.value
            elif key_value.key == _KEY_ADAPTIVE_THRESH_CONSTANT:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.adaptiveThreshConstant = key_value.value
            elif key_value.key == _KEY_MIN_MARKER_PERIMETER_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minMarkerPerimeterRate = key_value.value
            elif key_value.key == _KEY_MAX_MARKER_PERIMETER_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.maxMarkerPerimeterRate = key_value.value
            elif key_value.key == _KEY_POLYGONAL_APPROX_ACCURACY_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.polygonalApproxAccuracyRate = key_value.value
            elif key_value.key == _KEY_MIN_CORNER_DISTANCE_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minCornerDistanceRate = key_value.value
            elif key_value.key == _KEY_MIN_MARKER_DISTANCE_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minMarkerDistanceRate = key_value.value
            elif key_value.key == _KEY_MIN_DISTANCE_TO_BORDER:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minDistanceToBorder = key_value.value
            elif key_value.key == _KEY_MARKER_BORDER_BITS:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.markerBorderBits = key_value.value
            elif key_value.key == _KEY_MIN_OTSU_STDDEV:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minOtsuStdDev = key_value.value
            elif key_value.key == _KEY_PERSPECTIVE_REMOVE_PIXEL_PER_CELL:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.perspectiveRemovePixelPerCell = key_value.value
            elif key_value.key == _KEY_PERSPECTIVE_REMOVE_IGNORED_MARGIN_PER_CELL:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.perspectiveRemoveIgnoredMarginPerCell = key_value.value
            elif key_value.key == _KEY_MAX_ERRONEOUS_BITS_IN_BORDER_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.maxErroneousBitsInBorderRate = key_value.value
            elif key_value.key == _KEY_ERROR_CORRECTION_RATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.errorCorrectionRate = key_value.value
            elif key_value.key == _KEY_DETECT_INVERTED_MARKER:
                if not isinstance(key_value, KeyValueSimpleBool):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.detectInvertedMarker = key_value.value
            elif key_value.key == _KEY_CORNER_REFINEMENT_METHOD:
                if not isinstance(key_value, KeyValueSimpleString):
                    mismatched_keys.append(key_value.key)
                    continue
                corner_refinement_method: str = key_value.value
                if corner_refinement_method in ArucoOpenCVCommon.CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT:
                    # noinspection PyTypeChecker
                    detection_parameters.cornerRefinementMethod = \
                        ArucoOpenCVCommon.CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT[
                            corner_refinement_method]
                else:
                    raise MCTSerializationError(
                        message=f"Failed to find corner refinement method {corner_refinement_method}.")
            elif key_value.key == _KEY_CORNER_REFINEMENT_WIN_SIZE:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.cornerRefinementWinSize = key_value.value
            elif key_value.key == _KEY_CORNER_REFINEMENT_MAX_ITERATIONS:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.cornerRefinementMaxIterations = key_value.value
            elif key_value.key == _KEY_CORNER_REFINEMENT_MIN_ACCURACY:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.cornerRefinementMinAccuracy = key_value.value
            elif key_value.key == _KEY_APRIL_TAG_CRITICAL_RAD:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagCriticalRad = key_value.value
            elif key_value.key == _KEY_APRIL_TAG_DEGLITCH:
                if not isinstance(key_value, KeyValueSimpleBool):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagDeglitch = int(key_value.value)
            elif key_value.key == _KEY_APRIL_TAG_MAX_LINE_FIT_MSE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagMaxLineFitMse = key_value.value
            elif key_value.key == _KEY_APRIL_TAG_MAX_N_MAXIMA:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagMaxNmaxima = key_value.value
            elif key_value.key == _KEY_APRIL_TAG_MIN_CLUSTER_PIXELS:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagMinClusterPixels = key_value.value
            elif key_value.key == _KEY_APRIL_TAG_MIN_WHITE_BLACK_DIFF:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagMinWhiteBlackDiff = key_value.value
            elif key_value.key == _KEY_APRIL_TAG_QUAD_DECIMATE:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagQuadDecimate = key_value.value
            elif key_value.key == _KEY_APRIL_TAG_QUAD_SIGMA:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.aprilTagQuadSigma = key_value.value
            elif key_value.key == _KEY_USE_ARUCO_3_DETECTION:
                if not isinstance(key_value, KeyValueSimpleBool):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.useAruco3Detection = key_value.value
            elif key_value.key == _KEY_MIN_MARKER_LENGTH_RATIO_ORIGINAL_IMG:
                if not isinstance(key_value, KeyValueSimpleFloat):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minMarkerLengthRatioOriginalImg = key_value.value
            elif key_value.key == _KEY_MIN_SIDE_LENGTH_CANONICAL_IMG:
                if not isinstance(key_value, KeyValueSimpleInt):
                    mismatched_keys.append(key_value.key)
                    continue
                detection_parameters.minSideLengthCanonicalImg = key_value.value
            else:
                mismatched_keys.append(key_value.key)
        return mismatched_keys

    @staticmethod
    def target_from_marker_parameters(
        base_label : str,
        marker_size: float
    ) -> Target:
        """
        :param base_label: Should correspond to the index of the ArUco marker in the dictionary
        :param marker_size:
        """
        corner_points: list[list[float]] = MathUtils.square_marker_corner_points(marker_size=marker_size)
        landmarks: list[Landmark] = [
            Landmark(
                feature_label=f"{base_label}{Landmark.RELATION_CHARACTER}{corner_index}",
                x=corner_point[0], y=corner_point[1], z=corner_point[2])
            for corner_index, corner_point in enumerate(corner_points)]
        target: Target = Target(label=base_label, landmarks=landmarks)
        return target
