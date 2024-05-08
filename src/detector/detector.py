from src.detector.api import \
    GetCaptureDeviceRequest, \
    GetCaptureDeviceResponse, \
    GetCaptureImageRequest, \
    GetCaptureImageResponse, \
    GetCapturePropertiesRequest, \
    GetCapturePropertiesResponse, \
    GetDetectionParametersRequest, \
    GetDetectionParametersResponse, \
    GetMarkerSnapshotsRequest, \
    GetMarkerSnapshotsResponse, \
    SetCaptureDeviceRequest, \
    SetCapturePropertiesRequest, \
    SetDetectionParametersRequest, \
    StartCaptureRequest, \
    StopCaptureRequest
from src.detector.fileio import DetectorConfiguration
from src.common import \
    EmptyResponse, \
    ErrorResponse, \
    get_kwarg, \
    MCastComponent, \
    MCastRequest, \
    MCastResponse
from src.common.structures import \
    CornerRefinementMethod, \
    CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT, \
    CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT, \
    DetectionParameters, \
    DetectorStatus, \
    MarkerSnapshot, \
    MarkerCornerImagePoint
import base64
import cv2.aruco
import datetime
import logging
import numpy
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Detector(MCastComponent):

    _detector_configuration: DetectorConfiguration

    _capture: cv2.VideoCapture | None
    _captured_image: numpy.ndarray | None
    _captured_timestamp_utc: datetime.datetime
    _capture_device_id: str | int

    _marker_dictionary: Any | None  # created by OpenCV, type cv2.aruco.Dictionary
    _marker_parameters: Any  # created by OpenCV, type cv2.aruco.DetectorParameters
    _marker_label_reverse_dictionary: dict[int, str]
    _marker_detected_snapshots: list[MarkerSnapshot]
    _marker_timestamp_utc: datetime.datetime

    _frame_count: int

    _status: DetectorStatus  # internal bookkeeping

    def __init__(
        self,
        detector_configuration: DetectorConfiguration
    ):
        super().__init__(
            status_source_label=detector_configuration.serial_identifier,
            send_status_messages_to_logger=True)

        self._detector_configuration = detector_configuration

        self._capture = None
        self._captured_image = None
        self._captured_timestamp_utc = datetime.datetime.min
        self._capture_device_id = self._detector_configuration.camera_connection.usb_id

        # TODO: Maybe these are best left at None to begin
        self._marker_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        self._marker_parameters = cv2.aruco.DetectorParameters_create()
        self._marker_label_reverse_dictionary = dict()
        self._marker_detected_snapshots = list()  # Markers that are determined to be valid, and are identified
        self._marker_rejected_snapshots = list()  # Things that looked at first like markers but got later filtered out
        self._marker_timestamp_utc = datetime.datetime.min

        self._frame_count = 0

        self._status = DetectorStatus()
        self._status.capture_status = DetectorStatus.Capture.STOPPED
        self._status.marker_status = DetectorStatus.Marker.STOPPED

        # TODO: DEBUGGING
        self._status.marker_status = DetectorStatus.Marker.RUNNING

    def __del__(self):
        if self._capture is not None:
            self._capture.release()

    async def internal_update(self):
        if self._status.capture_status == DetectorStatus.Capture.RUNNING:
            self.internal_update_capture()
        if self._status.marker_status == DetectorStatus.Marker.RUNNING and \
           self._captured_timestamp_utc > self._marker_timestamp_utc:
            self.internal_update_markers()
        self._frame_count += 1
        if self._frame_count % 1000 == 0:
            print(f"Update count: {self._frame_count}")

    def internal_update_capture(self):
        grabbed_frame: bool
        grabbed_frame = self._capture.grab()
        if not grabbed_frame:
            message: str = "Failed to grab frame."
            self._status.capture_errors.append(message)
            self._status.capture_status = DetectorStatus.Capture.FAILURE
            self.add_status_message(severity="error", message=message)
            return

        retrieved_frame: bool
        retrieved_frame, self._captured_image = self._capture.retrieve()
        if not retrieved_frame:
            message: str = "Failed to retrieve frame."
            logger.error(message)
            self._status.capture_errors.append(message)
            self._status.capture_status = DetectorStatus.Capture.FAILURE
            return

        logger.debug("TEST")
        self._captured_timestamp_utc = datetime.datetime.utcnow()

    def internal_update_markers(self):
        if self._marker_dictionary is None:
            message: str = "No marker dictionary has been set."
            logger.error(message)
            self._status.marker_errors.append(message)
            self._status.marker_status = DetectorStatus.Capture.FAILURE
            return

        # TODO: Add means of modifying parameters, probably another endpoint to do so

        image_greyscale = cv2.cvtColor(self._captured_image, cv2.COLOR_RGB2GRAY)
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
                        logger.error(message)
                        self._status.marker_errors.append(message)
                        self._status.marker_status = DetectorStatus.Capture.FAILURE
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

    def supported_request_types(self) -> dict[type[MCastRequest], Callable[[dict], MCastResponse]]:
        return_value: dict[type[MCastRequest], Callable[[dict], MCastResponse]] = super().supported_request_types()
        return_value.update({
            GetCaptureDeviceRequest: self.get_capture_device,
            GetCaptureImageRequest: self.get_capture_image,
            GetCapturePropertiesRequest: self.get_capture_properties,
            GetDetectionParametersRequest: self.get_detection_parameters,
            GetMarkerSnapshotsRequest: self.get_marker_snapshots,
            SetCaptureDeviceRequest: self.set_capture_device,
            SetCapturePropertiesRequest: self.set_capture_properties,
            SetDetectionParametersRequest: self.set_detection_parameters,
            StartCaptureRequest: self.start_capture,
            StopCaptureRequest: self.stop_capture})
        return return_value

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

    def get_capture_device(self, **_kwargs) -> GetCaptureDeviceResponse:
        return GetCaptureDeviceResponse(capture_device_id=str(self._capture_device_id))

    def get_capture_image(self, **kwargs) -> GetCaptureImageResponse:
        """
        :key request: GetCaptureImageRequest
        """

        request: GetCaptureImageRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=GetCaptureImageRequest)

        encoded_frame: bool
        encoded_image_rgb_single_row: numpy.array
        encoded, encoded_image_rgb_single_row = cv2.imencode(request.format, self._captured_image)
        encoded_image_rgb_bytes: bytes = encoded_image_rgb_single_row.tobytes()
        encoded_image_rgb_base64 = base64.b64encode(encoded_image_rgb_bytes)
        return GetCaptureImageResponse(
            format=request.format,
            image_base64=encoded_image_rgb_base64)

        # img_bytes = base64.b64decode(img_str)
        # img_buffer = numpy.frombuffer(img_bytes, dtype=numpy.uint8)
        # img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)

    def get_capture_properties(self, **_kwargs) -> GetCapturePropertiesResponse | ErrorResponse:
        if self._capture is None:
            return ErrorResponse(
                message="The capture is not active, and properties cannot be retrieved.")
        else:
            return GetCapturePropertiesResponse(
                resolution_x_px=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                resolution_y_px=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=int(round(self._capture.get(cv2.CAP_PROP_FPS))),
                auto_exposure=bool(round(self._capture.get(cv2.CAP_PROP_AUTO_EXPOSURE))),
                exposure=int(self._capture.get(cv2.CAP_PROP_EXPOSURE)),
                brightness=int(self._capture.get(cv2.CAP_PROP_BRIGHTNESS)),
                contrast=int(self._capture.get(cv2.CAP_PROP_CONTRAST)),
                sharpness=int(self._capture.get(cv2.CAP_PROP_SHARPNESS)),
                gamma=int(self._capture.get(cv2.CAP_PROP_GAMMA)))
        # TODO: Get powerline_frequency_hz and backlight_compensation

    def get_detection_parameters(self, **_kwargs) -> GetDetectionParametersResponse | ErrorResponse:
        if self._marker_parameters.cornerRefinementMethod not in CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT:
            return ErrorResponse(
                message=f"Corner refinement method appears to be set to an invalid value: "
                        f"{self._marker_parameters.corner_refinement_method}.")
        corner_refinement_method_text: CornerRefinementMethod = \
            CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT[self._marker_parameters.cornerRefinementMethod]
        params: DetectionParameters = DetectionParameters(
            adaptive_thresh_constant=self._marker_parameters.adaptiveThreshConstant,
            adaptive_thresh_win_size_max=self._marker_parameters.adaptiveThreshWinSizeMax,
            adaptive_thresh_win_size_min=self._marker_parameters.adaptiveThreshWinSizeMin,
            adaptive_thresh_win_size_step=self._marker_parameters.adaptiveThreshWinSizeStep,
            april_tag_critical_rad=self._marker_parameters.aprilTagCriticalRad,
            april_tag_deglitch=self._marker_parameters.aprilTagDeglitch,
            april_tag_max_line_fit_mse=self._marker_parameters.aprilTagMaxLineFitMse,
            april_tag_max_nmaxima=self._marker_parameters.aprilTagMaxNmaxima,
            april_tag_min_cluster_pixels=self._marker_parameters.aprilTagMinClusterPixels,
            april_tag_min_white_black_diff=self._marker_parameters.aprilTagMinWhiteBlackDiff,
            april_tag_quad_decimate=self._marker_parameters.aprilTagQuadDecimate,
            april_tag_quad_sigma=self._marker_parameters.aprilTagQuadSigma,
            corner_refinement_max_iterations=self._marker_parameters.cornerRefinementMaxIterations,
            corner_refinement_method=corner_refinement_method_text,
            corner_refinement_min_accuracy=self._marker_parameters.cornerRefinementMinAccuracy,
            corner_refinement_win_size=self._marker_parameters.cornerRefinementWinSize,
            detect_inverted_marker=self._marker_parameters.detectInvertedMarker,
            error_correction_rate=self._marker_parameters.errorCorrectionRate,
            marker_border_bits=self._marker_parameters.markerBorderBits,
            max_erroneous_bits_in_border_rate=self._marker_parameters.maxErroneousBitsInBorderRate,
            max_marker_perimeter_rate=self._marker_parameters.maxMarkerPerimeterRate,
            min_corner_distance_rate=self._marker_parameters.minCornerDistanceRate,
            min_distance_to_border=self._marker_parameters.minDistanceToBorder,
            min_marker_distance_rate=self._marker_parameters.minMarkerDistanceRate,
            min_marker_perimeter_rate=self._marker_parameters.minMarkerPerimeterRate,
            min_otsu_std_dev=self._marker_parameters.minOtsuStdDev,
            perspective_remove_ignored_margin_per_cell=self._marker_parameters.perspectiveRemoveIgnoredMarginPerCell,
            perspective_remove_pixel_per_cell=self._marker_parameters.perspectiveRemovePixelPerCell,
            polygonal_approx_accuracy_rate=self._marker_parameters.polygonalApproxAccuracyRate)
        return GetDetectionParametersResponse(parameters=params)

    def get_marker_snapshots(self, **kwargs) -> GetMarkerSnapshotsResponse:
        """
        :key request: GetMarkerSnapshotsRequest
        """

        request: GetMarkerSnapshotsRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=GetMarkerSnapshotsRequest)

        response_dict: dict = dict()
        if request.include_detected:
            response_dict["detected_marker_snapshots"] = self._marker_detected_snapshots
        if request.include_rejected:
            response_dict["rejected_marker_snapshots"] = self._marker_rejected_snapshots
        return GetMarkerSnapshotsResponse(**response_dict)

    def set_capture_device(self, **kwargs) -> EmptyResponse | ErrorResponse:
        """
        :key request: SetCaptureDeviceRequest
        """

        request: SetCaptureDeviceRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=SetCaptureDeviceRequest)

        input_device_id: int | str = request.capture_device_id
        if input_device_id.isnumeric():
            input_device_id = int(input_device_id)
        if self._capture_device_id != input_device_id:
            self._capture_device_id = input_device_id
            if self._capture is not None:
                self._capture.release()
                self._capture = cv2.VideoCapture(input_device_id, cv2.CAP_DSHOW)
                if not self._capture.isOpened():
                    return ErrorResponse(
                        message=f"Failed to open capture device {input_device_id}")
        return EmptyResponse()

    # noinspection DuplicatedCode
    def set_capture_properties(self, **kwargs) -> EmptyResponse:
        """
        :key request: SetCapturePropertiesRequest
        """

        request: SetCapturePropertiesRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=SetCapturePropertiesRequest)

        if self._capture is not None:
            if request.resolution_x_px is not None:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(request.resolution_x_px))
            if request.resolution_y_px is not None:
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(request.resolution_y_px))
            if request.fps is not None:
                self._capture.set(cv2.CAP_PROP_FPS, float(request.fps))
            if request.auto_exposure is not None:
                self._capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(request.auto_exposure))
            if request.exposure is not None:
                self._capture.set(cv2.CAP_PROP_EXPOSURE, float(request.exposure))
            if request.brightness is not None:
                self._capture.set(cv2.CAP_PROP_BRIGHTNESS, float(request.brightness))
            if request.contrast is not None:
                self._capture.set(cv2.CAP_PROP_CONTRAST, float(request.contrast))
            if request.sharpness is not None:
                self._capture.set(cv2.CAP_PROP_SHARPNESS, float(request.sharpness))
            if request.gamma is not None:
                self._capture.set(cv2.CAP_PROP_GAMMA, float(request.gamma))
        return EmptyResponse()

    # noinspection DuplicatedCode
    def set_detection_parameters(self, **kwargs) -> EmptyResponse | ErrorResponse:
        """
        :key client_identifier: str
        :key request: SetDetectionParametersRequest
        """

        request: SetDetectionParametersRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=SetDetectionParametersRequest)

        params: DetectionParameters = request.parameters

        if params is None:
            return ErrorResponse(message="Received empty parameters.")

        if params.adaptive_thresh_win_size_min:
            self._marker_parameters.adaptiveThreshWinSizeMin = params.adaptive_thresh_win_size_min
        if params.adaptive_thresh_win_size_max:
            self._marker_parameters.adaptiveThreshWinSizeMax = params.adaptive_thresh_win_size_max
        if params.adaptive_thresh_win_size_step:
            self._marker_parameters.adaptiveThreshWinSizeStep = params.adaptive_thresh_win_size_step
        if params.adaptive_thresh_constant:
            self._marker_parameters.adaptiveThreshConstant = params.adaptive_thresh_constant

        if params.min_marker_perimeter_rate:
            self._marker_parameters.minMarkerPerimeterRate = params.min_marker_perimeter_rate
        if params.max_marker_perimeter_rate:
            self._marker_parameters.maxMarkerPerimeterRate = params.max_marker_perimeter_rate
        if params.polygonal_approx_accuracy_rate:
            self._marker_parameters.polygonalApproxAccuracyRate = params.polygonal_approx_accuracy_rate
        if params.min_corner_distance_rate:
            self._marker_parameters.minCornerDistanceRate = params.min_corner_distance_rate
        if params.min_marker_distance_rate:
            self._marker_parameters.minMarkerDistanceRate = params.min_marker_distance_rate
        if params.min_distance_to_border:
            self._marker_parameters.minDistanceToBorder = params.min_distance_to_border

        if params.corner_refinement_max_iterations:
            self._marker_parameters.cornerRefinementMaxIterations = params.corner_refinement_max_iterations
        if params.corner_refinement_method:
            if params.corner_refinement_method in CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT:
                self._marker_parameters.cornerRefinementMethod = \
                    CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT[params.corner_refinement_method]
            else:
                return ErrorResponse(
                    message=f"Failed to find corner refinement method {params.corner_refinement_method}.")
        if params.corner_refinement_min_accuracy:
            self._marker_parameters.cornerRefinementMinAccuracy = params.corner_refinement_min_accuracy
        if params.corner_refinement_win_size:
            self._marker_parameters.cornerRefinementWinSize = params.corner_refinement_win_size

        if params.marker_border_bits:
            self._marker_parameters.markerBorderBits = params.marker_border_bits
        if params.min_otsu_std_dev:
            self._marker_parameters.minOtsuStdDev = params.min_otsu_std_dev
        if params.perspective_remove_pixel_per_cell:
            self._marker_parameters.perspectiveRemovePixelPerCell = params.perspective_remove_pixel_per_cell
        if params.perspective_remove_ignored_margin_per_cell:
            self._marker_parameters.perspectiveRemoveIgnoredMarginPerCell = \
                params.perspective_remove_ignored_margin_per_cell

        if params.max_erroneous_bits_in_border_rate:
            self._marker_parameters.maxErroneousBitsInBorderRate = params.max_erroneous_bits_in_border_rate
        if params.error_correction_rate:
            self._marker_parameters.errorCorrectionRate = params.error_correction_rate
        if params.detect_inverted_marker:
            self._marker_parameters.detectInvertedMarker = params.detect_inverted_marker

        if params.april_tag_critical_rad:
            self._marker_parameters.aprilTagCriticalRad = params.april_tag_critical_rad
        if params.april_tag_deglitch:
            self._marker_parameters.aprilTagDeglitch = params.april_tag_deglitch
        if params.april_tag_max_line_fit_mse:
            self._marker_parameters.aprilTagMaxLineFitMse = params.april_tag_max_line_fit_mse
        if params.april_tag_max_nmaxima:
            self._marker_parameters.aprilTagMaxNmaxima = params.april_tag_max_nmaxima
        if params.april_tag_min_cluster_pixels:
            self._marker_parameters.aprilTagMinClusterPixels = params.april_tag_min_cluster_pixels
        if params.april_tag_min_white_black_diff:
            self._marker_parameters.aprilTagMinWhiteBlackDiff = params.april_tag_min_white_black_diff
        if params.april_tag_quad_decimate:
            self._marker_parameters.aprilTagQuadDecimate = params.april_tag_quad_decimate
        if params.april_tag_quad_sigma:
            self._marker_parameters.aprilTagQuadSigma = params.april_tag_quad_sigma

        # Note: a relatively recent addition to OpenCV, may not be available in some python versions
        if hasattr(self._marker_parameters, "useAruco3Detection"):
            if params.use_aruco_3_detection:
                self._marker_parameters.useAruco3Detection = params.use_aruco_3_detection
            if params.min_side_length_canonical_img:
                self._marker_parameters.minSideLengthCanonicalImg = params.min_side_length_canonical_img
            if params.min_marker_length_ratio_original_img:
                self._marker_parameters.minMarkerLengthRatioOriginalImg = params.min_marker_length_ratio_original_img

        return EmptyResponse()

    def start_capture(self, **kwargs) -> MCastResponse:
        """
        :key client_identifier: str
        """

        client_identifier: str = get_kwarg(
            kwargs=kwargs,
            key="client_identifier",
            arg_type=str)

        if isinstance(self._capture_device_id, str) and self._capture_device_id.isnumeric():
            self._capture_device_id.isnumeric = int(self._capture_device_id)
        if self._capture is not None:
            return EmptyResponse()
        self.add_status_message(severity="info", message=f"{client_identifier} is starting capture.")

        self._capture = cv2.VideoCapture(self._capture_device_id,  cv2.CAP_DSHOW)
        # NOTE: The USB3 cameras bought for this project appear to require some basic parameters to be set,
        #       otherwise frame grab results in error
        self._capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._capture.set(cv2.CAP_PROP_FPS, 60)

        self._status.capture_status = DetectorStatus.Capture.RUNNING
        return EmptyResponse()

    def stop_capture(self, **kwargs) -> MCastResponse:
        """
        :key client_identifier: str
        """

        client_identifier: str = get_kwarg(
            kwargs=kwargs,
            key="client_identifier",
            arg_type=str)

        self.add_status_message(severity="info", message=f"{client_identifier} is stopping capture.")
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._status.capture_status = DetectorStatus.Capture.STOPPED
        return EmptyResponse()
