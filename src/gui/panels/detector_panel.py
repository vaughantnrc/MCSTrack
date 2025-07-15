from .base_panel import \
    BasePanel
from .feedback import \
    ImagePanel
from .parameters import \
    ParameterBase, \
    ParameterCheckbox, \
    ParameterSpinboxFloat, \
    ParameterSelector
from src.common import \
    Annotation, \
    DetectorFrame, \
    ErrorResponse, \
    EmptyResponse, \
    ImageFormat, \
    ImageResolution, \
    ImageUtils, \
    KeyValueSimpleAny, \
    MCTRequestSeries, \
    MCTResponse, \
    MCTResponseSeries, \
    StatusMessageSource
from src.controller import \
    MCTController
from src.detector.api import \
    CalibrationImageAddRequest, \
    CalibrationImageAddResponse, \
    CameraImageGetRequest, \
    CameraImageGetResponse, \
    CameraParametersGetRequest, \
    CameraParametersGetResponse, \
    CameraParametersSetRequest, \
    CameraParametersSetResponse, \
    AnnotatorParametersGetRequest, \
    AnnotatorParametersGetResponse, \
    AnnotatorParametersSetRequest
import cv2
from io import BytesIO
import logging
import numpy
from typing import Final, Optional
import uuid
import wx


logger = logging.getLogger(__name__)

_UPDATE_INTERVAL_MILLISECONDS: Final[int] = 16
_SUPPORTED_RESOLUTIONS: Final[list[ImageResolution]] = [
    ImageUtils.StandardResolutions.RES_640X480,
    ImageUtils.StandardResolutions.RES_1280X720,
    ImageUtils.StandardResolutions.RES_1920X1080]
_SUPPORTED_FPS: Final[list[str]] = [
    "15",
    "30",
    "60"]
_SUPPORTED_CORNER_REFINEMENT_METHODS: Final[list[str]] = [
    "NONE",
    "SUBPIX",
    "CONTOUR",
    "APRILTAG"]
_CAPTURE_FORMAT: ImageFormat = ImageFormat.FORMAT_JPG

_CAMERA_PARAMETER_SLOT_COUNT: Final[int] = 100


class DetectorPanel(BasePanel):

    _controller: MCTController

    _control_blocking_request_id: uuid.UUID | None
    _live_preview_request_id: uuid.UUID | None

    _live_preview_image_base64: str | None
    _live_markers_detected: list[Annotation]
    _live_markers_rejected: list[Annotation]
    _live_resolution: ImageResolution | None

    _detector_selector: ParameterSelector
    _preview_scale_factor: ParameterSpinboxFloat
    _preview_image_checkbox: ParameterCheckbox
    _annotate_detected_checkbox: ParameterCheckbox
    _annotate_rejected_checkbox: ParameterCheckbox
    _send_capture_parameters_button: wx.Button
    _send_detection_parameters_button: wx.Button

    _camera_parameter_panel: wx.Panel
    _camera_parameter_sizer: wx.BoxSizer
    _camera_parameter_uis: list[ParameterBase]

    _marker_parameter_panel: wx.Panel
    _marker_parameter_sizer: wx.BoxSizer
    _marker_parameter_uis: list[ParameterBase]

    _calibration_capture_button: wx.Button

    _image_panel: ImagePanel

    def __init__(
        self,
        parent: wx.Window,
        controller: MCTController,
        status_message_source: StatusMessageSource,
        name: str = "DetectorPanel"
    ):
        super().__init__(
            parent=parent,
            status_message_source=status_message_source,
            name=name)
        self._controller = controller
        self._capture_active = False
        self._live_preview_request_id = None
        self._control_blocking_request_id = None

        self._live_preview_image_base64 = None
        self._live_markers_detected = list()
        self._live_markers_rejected = list()
        self._live_resolution = None

        self._camera_parameter_uis = list()
        self._marker_parameter_uis = list()

        horizontal_split_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.HORIZONTAL)

        control_border_panel: wx.Panel = wx.Panel(parent=self)
        control_border_box: wx.StaticBoxSizer = wx.StaticBoxSizer(
            orient=wx.VERTICAL,
            parent=control_border_panel)
        control_panel: wx.ScrolledWindow = wx.ScrolledWindow(
            parent=control_border_panel)
        control_panel.SetScrollRate(
            xstep=1,
            ystep=1)
        control_panel.ShowScrollbars(
            horz=wx.SHOW_SB_NEVER,
            vert=wx.SHOW_SB_ALWAYS)

        control_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.VERTICAL)

        self._detector_selector = self.add_control_selector(
            parent=control_panel,
            sizer=control_sizer,
            label="Detector",
            selectable_values=list())

        self.add_horizontal_line_to_spacer(
            parent=control_panel,
            sizer=control_sizer)

        self.add_text_label(
            parent=control_panel,
            sizer=control_sizer,
            label="Detector",
            font_size_delta=2,
            bold=True)

        self._preview_image_checkbox = self.add_control_checkbox(
            parent=control_panel,
            sizer=control_sizer,
            label="Preview Image")

        self._preview_scale_factor = self.add_control_spinbox_float(
            parent=control_panel,
            sizer=control_sizer,
            label="Preview Scale",
            minimum_value=0.03125,  # 1/32 in each dimension, for a minimum of 1/1024 original resolution
            maximum_value=1,  # No scaling
            initial_value=0.25,  # 1/4 in each dimension, for a default of 1/16 original resolution
            step_value=0.125,
            digit_count=4)

        self._annotate_detected_checkbox = self.add_control_checkbox(
            parent=control_panel,
            sizer=control_sizer,
            label="Annotate Detected")

        self._annotate_rejected_checkbox = self.add_control_checkbox(
            parent=control_panel,
            sizer=control_sizer,
            label="Annotate Rejected")

        self._calibration_capture_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Capture Calibration Image")

        self.add_horizontal_line_to_spacer(
            parent=control_panel,
            sizer=control_sizer)

        self.add_text_label(
            parent=control_panel,
            sizer=control_sizer,
            label="Capture",
            font_size_delta=2,
            bold=True)

        self._camera_parameter_panel: wx.Panel = wx.Panel(parent=control_panel)
        self._camera_parameter_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.VERTICAL)
        self._camera_parameter_panel.SetSizer(sizer=self._camera_parameter_sizer)
        control_sizer.Add(
            window=self._camera_parameter_panel,
            flags=wx.SizerFlags(0).Expand())

        self._send_capture_parameters_button: wx.Button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Send Capture Parameters")

        self.add_horizontal_line_to_spacer(
            parent=control_panel,
            sizer=control_sizer)

        self.add_text_label(
            parent=control_panel,
            sizer=control_sizer,
            label="Detection",
            font_size_delta=2,
            bold=True)

        self._marker_parameter_panel: wx.Panel = wx.Panel(parent=control_panel)
        self._marker_parameter_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.VERTICAL)
        self._marker_parameter_panel.SetSizer(sizer=self._marker_parameter_sizer)
        control_sizer.Add(
            window=self._marker_parameter_panel,
            flags=wx.SizerFlags(0).Expand())

        self._send_detection_parameters_button = self.add_control_button(
            parent=control_panel,
            sizer=control_sizer,
            label="Send Detection Parameters")

        control_spacer_sizer: wx.BoxSizer = wx.BoxSizer(orient=wx.HORIZONTAL)
        control_sizer.Add(
            sizer=control_spacer_sizer,
            flags=wx.SizerFlags(1).Expand())

        control_panel.SetSizerAndFit(sizer=control_sizer)
        control_border_box.Add(
            window=control_panel,
            flags=wx.SizerFlags(1).Expand())
        control_border_panel.SetSizer(sizer=control_border_box)
        horizontal_split_sizer.Add(
            window=control_border_panel,
            flags=wx.SizerFlags(35).Expand())

        self._image_panel = ImagePanel(parent=self)
        self._image_panel.SetBackgroundColour(colour=wx.BLACK)
        horizontal_split_sizer.Add(
            window=self._image_panel,
            flags=wx.SizerFlags(65).Expand())

        self.SetSizerAndFit(sizer=horizontal_split_sizer)

        self._detector_selector.selector.Bind(
            event=wx.EVT_CHOICE,
            handler=self.on_detector_selected)
        self._preview_image_checkbox.checkbox.Bind(
            event=wx.EVT_CHECKBOX,
            handler=self.on_display_mode_changed)
        self._annotate_detected_checkbox.checkbox.Bind(
            event=wx.EVT_CHECKBOX,
            handler=self.on_display_mode_changed)
        self._annotate_rejected_checkbox.checkbox.Bind(
            event=wx.EVT_CHECKBOX,
            handler=self.on_display_mode_changed)
        self._send_capture_parameters_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_send_capture_parameters_pressed)
        self._send_detection_parameters_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_send_detection_parameters_pressed)
        self._calibration_capture_button.Bind(
            event=wx.EVT_BUTTON,
            handler=self.on_calibration_capture_pressed)

        self._update_ui_controls()

    def begin_capture_calibration(self) -> None:
        selected_detector_label: str = self._detector_selector.selector.GetStringSelection()
        request_series: MCTRequestSeries = MCTRequestSeries(series=[CalibrationImageAddRequest()])
        self._control_blocking_request_id = self._controller.request_series_push(
            connection_label=selected_detector_label,
            request_series=request_series)
        self._update_ui_controls()

    def begin_capture_snapshot(self, requested_resolution: ImageResolution | None = None):
        selected_detector_label: str = self._detector_selector.selector.GetStringSelection()
        request_series: MCTRequestSeries = MCTRequestSeries(
            series=[CameraImageGetRequest(
                format=_CAPTURE_FORMAT,
                requested_resolution=requested_resolution)])
        self._live_preview_request_id = self._controller.request_series_push(
            connection_label=selected_detector_label,
            request_series=request_series)

    def begin_get_detector_parameters(self):
        selected_detector_label: str = self._detector_selector.selector.GetStringSelection()
        request_series: MCTRequestSeries = MCTRequestSeries(
            series=[
                CameraParametersGetRequest(),
                AnnotatorParametersGetRequest()])
        self._control_blocking_request_id = self._controller.request_series_push(
            connection_label=selected_detector_label,
            request_series=request_series)
        self._update_ui_controls()

    def begin_set_capture_parameters(self):
        selected_detector_label: str = self._detector_selector.selector.GetStringSelection()
        key_values: list[KeyValueSimpleAny] = self.populate_key_value_list_from_dynamic_ui(
            parameter_uis=self._camera_parameter_uis)
        request_series: MCTRequestSeries = MCTRequestSeries(
            series=[
                CameraParametersSetRequest(parameters=key_values),
                CameraParametersGetRequest()])  # sync
        self._control_blocking_request_id = self._controller.request_series_push(
            connection_label=selected_detector_label,
            request_series=request_series)
        self._update_ui_controls()

    def begin_set_detection_parameters(self):
        selected_detector_label: str = self._detector_selector.selector.GetStringSelection()
        key_values: list[KeyValueSimpleAny] = self.populate_key_value_list_from_dynamic_ui(
            parameter_uis=self._marker_parameter_uis)
        request_series: MCTRequestSeries = MCTRequestSeries(series=[
            AnnotatorParametersSetRequest(parameters=key_values),
            AnnotatorParametersGetRequest()])  # sync
        self._control_blocking_request_id = self._controller.request_series_push(
            connection_label=selected_detector_label,
            request_series=request_series)
        self._update_ui_controls()

    def handle_response_series(
        self,
        response_series: MCTResponseSeries,
        task_description: Optional[str] = None,
        expected_response_count: Optional[int] = None
    ) -> None:
        response: MCTResponse
        for response in response_series.series:
            if isinstance(response, CalibrationImageAddResponse):
                self._handle_add_calibration_image_response(response=response)
            elif isinstance(response, CameraImageGetResponse):
                self._handle_capture_snapshot_response(response=response)
            elif isinstance(response, CameraParametersGetResponse):
                self._handle_get_capture_parameters_response(response=response)
            elif isinstance(response, AnnotatorParametersGetResponse):
                self._handle_get_detection_parameters_response(response=response)
            elif isinstance(response, ErrorResponse):
                self.handle_error_response(response=response)
            elif not isinstance(response, (EmptyResponse, CameraParametersSetResponse)):
                self.handle_unknown_response(response=response)

    def _handle_add_calibration_image_response(
        self,
        response: CalibrationImageAddResponse
    ):
        self.status_message_source.enqueue_status_message(
            severity="info",
            message=f"Added image {response.image_identifier}.")

    def _handle_capture_snapshot_response(
        self,
        response: CameraImageGetResponse
    ):
        if self._preview_image_checkbox.checkbox.GetValue():
            self._live_preview_image_base64 = response.image_base64

    # noinspection DuplicatedCode
    def _handle_get_capture_parameters_response(
        self,
        response: CameraParametersGetResponse
    ):
        self._camera_parameter_panel.Freeze()
        self._camera_parameter_sizer.Clear(True)
        self._camera_parameter_sizer = wx.BoxSizer(orient=wx.VERTICAL)
        self._camera_parameter_uis = self.populate_dynamic_ui_from_key_value_list(
            key_value_list=response.parameters,
            containing_panel=self._camera_parameter_panel,
            containing_sizer=self._camera_parameter_sizer)
        self._camera_parameter_panel.SetSizer(self._camera_parameter_sizer)
        self._camera_parameter_panel.Thaw()
        self.Layout()

    # noinspection DuplicatedCode
    def _handle_get_detection_parameters_response(
        self,
        response: AnnotatorParametersGetResponse
    ):
        self._marker_parameter_panel.Freeze()
        self._marker_parameter_sizer.Clear(True)
        self._marker_parameter_sizer = wx.BoxSizer(orient=wx.VERTICAL)
        self._marker_parameter_uis = self.populate_dynamic_ui_from_key_value_list(
            key_value_list=response.parameters,
            containing_panel=self._marker_parameter_panel,
            containing_sizer=self._marker_parameter_sizer)
        self._marker_parameter_panel.SetSizer(self._marker_parameter_sizer)
        self._marker_parameter_panel.Thaw()
        self.Layout()

    @staticmethod
    def _marker_snapshot_list_to_opencv_points(
        marker_snapshot_list: list[Annotation],
        scale: float
    ) -> numpy.ndarray:
        if len(marker_snapshot_list) <= 0:
            return numpy.asarray([], dtype=numpy.int32)
        return_value: list[list[list[(float, float)]]] = list()
        current_base_label: str = marker_snapshot_list[0].base_feature_label()
        current_shape_points: list[list[(float, float)]] = [[
            marker_snapshot_list[0].x_px * scale,
            marker_snapshot_list[0].y_px * scale]]
        for marker_snapshot in marker_snapshot_list:
            annotation_base_label = marker_snapshot.base_feature_label()
            if annotation_base_label != current_base_label:
                return_value.append(current_shape_points)
                current_base_label = annotation_base_label
            current_shape_points.append([
                marker_snapshot.x_px * scale,
                marker_snapshot.y_px * scale])
        return_value.append(current_shape_points)
        return_value = numpy.asarray(return_value, dtype=numpy.int32)
        return return_value

    def on_calibration_capture_pressed(self, _event: wx.CommandEvent):
        self.begin_capture_calibration()

    def on_detector_selected(self, _event: wx.CommandEvent):
        self._live_preview_image_base64 = None
        self.begin_get_detector_parameters()
        self._update_ui_image()
        self._update_ui_controls()

    def on_display_mode_changed(self, _event: wx.CommandEvent):
        if not self._preview_image_checkbox.checkbox.GetValue():
            self._live_preview_image_base64 = None
        self._update_ui_image()

    def on_page_select(self):
        super().on_page_select()
        available_detector_labels: list[str] = self._controller.get_active_detector_labels()
        self._detector_selector.set_options(option_list=available_detector_labels)
        self._update_ui_controls()

    def on_send_capture_parameters_pressed(self, _event: wx.CommandEvent):
        self.begin_set_capture_parameters()

    def on_send_detection_parameters_pressed(self, _event: wx.CommandEvent):
        self.begin_set_detection_parameters()

    def _set_display_controls_enabled(
        self,
        enable: bool
    ):
        self._preview_image_checkbox.Enable(enable=enable)
        self._preview_scale_factor.Enable(enable=enable)
        self._annotate_detected_checkbox.Enable(enable=enable)
        self._annotate_rejected_checkbox.Enable(enable=enable)

    def _set_parameter_controls_enabled(
        self,
        enable: bool
    ):
        for parameter_ui in self._camera_parameter_uis:
            parameter_ui.set_enabled(enable=enable)
        for parameter_ui in self._marker_parameter_uis:
            parameter_ui.set_enabled(enable=enable)
        self._send_capture_parameters_button.Enable(enable=enable)
        self._send_detection_parameters_button.Enable(enable=enable)

    def update_loop(self):
        super().update_loop()

        response_series: MCTResponseSeries | None
        if self._control_blocking_request_id is not None:
            self._control_blocking_request_id, response_series = self._controller.response_series_pop(
                request_series_id=self._control_blocking_request_id)
            if response_series is not None:  # self._control_blocking_request_id will be None
                self.handle_response_series(response_series)
                self._update_ui_controls()
        elif self._live_preview_request_id is not None:
            self._live_preview_request_id, response_series = self._controller.response_series_pop(
                request_series_id=self._live_preview_request_id)
            if response_series is not None:
                self.handle_response_series(response_series)

        detector_label: str = self._detector_selector.selector.GetStringSelection()
        if detector_label is not None and len(detector_label) > 0:
            detector_frame: DetectorFrame | None = self._controller.get_live_detector_frame(
                detector_label=detector_label)
            if detector_frame is not None:
                self._live_markers_detected = detector_frame.annotations_identified
                self._live_markers_rejected = detector_frame.annotations_unidentified
                self._live_resolution = detector_frame.image_resolution
            if self._preview_image_checkbox.checkbox.GetValue() and self._live_preview_request_id is None:
                if self._live_resolution is not None:
                    preview_resolution: ImageResolution = ImageResolution(
                        x_px=int(round(self._live_resolution.x_px * self._preview_scale_factor.get_value())),
                        y_px=int(round(self._live_resolution.y_px * self._preview_scale_factor.get_value())))
                    self.begin_capture_snapshot(requested_resolution=preview_resolution)
                else:
                    self.begin_capture_snapshot()

        if self._preview_image_checkbox.checkbox.GetValue() or \
           self._annotate_detected_checkbox.checkbox.GetValue() or \
           self._annotate_rejected_checkbox.checkbox.GetValue():
            self._update_ui_image()

    def _update_ui_controls(self):
        self._detector_selector.set_enabled(enable=False)
        self._set_display_controls_enabled(enable=False)
        self._set_parameter_controls_enabled(enable=False)
        self._calibration_capture_button.Enable(enable=False)
        if not self._controller.is_running():
            return
        self._detector_selector.set_enabled(enable=True)
        selected_detector_label: str = self._detector_selector.selector.GetStringSelection()
        if selected_detector_label is None or len(selected_detector_label) <= 0:
            return
        if self._control_blocking_request_id is not None:
            return
        self._set_display_controls_enabled(enable=True)
        self._set_parameter_controls_enabled(enable=True)
        self._calibration_capture_button.Enable(enable=True)

    def _update_ui_image(self):
        display_image: numpy.ndarray
        if self._live_resolution is None:
            display_image = ImageUtils.black_image(resolution_px=self._image_panel.GetSize())
        else:
            scale: float | None
            if self._live_preview_image_base64 is not None:
                opencv_image: numpy.ndarray = ImageUtils.base64_to_image(input_base64=self._live_preview_image_base64)
                display_image: numpy.ndarray = ImageUtils.image_resize_to_fit(
                    opencv_image=opencv_image,
                    available_size=self._image_panel.GetSize())
                scale: float = self._preview_scale_factor.get_value() * display_image.shape[0] / opencv_image.shape[0]
            else:
                display_image = ImageUtils.black_image(resolution_px=self._image_panel.GetSize())
                panel_size_px: tuple[int, int] = self._image_panel.GetSize()
                rescaled_resolution_px: tuple[int, int] = ImageUtils.scale_factor_for_available_space_px(
                    source_resolution_px=(self._live_resolution.x_px, self._live_resolution.y_px),
                    available_size_px=panel_size_px)
                scale: float = rescaled_resolution_px[1] / self._live_resolution.y_px

            if scale is not None:
                if self._annotate_detected_checkbox.checkbox.GetValue():
                    corners: numpy.ndarray = self._marker_snapshot_list_to_opencv_points(
                        marker_snapshot_list=self._live_markers_detected,
                        scale=scale)
                    cv2.polylines(
                        img=display_image,
                        pts=corners,
                        isClosed=True,
                        color=[255, 191, 127],  # blue
                        thickness=2)
                if self._annotate_rejected_checkbox.checkbox.GetValue():
                    corners: numpy.ndarray = self._marker_snapshot_list_to_opencv_points(
                        marker_snapshot_list=self._live_markers_rejected,
                        scale=scale)
                    cv2.polylines(
                        img=display_image,
                        pts=corners,
                        isClosed=True,
                        color=[127, 191, 255],  # orange
                        thickness=2)

        image_buffer: bytes = ImageUtils.image_to_bytes(image_data=display_image, image_format=".jpg")
        image_buffer_io: BytesIO = BytesIO(image_buffer)
        wx_image: wx.Image = wx.Image(image_buffer_io)
        wx_bitmap: wx.Bitmap = wx_image.ConvertToBitmap()
        self._image_panel.set_bitmap(wx_bitmap)
        self._image_panel.paint()
