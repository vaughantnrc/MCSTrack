from .api import \
    DetectorFrameGetRequest, \
    DetectorFrameGetResponse, \
    DetectorParametersGetResponse, \
    DetectorQueryResponse, \
    IntrinsicCalibrationResultGetActiveResponse
from .detector import \
    Detector
from src.common import \
    Annotator, \
    Camera, \
    EmptyResponse, \
    IntrinsicCalibrator, \
    TimestampGetResponse, \
    TimeSyncStartRequest, \
    TimeSyncStopRequest
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
import hjson
import logging
import os
from typing import Final


logger = logging.getLogger(__name__)

CONFIGURATION_FILEPATH_ENV_VAR: Final[str] = "MCSTRACK_DETECTOR_CONFIGURATION_FILEPATH"


def create_app() -> FastAPI:
    configuration_filepath: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "configuration", "detector", "opencv_aruco.json")
    configuration_filepath = os.getenv(CONFIGURATION_FILEPATH_ENV_VAR, configuration_filepath)
    configuration: Detector.Configuration
    with open(configuration_filepath, 'r') as infile:
        file_contents: str = infile.read()
        configuration_dict = hjson.loads(file_contents)
        configuration = Detector.Configuration(**configuration_dict)

    # Eventually it would be preferable to put the initialization logic/mapping below into an abstract factory,
    # and allow end-users to register custom classes that are not necessarily shipped within this library.

    camera_type: type[Camera]
    if configuration.camera.implementation == "opencv_capture_device":
        from src.implementations.camera_opencv_capture_device import OpenCVCaptureDeviceCamera
        camera_type = OpenCVCaptureDeviceCamera
    elif configuration.camera.implementation == "picamera2":
        from src.implementations.camera_picamera2 import Picamera2Camera
        camera_type = Picamera2Camera
    elif configuration.camera.implementation == "mock":
        from src.implementations.camera_mock import MockCamera
        camera_type = MockCamera
    else:
        raise RuntimeError(f"Unsupported camera implementation {configuration.camera.implementation}.")

    marker_type: type[Annotator]
    if configuration.annotator.implementation == "aruco_opencv":
        from src.implementations.annotator_aruco_opencv import ArucoOpenCVAnnotator
        marker_type = ArucoOpenCVAnnotator
    elif configuration.annotator.implementation == "mock":
        from src.implementations.annotator_mock import MockAnnotator
        marker_type = MockAnnotator
    else:
        raise RuntimeError(f"Unsupported annotation implementation {configuration.annotator_configuration.method}.")

    intrinsic_calibrator_type: type[IntrinsicCalibrator]
    if configuration.intrinsic_calibrator.implementation == "charuco_opencv":
        from src.implementations.intrinsic_charuco_opencv import CharucoOpenCVIntrinsicCalibrator
        intrinsic_calibrator_type = CharucoOpenCVIntrinsicCalibrator
    elif configuration.intrinsic_calibrator.implementation == "mock":
        from src.implementations.intrinsic_mock import MockIntrinsicCalibrator
        intrinsic_calibrator_type = MockIntrinsicCalibrator
    else:
        raise RuntimeError(f"Unsupported intrinsic implementation {configuration.camera.implementation}.")

    detector = Detector(
        configuration=configuration,
        camera_type=camera_type,
        annotator_type=marker_type,
        intrinsic_calibrator_type=intrinsic_calibrator_type)
    detector_app = FastAPI()

    # CORS Middleware
    origins = ["http://localhost"]
    detector_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    @detector_app.get("/detector/parameters_get")
    async def camera_get_parameters() -> DetectorParametersGetResponse:
        result: DetectorParametersGetResponse = detector.detector_parameters_get()
        return result

    @detector_app.get("/detector/query")
    async def detector_query() -> DetectorQueryResponse:
        return detector.detector_query()

    @detector_app.head("/detector/start")
    async def detector_start() -> None:
        detector.detector_start()

    @detector_app.head("/detector/stop")
    async def detector_stop() -> None:
        detector.detector_stop()

    @detector_app.post("/detector/start_time_sync")
    async def start_time_sync(
        request: TimeSyncStartRequest
    ) -> EmptyResponse:
        return detector.time_sync_start(
            request=request)
    
    @detector_app.post("/detector/stop_time_sync")
    async def stop_time_sync(
        request: TimeSyncStopRequest
    ) -> EmptyResponse:
        return detector.time_sync_stop(
            request=request)

    @detector_app.post("/detector/get_frame")
    async def detector_get_frame(
        request: DetectorFrameGetRequest
    ) -> DetectorFrameGetResponse:
        return detector.detector_frame_get(request=request)
    
    @detector_app.get("/detector/get_timestamp")
    async def get_timestamp() -> TimestampGetResponse:
        return detector.timestamp_get()

    @detector_app.get("/calibration/get_result_active")
    async def calibration_get_result_active() -> IntrinsicCalibrationResultGetActiveResponse:
        return detector.calibration_result_get_active()

    @detector_app.websocket("/websocket")
    async def websocket_handler(websocket: WebSocket) -> None:
        await detector.websocket_handler(websocket=websocket)

    @detector_app.on_event("startup")
    async def internal_update() -> None:
        await detector.update()
        asyncio.create_task(internal_update())

    return detector_app


app: FastAPI = create_app()
