from .api import \
    PoseSolverAddDetectorFrameRequest, \
    PoseSolverAddTargetRequest, \
    PoseSolverGetPosesResponse, \
    MixerUpdateIntrinsicParametersRequest
from .mixer import \
    Mixer
from src.common import \
    EmptyResponse, \
    ErrorResponse
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
import hjson
import logging
import os


# Note: This is the only implementation, currently.
from src.implementations.extrinsic_charuco_opencv import CharucoOpenCVExtrinsicCalibrator


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    configuration_filepath: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "configuration", "mixer_config.json")
    configuration: Mixer.Configuration
    with open(configuration_filepath, 'r') as infile:
        file_contents: str = infile.read()
        configuration_dict = hjson.loads(file_contents)
        configuration = Mixer.Configuration(**configuration_dict)

    mixer = Mixer(
        configuration=configuration,
        extrinsic_calibrator_type=CharucoOpenCVExtrinsicCalibrator)
    mixer_app = FastAPI()

    # CORS Middleware
    origins = ["http://localhost"]
    mixer_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    @mixer_app.post("/add_detector_frame")
    async def add_marker_corners(
        request: PoseSolverAddDetectorFrameRequest
    ) -> EmptyResponse | ErrorResponse:
        return mixer.pose_solver_add_detector_frame(request=request)

    @mixer_app.post("/add_target")
    async def add_target_marker(
        request: PoseSolverAddTargetRequest
    ) -> EmptyResponse | ErrorResponse:
        return mixer.pose_solver_add_target(request=request)

    @mixer_app.get("/get_poses")
    async def get_poses() -> PoseSolverGetPosesResponse | ErrorResponse:
        return mixer.pose_solver_get_poses()

    @mixer_app.post("/set_intrinsic_parameters")
    async def set_intrinsic_parameters(
        request: MixerUpdateIntrinsicParametersRequest
    ) -> EmptyResponse | ErrorResponse:
        return mixer.mixer_update_intrinsic_parameters(request=request)

    @mixer_app.head("/start_capture")
    async def start_capture() -> None:
        mixer.mixer_start()

    @mixer_app.head("/stop_capture")
    async def stop_capture() -> None:
        mixer.mixer_stop()

    @mixer_app.websocket("/websocket")
    async def websocket_handler(websocket: WebSocket) -> None:
        await mixer.websocket_handler(websocket=websocket)

    @mixer_app.on_event("startup")
    async def internal_update() -> None:
        await mixer.update()
        asyncio.create_task(internal_update())

    return mixer_app


app: FastAPI = create_app()
