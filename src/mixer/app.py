from .api import \
    PoseSolverDetectorFrameAddRequest, \
    PoseSolverTargetAddRequest, \
    PoseSolverPosesGetResponse, \
    MixerQueryResponse, \
    MixerIntrinsicUpdateRequest
from .mixer import \
    Mixer
from src.common import \
    EmptyResponse, \
    ErrorResponse, \
    ExtrinsicCalibrator, \
    PoseSolver
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
import hjson
import logging
import os
from typing import Final


logger = logging.getLogger(__name__)

CONFIGURATION_FILEPATH_ENV_VAR: Final[str] = "MCSTRACK_MIXER_CONFIGURATION_FILEPATH"


def create_app() -> FastAPI:
    configuration_filepath: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "configuration", "mixer", "aruco.json")
    configuration_filepath = os.getenv(CONFIGURATION_FILEPATH_ENV_VAR, configuration_filepath)
    configuration: Mixer.Configuration
    with open(configuration_filepath, 'r') as infile:
        file_contents: str = infile.read()
        configuration_dict = hjson.loads(file_contents)
        configuration = Mixer.Configuration(**configuration_dict)

    # Eventually it would be preferable to put the initialization logic/mapping below into an abstract factory,
    # and allow end-users to register custom classes that are not necessarily shipped within this library.

    pose_solver_type: type[PoseSolver]
    if configuration.pose_solver.implementation == "standard":
        from src.implementations.pose_solver_standard import StandardPoseSolver
        pose_solver_type = StandardPoseSolver
    elif configuration.pose_solver.implementation == "mock":
        from src.implementations.pose_solver_mock import MockPoseSolver
        pose_solver_type = MockPoseSolver
    else:
        raise RuntimeError(f"Unsupported pose solver implementation {configuration.pose_solver.implementation}.")

    extrinsic_calibrator_type: type[ExtrinsicCalibrator]
    if configuration.extrinsic_calibrator.implementation == "charuco_opencv":
        from src.implementations.extrinsic_charuco_opencv import CharucoOpenCVExtrinsicCalibrator
        extrinsic_calibrator_type = CharucoOpenCVExtrinsicCalibrator
    elif configuration.extrinsic_calibrator.implementation == "mock":
        from src.implementations.extrinsic_mock import MockExtrinsicCalibrator
        extrinsic_calibrator_type = MockExtrinsicCalibrator
    else:
        raise RuntimeError(f"Unsupported intrinsic implementation {configuration.camera.implementation}.")

    mixer = Mixer(
        configuration=configuration,
        pose_solver_type=pose_solver_type,
        extrinsic_calibrator_type=extrinsic_calibrator_type)
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
        request: PoseSolverDetectorFrameAddRequest
    ) -> EmptyResponse | ErrorResponse:
        return mixer.pose_solver_detector_frame_add(request=request)

    @mixer_app.post("/add_target")
    async def add_target_marker(
        request: PoseSolverTargetAddRequest
    ) -> EmptyResponse | ErrorResponse:
        return mixer.pose_solver_target_add(request=request)

    @mixer_app.get("/get_poses")
    async def get_poses() -> PoseSolverPosesGetResponse | ErrorResponse:
        return mixer.pose_solver_poses_get()

    @mixer_app.post("/set_intrinsic_parameters")
    async def set_intrinsic_parameters(
        request: MixerIntrinsicUpdateRequest
    ) -> EmptyResponse | ErrorResponse:
        return mixer.mixer_update_intrinsic_parameters(request=request)

    @mixer_app.get("/query")
    async def query() -> MixerQueryResponse | ErrorResponse:
        return mixer.mixer_query()

    @mixer_app.head("/start")
    async def start() -> None:
        mixer.mixer_start()

    @mixer_app.head("/stop")
    async def stop() -> None:
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
