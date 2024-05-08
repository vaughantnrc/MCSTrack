from src.common import \
    client_identifier_from_connection, \
    EmptyResponse, \
    ErrorResponse
from . import Calibrator
from .api import \
    AddCalibrationImageRequest, \
    AddCalibrationImageResponse, \
    CalibrateRequest, \
    CalibrateResponse, \
    DeleteStagedRequest, \
    ListCalibrationImageMetadataRequest, \
    ListCalibrationImageMetadataResponse, \
    GetCalibrationImageRequest, \
    GetCalibrationImageResponse, \
    ListCalibrationResultMetadataRequest, \
    ListCalibrationResultMetadataResponse, \
    GetCalibrationResultRequest, \
    GetCalibrationResultResponse, \
    UpdateCalibrationImageMetadataRequest, \
    UpdateCalibrationResultMetadataRequest
from .fileio import CalibratorConfiguration
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket
import hjson
import logging
import os


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    configuration_filepath: str = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "calibrator_config.json")
    configuration: CalibratorConfiguration
    with open(configuration_filepath, 'r') as infile:
        file_contents: str = infile.read()
        configuration_dict = hjson.loads(file_contents)
        configuration = CalibratorConfiguration(**configuration_dict)
    calibrator = Calibrator(calibrator_configuration=configuration)
    calibrator_app = FastAPI()

    # CORS Middleware
    origins = ["http://localhost"]
    calibrator_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"])

    @calibrator_app.post("/add_image")
    async def add_calibration_image(
        request_body: AddCalibrationImageRequest
    ) -> AddCalibrationImageResponse | ErrorResponse:
        return calibrator.add_calibration_image(request=request_body)

    @calibrator_app.post("/calibrate")
    async def calibrate(
        request_body: CalibrateRequest
    ) -> CalibrateResponse | ErrorResponse:
        return calibrator.calibrate(request=request_body)

    @calibrator_app.post("/delete_staged")
    async def delete_staged(
        request_body: DeleteStagedRequest
    ) -> EmptyResponse | ErrorResponse:
        return calibrator.delete_staged(request=request_body)

    @calibrator_app.post("/get_image")
    async def get_image(
        request_body: GetCalibrationImageRequest
    ) -> GetCalibrationImageResponse | ErrorResponse:
        return calibrator.get_calibration_image(request=request_body)

    @calibrator_app.post("/get_result")
    async def get_result(
        request_body: GetCalibrationResultRequest
    ) -> GetCalibrationResultResponse | ErrorResponse:
        return calibrator.get_calibration_result(request=request_body)

    @calibrator_app.post("/list_image_metadata")
    async def list_image_metadata(
        request_body: ListCalibrationImageMetadataRequest
    ) -> ListCalibrationImageMetadataResponse:
        return calibrator.list_calibration_image_metadata_list(request=request_body)

    @calibrator_app.post("/list_result_metadata")
    async def list_result_metadata(
        request_body: ListCalibrationResultMetadataRequest
    ) -> ListCalibrationResultMetadataResponse:
        return calibrator.list_calibration_result_metadata_list(request=request_body)

    @calibrator_app.post("/set_image_state")
    async def set_calibration_image_state(
        request_body: UpdateCalibrationImageMetadataRequest
    ) -> EmptyResponse | ErrorResponse:
        return calibrator.update_calibration_image_metadata(request=request_body)

    @calibrator_app.post("/set_result_state")
    async def set_calibration_result_state(
        request_body: UpdateCalibrationResultMetadataRequest
    ) -> EmptyResponse | ErrorResponse:
        return calibrator.update_calibration_result_metadata(request=request_body)

    @calibrator_app.websocket("/websocket")
    async def websocket_handler(websocket: WebSocket) -> None:
        await calibrator.websocket_handler(websocket=websocket)

    return calibrator_app


app = create_app()
