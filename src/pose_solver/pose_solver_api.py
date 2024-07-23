from .api import \
    AddMarkerCornersRequest, \
    AddTargetMarkerRequest, \
    AddTargetMarkerResponse, \
    GetPosesRequest, \
    GetPosesResponse, \
    SetIntrinsicParametersRequest, \
    SetReferenceMarkerRequest, \
    StartPoseSolverRequest, \
    StopPoseSolverRequest
from .exceptions import PoseSolverException
from .pose_solver import PoseSolver
from .structures import \
    PoseSolverConfiguration, \
    TargetMarker
from src.common import \
    EmptyResponse, \
    ErrorResponse, \
    get_kwarg, \
    MCTComponent, \
    MCTRequest, \
    MCTResponse
from src.common.structures import \
    DetectorFrame, \
    Pose, \
    PoseSolverStatus
import logging
from typing import Callable


logger = logging.getLogger(__name__)


class PoseSolverAPI(MCTComponent):
    """
    API-friendly layer overtop of a PoseSolver
    """
    _status: PoseSolverStatus
    _pose_solver: PoseSolver

    def __init__(
        self,
        configuration: PoseSolverConfiguration,
        pose_solver: PoseSolver
    ):
        super().__init__(
            status_source_label=configuration.serial_identifier,
            send_status_messages_to_logger=True)
        self._pose_solver = pose_solver
        self._status = PoseSolverStatus()

    def add_detector_frame(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: AddMarkerCornersRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=AddMarkerCornersRequest)
        try:
            self._pose_solver.add_detector_frame(
                detector_label=request.detector_label,
                detector_frame=DetectorFrame(
                    detected_marker_snapshots=request.detected_marker_snapshots,
                    rejected_marker_snapshots=request.rejected_marker_snapshots,
                    timestamp_utc_iso8601=request.detector_timestamp_utc_iso8601))
        except PoseSolverException as e:
            return ErrorResponse(message=e.message)
        return EmptyResponse()

    def add_target(self, **kwargs) -> AddTargetMarkerResponse | ErrorResponse:
        request: AddTargetMarkerRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=AddTargetMarkerRequest)
        try:
            self._pose_solver.add_target(
                target=TargetMarker(
                    target_id=str(request.marker_id),
                    marker_id=request.marker_id,
                    marker_diameter=request.marker_diameter))
        except PoseSolverException as e:
            return ErrorResponse(message=e.message)
        return AddTargetMarkerResponse(target_id=str(request.marker_id))

    def get_poses(self, **_kwargs) -> GetPosesResponse | ErrorResponse:
        detector_poses: list[Pose]
        target_poses: list[Pose]
        try:
            detector_poses, target_poses = self._pose_solver.get_poses()
        except PoseSolverException as e:
            return ErrorResponse(message=e.message)
        return GetPosesResponse(
            detector_poses=detector_poses,
            target_poses=target_poses)

    def set_intrinsic_parameters(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: SetIntrinsicParametersRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=SetIntrinsicParametersRequest)
        try:
            self._pose_solver.set_intrinsic_parameters(
                detector_label=request.detector_label,
                intrinsic_parameters=request.intrinsic_parameters)
        except PoseSolverException as e:
            return ErrorResponse(message=e.message)
        return EmptyResponse()

    def set_reference_marker(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: SetReferenceMarkerRequest = get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=SetReferenceMarkerRequest)
        try:
            self._pose_solver.set_reference_target(target_id=str(request.marker_id))
        except PoseSolverException as e:
            return ErrorResponse(message=e.message)
        return EmptyResponse()

    def start_pose_solver(self, **_kwargs) -> EmptyResponse:
        self._status.solve_status = PoseSolverStatus.Solve.RUNNING
        return EmptyResponse()

    def stop_pose_solver(self, **_kwargs) -> EmptyResponse:
        self._status.solve_status = PoseSolverStatus.Solve.STOPPED
        return EmptyResponse()

    def supported_request_types(self) -> dict[type[MCTRequest], Callable[[dict], MCTResponse]]:
        return_value: dict[type[MCTRequest], Callable[[dict], MCTResponse]] = super().supported_request_types()
        return_value.update({
            AddMarkerCornersRequest: self.add_detector_frame,
            AddTargetMarkerRequest: self.add_target,
            GetPosesRequest: self.get_poses,
            SetIntrinsicParametersRequest: self.set_intrinsic_parameters,
            SetReferenceMarkerRequest: self.set_reference_marker,
            StartPoseSolverRequest: self.start_pose_solver,
            StopPoseSolverRequest: self.stop_pose_solver})
        return return_value

    async def update(self):
        if self._status.solve_status == PoseSolverStatus.Solve.RUNNING:
            self._pose_solver.update()
