from .api import \
    PoseSolverAddDetectorFrameRequest, \
    PoseSolverAddTargetMarkerRequest, \
    PoseSolverGetPosesRequest, \
    PoseSolverGetPosesResponse, \
    PoseSolverSetExtrinsicRequest, \
    PoseSolverSetIntrinsicRequest, \
    PoseSolverSetReferenceRequest, \
    PoseSolverSetTargetsRequest, \
    PoseSolverStartRequest, \
    PoseSolverStopRequest
from .exceptions import PoseSolverException
from .pose_solver import PoseSolver
from .structures import \
    PoseSolverConfiguration
from src.common import \
    EmptyResponse, \
    ErrorResponse, \
    MCTComponent, \
    MCTRequest, \
    MCTResponse
from src.common.structures import \
    Pose
from enum import StrEnum
import logging
from typing import Callable, Final


logger = logging.getLogger(__name__)


_ROLE_LABEL: Final[str] = "pose_solver"


class PoseSolverAPI(MCTComponent):
    """
    API-friendly layer overtop of a PoseSolver
    """

    class Status:

        class Solve(StrEnum):
            STOPPED: Final[int] = "stopped"
            RUNNING: Final[int] = "running"
            FAILURE: Final[int] = "failure"

        solve_status: Solve
        solve_errors: list[str]

        def __init__(self):
            self.solve_status = PoseSolverAPI.Status.Solve.STOPPED
            self.solve_errors = list()

    _status: Status
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
        self._status = PoseSolverAPI.Status()

    def add_detector_frame(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: PoseSolverAddDetectorFrameRequest = self.get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=PoseSolverAddDetectorFrameRequest)
        try:
            self._pose_solver.add_detector_frame(
                detector_label=request.detector_label,
                detector_frame=request.detector_frame)
        except PoseSolverException as e:
            return ErrorResponse(message=e.message)
        return EmptyResponse()

    def add_target(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: PoseSolverAddTargetMarkerRequest = self.get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=PoseSolverAddTargetMarkerRequest)
        try:
            self._pose_solver.add_target(target=request.target)
        except PoseSolverException as e:
            return ErrorResponse(message=e.message)
        return EmptyResponse()

    def get_poses(self, **_kwargs) -> PoseSolverGetPosesResponse | ErrorResponse:
        detector_poses: list[Pose]
        target_poses: list[Pose]
        try:
            detector_poses, target_poses = self._pose_solver.get_poses()
        except PoseSolverException as e:
            return ErrorResponse(message=e.message)
        return PoseSolverGetPosesResponse(
            detector_poses=detector_poses,
            target_poses=target_poses)

    @staticmethod
    def get_role_label():
        return _ROLE_LABEL

    def set_extrinsic_matrix(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: PoseSolverSetExtrinsicRequest = self.get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=PoseSolverSetExtrinsicRequest)
        try:
            self._pose_solver.set_extrinsic_matrix(
                detector_label=request.detector_label,
                transform_to_reference=request.transform_to_reference)
        except PoseSolverException as e:
            return ErrorResponse(message=e.message)
        return EmptyResponse()

    def set_intrinsic_parameters(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: PoseSolverSetIntrinsicRequest = self.get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=PoseSolverSetIntrinsicRequest)
        try:
            self._pose_solver.set_intrinsic_parameters(
                detector_label=request.detector_label,
                intrinsic_parameters=request.intrinsic_parameters)
        except PoseSolverException as e:
            return ErrorResponse(message=e.message)
        return EmptyResponse()

    def set_reference_marker(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: PoseSolverSetReferenceRequest = self.get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=PoseSolverSetReferenceRequest)
        try:
            self._pose_solver.set_reference_target(target_id=str(request.marker_id))
        except PoseSolverException as e:
            return ErrorResponse(message=e.message)
        return EmptyResponse()

    def set_targets(self, **kwargs) -> EmptyResponse | ErrorResponse:
        request: PoseSolverSetTargetsRequest = self.get_kwarg(
            kwargs=kwargs,
            key="request",
            arg_type=PoseSolverSetTargetsRequest)
        try:
            self._pose_solver.set_targets(targets=request.targets)
        except PoseSolverException as e:
            return ErrorResponse(message=e.message)
        return EmptyResponse()

    def start_pose_solver(self, **_kwargs) -> EmptyResponse:
        self._status.solve_status = PoseSolverAPI.Status.Solve.RUNNING
        return EmptyResponse()

    def stop_pose_solver(self, **_kwargs) -> EmptyResponse:
        self._status.solve_status = PoseSolverAPI.Status.Solve.STOPPED
        return EmptyResponse()

    def supported_request_types(self) -> dict[type[MCTRequest], Callable[[dict], MCTResponse]]:
        return_value: dict[type[MCTRequest], Callable[[dict], MCTResponse]] = super().supported_request_types()
        return_value.update({
            PoseSolverAddDetectorFrameRequest: self.add_detector_frame,
            PoseSolverAddTargetMarkerRequest: self.add_target,
            PoseSolverGetPosesRequest: self.get_poses,
            PoseSolverSetExtrinsicRequest: self.set_extrinsic_matrix,
            PoseSolverSetIntrinsicRequest: self.set_intrinsic_parameters,
            PoseSolverSetReferenceRequest: self.set_reference_marker,
            PoseSolverSetTargetsRequest: self.set_targets,
            PoseSolverStartRequest: self.start_pose_solver,
            PoseSolverStopRequest: self.stop_pose_solver})
        return return_value

    async def update(self):
        if self.time_sync_active:
            return
        if self._status.solve_status == PoseSolverAPI.Status.Solve.RUNNING:
            self._pose_solver.update()
