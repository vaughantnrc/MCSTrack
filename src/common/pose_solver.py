from .image_processing import Annotation
from .math import \
    IntrinsicParameters, \
    Matrix4x4, \
    Pose, \
    Target
from .status import \
    MCTError, \
    StatusMessageSource
import abc
import datetime
from enum import StrEnum
from pydantic import BaseModel


class _Configuration(BaseModel):  pass


class DetectorPoseMode(StrEnum):
    STATIC_EXTERNAL = "static_external"
    STATIC_CALIBRATED = "static_calibrated"
    DYNAMIC_TARGET = "dynamic_target"

    @staticmethod
    def default_mode() -> 'DetectorPoseMode':
        return DetectorPoseMode.DYNAMIC_TARGET


class PoseSolverDetectorRecord:
    """
    Class whose purpose is to keep track of the latest position of each landmark (in annotation form)
    for a single detector.
    """

    class TimestampedAnnotation:
        annotation: Annotation
        timestamp_utc: datetime.datetime
        def __init__(
            self,
            annotation: Annotation,
            timestamp_utc: datetime.datetime
        ):
            self.annotation = annotation
            self.timestamp_utc = timestamp_utc

    _timestamped_annotations: dict[str, TimestampedAnnotation]

    def __init__(self):
        self._timestamped_annotations = dict()

    def add_frame_record(
        self,
        frame_annotations: list[Annotation],
        frame_timestamp_utc: datetime.datetime
    ) -> None:
        for annotation in frame_annotations:
            if annotation.feature_label not in self._timestamped_annotations:
                self._timestamped_annotations[annotation.feature_label] = PoseSolverDetectorRecord.TimestampedAnnotation(
                    annotation=annotation,
                    timestamp_utc=frame_timestamp_utc)
                continue
            timestamped_annotation: PoseSolverDetectorRecord.TimestampedAnnotation = \
                self._timestamped_annotations[annotation.feature_label]
            if frame_timestamp_utc > timestamped_annotation.timestamp_utc:
                self._timestamped_annotations[annotation.feature_label] = PoseSolverDetectorRecord.TimestampedAnnotation(
                    annotation=annotation,
                    timestamp_utc=frame_timestamp_utc)

    def clear_frame_records(self):
        self._timestamped_annotations.clear()

    def clear_frame_records_older_than(
        self,
        timestamp_utc: datetime.datetime
    ) -> bool:
        """
        returns True if any changes were made
        """
        feature_labels_to_remove: list[str] = list()
        entry: PoseSolverDetectorRecord.TimestampedAnnotation
        for entry in self._timestamped_annotations.values():
            if entry.timestamp_utc < timestamp_utc:
                feature_labels_to_remove.append(entry.annotation.feature_label)
        if len(feature_labels_to_remove) <= 0:
            return False
        for feature_label in feature_labels_to_remove:
            del self._timestamped_annotations[feature_label]
        return True

    def get_annotations(
        self,
        deep_copy: bool = True
    ) -> list[Annotation]:
        if deep_copy:
            return [entry.annotation.model_copy() for entry in self._timestamped_annotations.values()]
        return [entry.annotation for entry in self._timestamped_annotations.values()]


class PoseSolverException(MCTError):
    message: str

    def __init__(self, message: str, *args, **kwargs):
        super().__init__(args, kwargs)
        self.message = message


class PoseSolver(abc.ABC):
    """
    Class containing the actual "solver" logic, kept separate from the API.
    """

    Configuration: type[_Configuration] = _Configuration

    _configuration: Configuration
    _status_message_source: StatusMessageSource

    # bookkeeping
    _last_change_timestamp_utc: datetime.datetime
    _last_updated_timestamp_utc: datetime.datetime

    # inputs
    _intrinsics_by_detector_label: dict[str, IntrinsicParameters]
    _extrinsics_by_detector_label: dict[str, Matrix4x4]
    _targets: list[Target]  # First target is considered the "reference"
    # input per frame
    _detector_records_by_detector_label: dict[str, PoseSolverDetectorRecord]

    # use this to make sure each marker is associated uniquely to a single target
    _landmark_target_map: dict[str, Target]  # Each marker shall be used at most once by a single target

    # outputs
    _poses_by_target_label: dict[str, Matrix4x4]
    _poses_by_detector_label: dict[str, Matrix4x4]

    def __init__(
        self,
        configuration: Configuration,
        status_message_source: StatusMessageSource
    ):
        self._configuration = configuration
        self._status_message_source = status_message_source

        self._last_change_timestamp_utc = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        self._last_updated_timestamp_utc = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

        self._intrinsics_by_detector_label = dict()
        self._extrinsics_by_detector_label = dict()
        self._targets = list()

        self._detector_records_by_detector_label = dict()

        self._landmark_target_map = dict()

        self._poses_by_target_label = dict()
        self._poses_by_detector_label = dict()

    def add_detector_frame(
        self,
        detector_label: str,
        frame_annotations: list[Annotation],
        frame_timestamp_utc: datetime.datetime
    ) -> None:
        if detector_label not in self._detector_records_by_detector_label:
            self._detector_records_by_detector_label[detector_label] = PoseSolverDetectorRecord()
        self._detector_records_by_detector_label[detector_label].clear_frame_records()
        self._detector_records_by_detector_label[detector_label].add_frame_record(
            frame_annotations=frame_annotations,
            frame_timestamp_utc=frame_timestamp_utc)
        self._last_change_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)

    def add_target(
        self,
        target: Target
    ) -> None:
        for existing_target in self._targets:
            if target.label == existing_target.label:
                raise PoseSolverException(
                    f"Target with name {target.label} is already registered. "
                    f"Please use a different name, and also make sure you are not adding the same target twice.")
        landmark_labels: list[str] = [landmark.feature_label for landmark in target.landmarks]
        for landmark_label in landmark_labels:
            if landmark_label in self._landmark_target_map:
                target_id: str = self._landmark_target_map[landmark_label].label
                raise PoseSolverException(
                    f"Landmark {landmark_label} is already used with target {target_id} and it cannot be reused.")
        target_index = len(self._targets)
        self._targets.append(target)
        for landmark_label in landmark_labels:
            self._landmark_target_map[landmark_label] = self._targets[target_index]
        self._last_change_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)

    def clear_extrinsic_matrices(self):
        self._extrinsics_by_detector_label.clear()
        self._last_change_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)

    def clear_intrinsic_parameters(self):
        self._intrinsics_by_detector_label.clear()
        self._last_change_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)

    def clear_targets(self):
        self._targets.clear()
        self._landmark_target_map.clear()
        self._last_change_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)

    def get_detector_frame_timestamp(self) -> datetime.datetime:
        return self._last_change_timestamp_utc  # TODO: track target and detector change timestamps separately

    def get_poses(
        self
    ) -> tuple[list[Pose], list[Pose]]:
        """
        Returns detector_poses, target_poses
        """
        detector_poses: list[Pose] = [
            Pose(
                target_id=detector_label,
                object_to_reference_matrix=pose,
                solver_timestamp_utc_iso8601=self._last_updated_timestamp_utc.isoformat())
            for detector_label, pose in self._poses_by_detector_label.items()]
        target_poses: list[Pose] = [
            Pose(
                target_id=str(target_id),
                object_to_reference_matrix=pose,
                solver_timestamp_utc_iso8601=self._last_updated_timestamp_utc.isoformat())
            for target_id, pose in self._poses_by_target_label.items()]
        return detector_poses, target_poses

    def list_targets(self) -> list[Target]:
        return self._targets

    def set_extrinsic_matrix(
        self,
        detector_label: str,
        transform_to_reference: Matrix4x4
    ) -> None:
        self._extrinsics_by_detector_label[detector_label] = transform_to_reference
        self._last_change_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)

    def set_intrinsic_parameters(
        self,
        detector_label: str,
        intrinsic_parameters: IntrinsicParameters
    ) -> None:
        self._intrinsics_by_detector_label[detector_label] = intrinsic_parameters
        self._last_change_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)

    def set_reference_target(
        self,
        target_id: str
    ) -> None:
        found: bool = False
        for target_index, target in enumerate(self._targets):
            if target.label == target_id:
                self._targets[0], self._targets[target_index] = self._targets[target_index], self._targets[0]
                self._last_change_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)
                found = True
                break
        if not found:
            raise PoseSolverException(f"{target_id} was not found.")

    def set_targets(
        self,
        targets: list[Target]
    ) -> None:
        self._targets = targets
        self._poses_by_target_label.clear()
        self._poses_by_detector_label.clear()
        self._last_change_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)

    @abc.abstractmethod
    def update(self) -> None:
        """
        This function is expected to calculate _poses_by_target_label and _poses_by_detector_label.
        Make sure to update _last_updated_timestamp_utc as well.
        """
        pass