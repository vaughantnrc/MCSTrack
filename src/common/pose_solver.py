from .image_processing import Annotation
from .math import \
    IntrinsicParameters, \
    IterativeClosestPointParameters, \
    MathUtils, \
    Matrix4x4, \
    Pose, \
    Ray, \
    Target
from .status import MCTError
import cv2
import cv2.aruco
import datetime
import itertools
import numpy
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation
from typing import Final, TypeVar


EPSILON: Final[float] = 0.0001
_CORNER_COUNT: Final[int] = 4

KeyType = TypeVar("KeyType")
ValueType = TypeVar("ValueType")


class _DetectorRecord:
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
                self._timestamped_annotations[annotation.feature_label] = _DetectorRecord.TimestampedAnnotation(
                    annotation=annotation,
                    timestamp_utc=frame_timestamp_utc)
                continue
            timestamped_annotation: _DetectorRecord.TimestampedAnnotation = \
                self._timestamped_annotations[annotation.feature_label]
            if frame_timestamp_utc > timestamped_annotation.timestamp_utc:
                self._timestamped_annotations[annotation.feature_label] = _DetectorRecord.TimestampedAnnotation(
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
        entry: _DetectorRecord.TimestampedAnnotation
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


class PoseSolver:
    """
    Class containing the actual "solver" logic, kept separate from the API.
    """

    class Configuration(BaseModel):
        minimum_detector_count: int = Field(default=2)
        ray_intersection_maximum_distance: float = Field(default=10.0, description="millimeters")
        icp_termination_iteration_count: int = Field(default=50)
        icp_termination_translation: float = Field(default=0.005, description="millimeters")
        icp_termination_rotation_radians: float = Field(default=0.0005)
        icp_termination_mean_point_distance: float = Field(default=0.1, description="millimeters")
        icp_termination_rms_point_distance: float = Field(default=0.1, description="millimeters")

        denoise_outlier_maximum_distance: float = Field(default=10.0)
        denoise_outlier_maximum_angle_degrees: float = Field(default=5.0)
        denoise_storage_size: int = Field(default=10)
        denoise_filter_size: int = Field(default=7)
        denoise_required_starting_streak: int = Field(default=3)

        # aruco_pose_estimator_method: int
        #   SOLVEPNP_ITERATIVE works okay but is susceptible to optical illusions (flipping)
        #   SOLVEPNP_P3P appears to return nan's on rare occasion
        #   SOLVEPNP_SQPNP appears to return nan's on rare occasion
        #   SOLVEPNP_IPPE_SQUARE does not seem to work very well at all, translation is much smaller than expected

    # bookkeeping
    _last_change_timestamp_utc: datetime.datetime
    _last_updated_timestamp_utc: datetime.datetime

    # inputs
    _configuration: Configuration
    _intrinsics_by_detector_label: dict[str, IntrinsicParameters]
    _extrinsics_by_detector_label: dict[str, Matrix4x4]
    _targets: list[Target]  # First target is considered the "reference"
    # input per frame
    _detector_records_by_detector_label: dict[str, _DetectorRecord]

    # use this to make sure each marker is associated uniquely to a single target
    _landmark_target_map: dict[str, Target]  # Each marker shall be used at most once by a single target

    # outputs
    _poses_by_target_label: dict[str, Matrix4x4]
    _poses_by_detector_label: dict[str, Matrix4x4]

    def __init__(
        self
    ):
        self._last_change_timestamp_utc = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
        self._last_updated_timestamp_utc = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

        self._configuration = PoseSolver.Configuration()
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
            self._detector_records_by_detector_label[detector_label] = _DetectorRecord()
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

    def _calculate_reprojection_error_for_pose(
        self,
        ray_set: ...,
        object_points_target: list[list[float]],
        object_to_reference_translation: list[float],
        object_to_reference_rotation_quaternion: list[float]
    ) -> float:
        object_to_reference_matrix = numpy.identity(4, dtype="float32")
        # noinspection PyArgumentList
        object_to_reference_matrix[0:3, 0:3] = Rotation.from_quat(object_to_reference_rotation_quaternion).as_matrix()
        object_to_reference_matrix[0:3, 3] = object_to_reference_translation
        object_points_reference = numpy.empty((len(object_points_target), 3), dtype="float32")
        for object_point_index, object_point_target in enumerate(object_points_target):
            object_point_reference = numpy.matmul(
                object_to_reference_matrix,
                [object_point_target[0], object_point_target[1], object_point_target[2], 1.0])
            object_points_reference[object_point_index, 0:3] = object_point_reference[0:3]
        detector_label: str = ray_set.detector_label
        reference_to_detector_matrix: Matrix4x4 = ray_set.detector_to_reference_matrix
        reference_to_detector: numpy.ndarray = reference_to_detector_matrix.as_numpy_array()
        # noinspection PyArgumentList
        reference_to_detector_rotation_vector = \
            Rotation.as_rotvec(Rotation.from_matrix(reference_to_detector[0:3, 0:3]))
        reference_to_detector_translation = reference_to_detector[0:3, 3]
        camera_matrix = numpy.array(self._intrinsics_by_detector_label[detector_label].get_matrix(), dtype="float32")
        camera_distortion_coefficients = \
            numpy.array(self._intrinsics_by_detector_label[detector_label].get_distortion_coefficients(),
                        dtype="float32")
        project_points_result = cv2.projectPoints(
            objectPoints=object_points_reference,
            rvec=reference_to_detector_rotation_vector,
            tvec=reference_to_detector_translation,
            cameraMatrix=camera_matrix,
            distCoeffs=camera_distortion_coefficients)
        projected_points = project_points_result[0]
        sum_reprojection_errors_squared: float = 0.0
        for point_index, image_point in enumerate(ray_set.image_points):
            reprojection_error_for_point = \
                numpy.linalg.norm(projected_points[point_index, 0, 0:2] - image_point)
            sum_reprojection_errors_squared += float(reprojection_error_for_point) ** 2
        mean_reprojection_errors_squared: float = sum_reprojection_errors_squared / len(object_points_target)
        rms_reprojection_error = numpy.sqrt(mean_reprojection_errors_squared)
        return rms_reprojection_error

    def _denoise_is_pose_pair_outlier(
        self,
        pose_a_object_to_reference_matrix: numpy.ndarray,
        pose_b_object_to_reference_matrix: numpy.ndarray
    ) -> bool:

        position_a = pose_a_object_to_reference_matrix[0:3, 3]
        position_b = pose_b_object_to_reference_matrix[0:3, 3]
        distance_millimeters = numpy.linalg.norm(position_a - position_b)
        if distance_millimeters > self._configuration.denoise_outlier_maximum_distance:
            return True

        orientation_a = pose_a_object_to_reference_matrix[0:3, 0:3]
        orientation_b = pose_b_object_to_reference_matrix[0:3, 0:3]
        rotation_a_to_b = numpy.matmul(orientation_a, numpy.linalg.inv(orientation_b))
        # noinspection PyArgumentList
        angle_degrees = numpy.linalg.norm(Rotation.from_matrix(rotation_a_to_b).as_rotvec())
        if angle_degrees > self._configuration.denoise_outlier_maximum_angle_degrees:
            return True

        return False

    # Take an average of recent poses, with detection and removal of outliers
    # TODO: Currently not used - but it's probably a good feature to have
    def _denoise_detector_to_reference_pose(
        self,
        object_label: str,
        raw_poses: list[...]  # In order, oldest to newest
    ) -> ...:
        most_recent_pose = raw_poses[-1]
        max_storage_size: int = self._configuration.denoise_storage_size
        filter_size: int = self._configuration.denoise_filter_size
        if filter_size <= 1 or max_storage_size <= 1:
            return most_recent_pose  # trivial case

        # find a consistent range of recent indices
        poses: list[...] = list(raw_poses)
        poses.reverse()  # now they are sorted so that the first element is most recent
        required_starting_streak: int = self._configuration.denoise_required_starting_streak
        starting_index: int = -1  # not yet known, we want to find this
        if required_starting_streak <= 1:
            starting_index = 0  # trivial case
        else:
            current_inlier_streak: int = 1  # comparison always involves a streak of size 1 or more
            for pose_index in range(1, len(raw_poses)):
                if self._denoise_is_pose_pair_outlier(poses[pose_index - 1], poses[pose_index]):
                    current_inlier_streak = 1
                    continue
                current_inlier_streak += 1
                if current_inlier_streak >= required_starting_streak:
                    starting_index = pose_index - required_starting_streak + 1
                    break

        if starting_index < 0:
            if len(poses) >= max_storage_size:  # There appear to be enough poses, so data seems to be inconsistent
                print(
                    "Warning: Can't find consistent pose streak. "
                    "Will use most recent raw pose for object " + object_label + ".")
            return most_recent_pose

        poses_to_average: list[...] = [poses[starting_index]]
        for pose_index in range(starting_index + 1, len(poses)):
            if (not self._denoise_is_pose_pair_outlier(poses[pose_index - 1], poses[pose_index])) or \
               (not self._denoise_is_pose_pair_outlier(poses[starting_index], poses[pose_index])):
                poses_to_average.append(poses[pose_index])
                if len(poses_to_average) > filter_size:
                    break

        translations = [list(pose.object_to_reference_matrix[0:3, 3])
                        for pose in poses_to_average]
        # noinspection PyArgumentList
        orientations = [list(Rotation.from_matrix(pose.object_to_reference_matrix[0:3, 0:3]).as_quat(canonical=True))
                        for pose in poses_to_average]
        filtered_translation = MathUtils.average_vector(translations)
        filtered_orientation = MathUtils.average_quaternion(orientations)
        filtered_object_to_reference_matrix = numpy.identity(4, dtype="float32")
        # noinspection PyArgumentList
        filtered_object_to_reference_matrix[0:3, 0:3] = Rotation.from_quat(filtered_orientation).as_matrix()
        filtered_object_to_reference_matrix[0:3, 3] = filtered_translation
        # return PoseData(
        #     object_label=object_label,
        #     object_to_reference_matrix=filtered_object_to_reference_matrix,
        #     ray_sets=most_recent_pose.ray_sets)

    def update(self) -> None:
        if self._last_updated_timestamp_utc >= self._last_change_timestamp_utc:
            return

        if len(self._targets) == 0:
            return

        self._last_updated_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)
        self._poses_by_detector_label.clear()
        self._poses_by_target_label.clear()

        annotation_list_by_detector_label: dict[str, list[Annotation]]
        annotation_list_by_detector_label = {
            detector_label: detector_record.get_annotations(deep_copy=True)
            for detector_label, detector_record in self._detector_records_by_detector_label.items()}
        detector_labels: list[str] = list(self._detector_records_by_detector_label.keys())

        for detector_label in detector_labels:
            if detector_label not in self._intrinsics_by_detector_label:
                # TODO: Output a suitable warning that no intrinsics have been received, but don't do it every frame
                del annotation_list_by_detector_label[detector_label]

        reference_target: Target = self._targets[0]
        for detector_label in detector_labels:
            if detector_label in self._extrinsics_by_detector_label:
                self._poses_by_detector_label[detector_label] = self._extrinsics_by_detector_label[detector_label]
            else:
                intrinsics: IntrinsicParameters = self._intrinsics_by_detector_label[detector_label]
                estimated: bool
                reference_to_detector: Matrix4x4 | None
                estimated, reference_to_detector = MathUtils.estimate_matrix_transform_to_detector(
                    annotations=annotation_list_by_detector_label[detector_label],
                    landmarks=reference_target.landmarks,
                    detector_intrinsics=intrinsics)
                if estimated:
                    detector_to_reference: Matrix4x4 = Matrix4x4.from_numpy_array(
                        numpy.linalg.inv(reference_to_detector.as_numpy_array()))
                    self._poses_by_detector_label[detector_label] = detector_to_reference

        # At the time of writing, each feature label can be used only by one target.
        # So we can remove annotations whose feature labels match those of the reference_target
        # to avoid unnecessary processing.
        reference_feature_labels: set[str] = set([landmark.feature_label for landmark in reference_target.landmarks])
        for detector_label in detector_labels:
            indices_to_remove: list[int] = list()
            for annotation_index, annotation in enumerate(annotation_list_by_detector_label[detector_label]):
                if annotation.feature_label in reference_feature_labels:
                    indices_to_remove.append(annotation_index)
            for annotation_index in reversed(indices_to_remove):
                annotation_list_by_detector_label[detector_label].pop(annotation_index)

        # Convert annotations to rays
        rays_by_feature_and_detector: dict[str, dict[str, Ray]] = dict()  # indexed as [feature_label][detector_label]
        for detector_label in detector_labels:
            if detector_label not in self._poses_by_detector_label:
                continue
            annotations: list[Annotation] = annotation_list_by_detector_label[detector_label]
            annotation_points: list[list[float]] = [[annotation.x_px, annotation.y_px] for annotation in annotations]
            detector_to_reference: Matrix4x4 = self._poses_by_detector_label[detector_label]
            intrinsics: IntrinsicParameters = self._intrinsics_by_detector_label[detector_label]
            ray_origin: list[float] = detector_to_reference.get_translation()
            ray_directions = MathUtils.convert_detector_points_to_vectors(
                points=annotation_points,
                detector_intrinsics=intrinsics,
                detector_to_reference_matrix=detector_to_reference)
            feature_labels: list[str] = [annotation.feature_label for annotation in annotations]
            # note: annotation_labels and ray_directions have a 1:1 correspondence by index
            assert len(ray_directions) == len(feature_labels)
            for feature_index, feature_label in enumerate(feature_labels):
                if feature_label not in rays_by_feature_and_detector:
                    rays_by_feature_and_detector[feature_label] = dict()
                rays_by_feature_and_detector[feature_label][detector_label] = Ray(
                    source_point=ray_origin,
                    direction=ray_directions[feature_index])

        # intersect rays to find the 3D points for each feature, in reference coordinates
        # If intersection is not possible, then still note that rays exist via standalone_ray_feature_labels
        intersections_by_feature_label: dict[str, list[float]] = dict()  # [feature_label][dimension_index]
        feature_labels_with_rays_only: list[str] = list()
        for feature_label, rays_by_detector_label in rays_by_feature_and_detector.items():
            intersection_result = MathUtils.closest_intersection_between_n_lines(
                rays=list(rays_by_detector_label.values()),
                maximum_distance=self._configuration.ray_intersection_maximum_distance)
            if intersection_result.centroids.shape[0] == 0:
                feature_labels_with_rays_only.append(feature_label)
                break
            intersections_by_feature_label[feature_label] = list(intersection_result.centroid().flatten())

        # We estimate the pose of each target based on the calculated intersections
        # and the rays projected from each detector
        for target in self._targets:
            if target.label == str(reference_target.label):
                continue  # everything is expressed relative to the reference, so it's a "known" coordinate system
            feature_labels_in_target: list[str] = [landmark.feature_label for landmark in target.landmarks]

            target_feature_labels_with_intersections: list[str] = list()
            target_feature_labels_with_rays: list[str] = list()
            detector_labels_seeing_target: set[str] = set()
            for target_feature_label in feature_labels_in_target:
                if target_feature_label in intersections_by_feature_label:
                    target_feature_labels_with_intersections.append(target_feature_label)
                if target_feature_label in feature_labels_with_rays_only:
                    target_feature_labels_with_rays.append(target_feature_label)
                detector_labels_seeing_target |= set(rays_by_feature_and_detector[target_feature_label].keys())

            if len(target_feature_labels_with_intersections) <= 0 and len(target_feature_labels_with_rays) <= 0:
                continue  # No information on which to base a pose

            detector_count_seeing_target: int = len(detector_labels_seeing_target)
            if detector_count_seeing_target < self._configuration.minimum_detector_count or \
               detector_count_seeing_target <= 0:
                continue

            one_detector_only: bool = (len(detector_labels_seeing_target) == 1)
            if one_detector_only:
                # Note: there cannot be any intersections in this case
                detector_label: str = next(iter(detector_labels_seeing_target))
                intrinsics: IntrinsicParameters = self._intrinsics_by_detector_label[detector_label]
                estimated: bool
                detected_to_detector_matrix4x4: Matrix4x4
                estimated, detected_to_detector_matrix4x4 = MathUtils.estimate_matrix_transform_to_detector(
                    annotations=annotation_list_by_detector_label[detector_label],
                    landmarks=target.landmarks,
                    detector_intrinsics=intrinsics)
                if estimated:
                    detected_to_detector: numpy.ndarray = detected_to_detector_matrix4x4.as_numpy_array()
                    detector_to_reference: numpy.ndarray = self._poses_by_detector_label[detector_label].as_numpy_array()
                    detected_to_reference: numpy.ndarray = detector_to_reference @ detected_to_detector
                    self._poses_by_target_label[target.label] = Matrix4x4.from_numpy_array(detected_to_reference)
            else:
                # Fill in the required variables for the customized iterative closest point
                detected_known_points: list[list[float]] = [
                    target.get_landmark_point(feature_label)
                    for feature_label in target_feature_labels_with_intersections]
                reference_known_points: list[list[float]] = [
                    intersections_by_feature_label[feature_label]
                    for feature_label in target_feature_labels_with_intersections]
                detected_ray_points: list[list[float]] = [
                    target.get_landmark_point(feature_label)
                    for feature_label in target_feature_labels_with_rays]
                reference_rays: list[Ray] = list(itertools.chain.from_iterable([
                    list(rays_by_feature_and_detector[feature_label].values())
                    for feature_label in target_feature_labels_with_rays]))
                iterative_closest_point_parameters = IterativeClosestPointParameters(
                    termination_iteration_count=self._configuration.icp_termination_iteration_count,
                    termination_delta_translation=self._configuration.icp_termination_translation,
                    termination_delta_rotation_radians=self._configuration.icp_termination_rotation_radians,
                    termination_mean_point_distance=self._configuration.icp_termination_mean_point_distance,
                    termination_rms_point_distance=self._configuration.icp_termination_rms_point_distance)
                if len(target_feature_labels_with_intersections) >= 1:
                    initial_detected_to_reference_matrix = MathUtils.register_corresponding_points(
                        point_set_from=detected_known_points,
                        point_set_to=reference_known_points,
                        collinearity_do_check=False)
                    icp_output = MathUtils.iterative_closest_point_for_points_and_rays(
                        source_known_points=detected_known_points,
                        target_known_points=reference_known_points,
                        source_ray_points=detected_ray_points,
                        target_rays=reference_rays,
                        initial_transformation_matrix=initial_detected_to_reference_matrix,
                        parameters=iterative_closest_point_parameters)
                else:
                    icp_output = MathUtils.iterative_closest_point_for_points_and_rays(
                        source_known_points=detected_known_points,
                        target_known_points=reference_known_points,
                        source_ray_points=detected_ray_points,
                        target_rays=reference_rays,
                        parameters=iterative_closest_point_parameters)
                self._poses_by_target_label[target.label] = icp_output.source_to_target_matrix
