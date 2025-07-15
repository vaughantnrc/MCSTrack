from src.common import \
    Annotation, \
    DetectorFrame
import cv2.aruco
import datetime
from pydantic import BaseModel, Field


class DetectorRecord:
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
        frame: DetectorFrame
    ) -> None:
        for annotation in frame.annotations:
            if annotation.feature_label not in self._timestamped_annotations:
                self._timestamped_annotations[annotation.feature_label] = DetectorRecord.TimestampedAnnotation(
                    annotation=annotation,
                    timestamp_utc=frame.timestamp_utc)
                continue
            timestamped_annotation: DetectorRecord.TimestampedAnnotation = \
                self._timestamped_annotations[annotation.feature_label]
            if frame.timestamp_utc > timestamped_annotation.timestamp_utc:
                self._timestamped_annotations[annotation.feature_label] = DetectorRecord.TimestampedAnnotation(
                    annotation=annotation,
                    timestamp_utc=frame.timestamp_utc)

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


class PoseSolverConfiguration(BaseModel):
    serial_identifier: str = Field()


class PoseSolverParameters(BaseModel):
    minimum_detector_count: int = Field(default=2)
    MAXIMUM_RAY_COUNT_FOR_INTERSECTION: int = Field(2)
    POSE_MULTI_CAMERA_LIMIT_RAY_AGE_SECONDS: float = Field(0.1)
    POSE_SINGLE_CAMERA_EXTRAPOLATION_MINIMUM_SURFACE_NORMAL_ANGLE_DEGREES: float = Field(15.0)
    POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_RAY_AGE_SECONDS: float = Field(1.0)
    POSE_SINGLE_CAMERA_EXTRAPOLATION_MAXIMUM_ORDER: int = Field(0)
    POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_ANGLE_DEGREES: float = Field(15.0)
    POSE_SINGLE_CAMERA_EXTRAPOLATION_LIMIT_DISTANCE: float = Field(15.0, description="millimeters")
    POSE_SINGLE_CAMERA_NEAREST_LIMIT_RAY_AGE_SECONDS: float = Field(0.8)
    POSE_SINGLE_CAMERA_NEAREST_LIMIT_ANGLE_DEGREES: float = Field(15.0)
    POSE_SINGLE_CAMERA_NEAREST_LIMIT_DISTANCE: float = Field(15.0)
    POSE_SINGLE_CAMERA_REPROJECTION_ERROR_FACTOR_BETA_OVER_ALPHA: float = Field(1.0)
    POSE_SINGLE_CAMERA_DEPTH_LIMIT_AGE_SECONDS: float = Field(0.4)
    # TODO: Is this next one detector-specific?
    POSE_SINGLE_CAMERA_DEPTH_CORRECTION: float = Field(-7.5, description="millimeters, observed tendency to overestimate depth.")
    POSE_DETECTOR_DENOISE_LIMIT_AGE_SECONDS: float = Field(1.0)
    INTERSECTION_MAXIMUM_DISTANCE: float = Field(10.0, description="millimeters")
    icp_termination_iteration_count: int = Field(50)
    icp_termination_translation: float = Field(0.005, description="millimeters")
    icp_termination_rotation_radians: float = Field(0.0005)
    icp_termination_mean_point_distance: float = Field(0.1, description="millimeters")
    icp_termination_rms_point_distance: float = Field(0.1, description="millimeters")
    DENOISE_OUTLIER_DISTANCE_MILLIMETERS: float = Field(10.0)
    DENOISE_OUTLIER_ANGLE_DEGREES: float = Field(5.0)
    DENOISE_STORAGE_SIZE: int = Field(10)
    DENOISE_FILTER_SIZE: int = Field(7)
    DENOISE_REQUIRED_STARTING_STREAK: int = Field(3)
    ARUCO_MARKER_DICTIONARY_ENUM: int = Field(cv2.aruco.DICT_4X4_100)
    ARUCO_POSE_ESTIMATOR_METHOD: int = Field(cv2.SOLVEPNP_ITERATIVE)
    # SOLVEPNP_ITERATIVE works okay but is susceptible to optical illusions (flipping)
    # SOLVEPNP_P3P appears to return nan's on rare occasion
    # SOLVEPNP_SQPNP appears to return nan's on rare occasion
    # SOLVEPNP_IPPE_SQUARE does not seem to work very well at all, translation is much smaller than expected
