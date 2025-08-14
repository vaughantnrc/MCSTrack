from .common_aruco_opencv import ArucoOpenCVCommon
from src.common import \
    Annotation, \
    ExtrinsicCalibration, \
    ExtrinsicCalibrationDetectorResult, \
    ExtrinsicCalibrator, \
    FeatureRay, \
    IntrinsicParameters, \
    Landmark, \
    MathUtils, \
    Matrix4x4, \
    Target
import cv2
import cv2.aruco
import datetime
import numpy
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation


class _ImageData(BaseModel):
    """
    Helper structure - data stored for each image
    """
    detector_label: str = Field()
    annotations: list[Annotation] = Field()
    rays: list[FeatureRay] = Field(default_factory=list)

    def annotations_as_points(self) -> list[list[float]]:
        return [
            [annotation.x_px, annotation.y_px]
            for annotation in self.annotations]


class _FeatureData(BaseModel):
    """
    Helper structure - data stored for each feature
    """
    feature_label: str = Field()
    position: Landmark | None = Field(default=None)  # None means it has not (or cannot) be calculated


class _TimestampData(BaseModel):
    """
    Helper structure - data stored for each unique timestamp
    """
    timestamp_utc_iso8601: str = Field()
    images: list[_ImageData] = Field(default_factory=list)
    features: list[_FeatureData] = Field(default_factory=list)


class _DetectorData(BaseModel):
    """
    Helper structure - data stored for each detector
    """
    detector_label: str = Field()
    intrinsic_parameters: IntrinsicParameters = Field()
    initial_to_reference: Matrix4x4 | None = Field(default=None)  # Stored primarily for analyses
    refined_to_reference: Matrix4x4 | None = Field(default=None)


class _CalibrationData(BaseModel):
    """
    Helper structure - container for all things related to calibration
    """
    timestamps: list[_TimestampData] = Field(default_factory=list)
    detectors: list[_DetectorData] = Field(default_factory=list)

    def get_detector_container(
        self,
        detector_label: str
    ) -> _DetectorData:
        for detector in self.detectors:
            if detector.detector_label == detector_label:
                return detector
        raise IndexError()

    def get_feature_container(
        self,
        timestamp_utc_iso8601: str,
        feature_label: str
    ) -> _FeatureData:
        for timestamp in self.timestamps:
            if timestamp.timestamp_utc_iso8601 == timestamp_utc_iso8601:
                for feature in timestamp.features:
                    if feature.feature_label == feature_label:
                        return feature
                break
        raise IndexError()

    def get_image_container(
        self,
        timestamp_utc_iso8601: str,
        detector_label: str
    ) -> _ImageData:
        for timestamp in self.timestamps:
            if timestamp.timestamp_utc_iso8601 == timestamp_utc_iso8601:
                for image in timestamp.images:
                    if image.detector_label == detector_label:
                        return image
                break
        raise IndexError()

    def get_timestamp_container(
        self,
        timestamp_utc_iso8601: str
    ) -> _TimestampData:
        for timestamp in self.timestamps:
            if timestamp.timestamp_utc_iso8601 == timestamp_utc_iso8601:
                return timestamp
        raise IndexError()


class _Configuration(ExtrinsicCalibrator.Configuration):
    termination_iteration_count: int = Field(default=500)
    termination_rotation_change_degrees: int = Field(default=0.05)
    termination_translation_change: int = Field(default=0.5)
    ray_intersection_maximum_distance: float = Field(default=50.0)


class CharucoOpenCVExtrinsicCalibrator(ExtrinsicCalibrator):

    Configuration: type[ExtrinsicCalibrator.Configuration] = _Configuration
    configuration: _Configuration

    def __init__(self, configuration: Configuration | dict):
        if isinstance(configuration, dict):
            configuration = _Configuration(**configuration)
        self.configuration = configuration
        super().__init__(configuration)

    @staticmethod
    def _annotate_image(
        aruco_detector_parameters: cv2.aruco.DetectorParameters,
        aruco_dictionary: cv2.aruco.Dictionary,
        image_metadata: ExtrinsicCalibrator.ImageMetadata
    ) -> list[Annotation]:
        image_rgb = cv2.imread(image_metadata.filepath)
        image_greyscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        annotations: list[Annotation]
        annotations, _ = ArucoOpenCVCommon.annotations_from_greyscale_image(
            aruco_detector_parameters=aruco_detector_parameters,
            aruco_dictionary=aruco_dictionary,
            image_greyscale=image_greyscale)
        return annotations

    def _calculate_implementation(
        self,
        detector_intrinsics_by_label: dict[str, IntrinsicParameters],
        image_metadata_list: list[ExtrinsicCalibrator.ImageMetadata]
    ) -> tuple[ExtrinsicCalibration, list[ExtrinsicCalibrator.ImageMetadata]]:
        charuco_spec: ArucoOpenCVCommon.CharucoBoard = ArucoOpenCVCommon.CharucoBoard()
        aruco_detector_parameters: cv2.aruco.DetectorParameters = \
            ArucoOpenCVCommon.standard_aruco_detection_parameters()
        aruco_dictionary: cv2.aruco.Dictionary = ArucoOpenCVCommon.standard_aruco_dictionary()
        charuco_target: Target = charuco_spec.as_target(target_label="board")

        # Populate _CalibrationData structure, including detection of annotations
        data: _CalibrationData = _CalibrationData()
        for metadata in image_metadata_list:
            annotations: list[Annotation] = self._annotate_image(
                aruco_detector_parameters=aruco_detector_parameters,
                aruco_dictionary=aruco_dictionary,
                image_metadata=metadata)
            image_data: _ImageData = _ImageData(
                detector_label=metadata.detector_label,
                annotations=annotations)
            timestamp_data: _TimestampData
            try:
                timestamp_data = data.get_timestamp_container(timestamp_utc_iso8601=metadata.timestamp_utc_iso8601)
            except IndexError:
                timestamp_data = _TimestampData(timestamp_utc_iso8601=metadata.timestamp_utc_iso8601)
                data.timestamps.append(timestamp_data)
            timestamp_data.images.append(image_data)
            try:
                data.get_detector_container(detector_label=image_data.detector_label)
            except IndexError:
                detector: _DetectorData = _DetectorData(
                    detector_label=metadata.detector_label,
                    intrinsic_parameters=detector_intrinsics_by_label[metadata.detector_label])
                data.detectors.append(detector)
            for annotation in annotations:
                try:
                    data.get_feature_container(
                        timestamp_utc_iso8601=metadata.timestamp_utc_iso8601,
                        feature_label=annotation.feature_label)
                except IndexError:
                    feature_data: _FeatureData = _FeatureData(feature_label=annotation.feature_label)
                    timestamp_data = data.get_timestamp_container(timestamp_utc_iso8601=metadata.timestamp_utc_iso8601)
                    timestamp_data.features.append(feature_data)

        # Initial estimate of the pose of each detector relative to first frame
        first_timestamp: _TimestampData = data.get_timestamp_container(
            timestamp_utc_iso8601=min([metadata.timestamp_utc_iso8601 for metadata in image_metadata_list]))
        for metadata in image_metadata_list:
            if metadata.timestamp_utc_iso8601 == first_timestamp.timestamp_utc_iso8601:
                image_data: _ImageData = data.get_image_container(
                    timestamp_utc_iso8601=metadata.timestamp_utc_iso8601,
                    detector_label=metadata.detector_label)
                intrinsic_parameters: IntrinsicParameters = detector_intrinsics_by_label[metadata.detector_label]
                reference_to_initial: Matrix4x4 = MathUtils.estimate_matrix_transform_to_detector(
                    annotations=image_data.annotations,
                    landmarks=charuco_target.landmarks,
                    detector_intrinsics=intrinsic_parameters)
                initial_to_reference: Matrix4x4 = reference_to_initial.inverse()
                detector: _DetectorData = data.get_detector_container(detector_label=image_data.detector_label)
                detector.initial_to_reference = initial_to_reference
                detector.refined_to_reference = initial_to_reference

        for i in range(0, self.configuration.termination_iteration_count):
            # Update each ray based on the current pose
            for timestamp_data in data.timestamps:
                for image_data in timestamp_data.images:
                    if len(image_data.annotations) == 0:
                        continue
                    detector_data: _DetectorData = data.get_detector_container(detector_label=image_data.detector_label)
                    feature_labels: list[str] = [annotation.feature_label for annotation in image_data.annotations]
                    ray_directions: list[list[float]] = MathUtils.convert_detector_points_to_vectors(
                        points=image_data.annotations_as_points(),
                        detector_intrinsics=detector_data.intrinsic_parameters,
                        detector_to_reference_matrix=detector_data.refined_to_reference)
                    source_point: list[float] = detector_data.refined_to_reference.get_translation()
                    annotation_count = len(image_data.annotations)
                    image_data.rays = [
                        FeatureRay(
                            feature_label=feature_labels[annotation_index],
                            source_point=source_point,
                            direction=ray_directions[annotation_index])
                        for annotation_index in range(0, annotation_count)]
            # For each (timestamp, feature_label), intersect rays to get 3D positions in a common coordinate system
            for timestamp_data in data.timestamps:
                for feature_data in timestamp_data.features:
                    ray_list: list[FeatureRay] = list()
                    feature_label = feature_data.feature_label
                    for image_data in timestamp_data.images:
                        for ray in image_data.rays:
                            if ray.feature_label == feature_label:
                                ray_list.append(ray)
                    ray_intersection: MathUtils.RayIntersectionNOutput = MathUtils.closest_intersection_between_n_lines(
                        rays=ray_list,
                        maximum_distance=self.configuration.ray_intersection_maximum_distance)
                    if ray_intersection.intersection_count() > 0:
                        position: numpy.ndarray = ray_intersection.centroid()
                        feature_data.position = Landmark(
                            feature_label=feature_label,
                            x=float(position[0]),
                            y=float(position[1]),
                            z=float(position[2]))
                    else:
                        feature_data.position = None
            # Use the newly-calculated 3D points together with the annotations to update the pose (PnP)
            converged: bool = True  # until shown otherwise
            for detector_data in data.detectors:
                landmarks: list[Landmark] = list()
                annotations: list[Annotation] = list()
                for timestamp_index, timestamp_data in enumerate(data.timestamps):
                    for feature_data in timestamp_data.features:
                        timestamped_feature_label: str = \
                            f"{feature_data.feature_label}{Annotation.RELATION_CHARACTER}{timestamp_index}"
                        for image_data in timestamp_data.images:
                            if image_data.detector_label != detector_data.detector_label:
                                continue
                            for annotation in image_data.annotations:
                                if annotation.feature_label == feature_data.feature_label and \
                                   feature_data.position is not None:
                                    landmarks.append(Landmark(
                                        feature_label=timestamped_feature_label,
                                        x=feature_data.position.x,
                                        y=feature_data.position.y,
                                        z=feature_data.position.z))
                                    annotations.append(Annotation(
                                        feature_label=timestamped_feature_label,
                                        x_px=annotation.x_px,
                                        y_px=annotation.y_px))
                reference_to_refined: Matrix4x4 = MathUtils.estimate_matrix_transform_to_detector(
                    annotations=annotations,
                    landmarks=landmarks,
                    detector_intrinsics=detector_data.intrinsic_parameters)
                refined_to_reference: Matrix4x4 = reference_to_refined.inverse()
                translation_change: float = numpy.linalg.norm(
                    numpy.asarray(refined_to_reference.get_translation()) -
                    numpy.asarray(detector_data.refined_to_reference.get_translation()))
                old_to_refined: numpy.ndarray = numpy.matmul(
                    reference_to_refined.as_numpy_array(),
                    detector_data.refined_to_reference.as_numpy_array())
                # noinspection PyArgumentList
                rotation_change_degrees: float = \
                    numpy.linalg.norm(Rotation.from_matrix(old_to_refined[0:3, 0:3]).as_rotvec(degrees=True))
                detector_data.refined_to_reference = refined_to_reference
                if rotation_change_degrees > self.configuration.termination_rotation_change_degrees or \
                   translation_change > self.configuration.termination_translation_change:
                    converged = False
            if converged:
                break

        extrinsic_calibration: ExtrinsicCalibration = ExtrinsicCalibration(
            timestamp_utc=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            calibrated_values=[
                ExtrinsicCalibrationDetectorResult(
                    detector_label=detector_data.detector_label,
                    detector_to_reference=detector_data.refined_to_reference)
                for detector_data in data.detectors],
            supplemental_data=data.model_dump())
        return extrinsic_calibration, image_metadata_list
