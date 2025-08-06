from .image_processing import Annotation
import cv2
import math  # Python's math module, not the one from this project!
import numpy
from pydantic import BaseModel, Field
from scipy.spatial.transform import Rotation
from typing import ClassVar, Final


_DEFAULT_EPSILON: Final[float] = 0.0001


class IntrinsicParameters(BaseModel):
    """
    Camera intrinsic parameters (focal length, optical center, distortion coefficients).
    See OpenCV's documentation: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    See Wikipedia article: https://en.wikipedia.org/wiki/Distortion_%28optics%29
    """

    focal_length_x_px: float = Field()
    focal_length_y_px: float = Field()
    optical_center_x_px: float = Field()
    optical_center_y_px: float = Field()

    radial_distortion_coefficients: list[float] = Field()  # k1, k2, k3 etc in OpenCV

    tangential_distortion_coefficients: list[float] = Field()  # p1, p2 in OpenCV

    def as_array(self) -> list[float]:
        return_value: list[float] = [
            self.focal_length_x_px,
            self.focal_length_y_px,
            self.optical_center_x_px,
            self.optical_center_y_px]
        return_value += self.get_distortion_coefficients()
        return return_value

    def get_matrix(self) -> list[list[float]]:
        """calibration matrix expected by OpenCV in some operations"""
        return \
            [[self.focal_length_x_px, 0.0, self.optical_center_x_px],
             [0.0, self.focal_length_y_px, self.optical_center_y_px],
             [0.0, 0.0, 1.0]]

    def get_distortion_coefficients(self) -> list[float]:
        """
        Distortion coefficients in array format expected by OpenCV in some operations.
        See https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
        calibrateCamera() documentation describes order of distortion coefficients that OpenCV works with
        """
        coefficients: list[float] = [
            self.radial_distortion_coefficients[0],
            self.radial_distortion_coefficients[1],
            self.tangential_distortion_coefficients[0],
            self.tangential_distortion_coefficients[1]]
        coefficients += self.radial_distortion_coefficients[2:]
        return coefficients

    @staticmethod
    def generate_zero_parameters(
        resolution_x_px: int,
        resolution_y_px: int,
        fov_x_degrees: float = 45.0,
        fov_y_degrees: float = 45.0
    ) -> "IntrinsicParameters":
        optical_center_x_px: int = int(round(resolution_x_px/2.0))
        fov_x_radians: float = fov_x_degrees * math.pi / 180.0
        focal_length_x_px = (resolution_x_px / 2.0) / math.tan(fov_x_radians / 2.0)
        optical_center_y_px: int = int(round(resolution_y_px/2.0))
        fov_y_radians: float = fov_y_degrees * math.pi / 180.0
        focal_length_y_px = (resolution_y_px / 2.0) / math.tan(fov_y_radians / 2.0)
        return IntrinsicParameters(
            focal_length_x_px=focal_length_x_px,
            focal_length_y_px=focal_length_y_px,
            optical_center_x_px=optical_center_x_px,
            optical_center_y_px=optical_center_y_px,
            radial_distortion_coefficients=[0.0, 0.0, 0.0],
            tangential_distortion_coefficients=[0.0, 0.0])


class IterativeClosestPointParameters(BaseModel):
    # ICP will stop after this many iterations
    termination_iteration_count: int = Field()

    # ICP will stop if distance *and* angle difference from one iteration to the next
    # is smaller than these
    termination_delta_translation: float = Field()
    termination_delta_rotation_radians: float = Field()

    # ICP will stop if overall point-to-point distance (between source and target)
    # mean *or* root-mean-square is less than specified
    termination_mean_point_distance: float = Field()
    termination_rms_point_distance: float = Field()  # root-mean-square


class Landmark(BaseModel):

    # These can denote that multiple landmarks are related if they share the same
    # "base label" (the part before the first and only occurrence of this character).
    RELATION_CHARACTER: ClassVar[str] = "$"

    """
    A distinct point in 3D space.
    Coordinates are in the unit of the user's choosing.
    """
    feature_label: str = Field()
    x: float = Field()
    y: float = Field()
    z: float = Field()

    def as_float_list(self) -> list[float]:
        return [self.x, self.y, self.z]

    def base_feature_label(self) -> str:
        """
        Part of the label before the RELATION_CHARACTER.
        """
        if self.RELATION_CHARACTER not in self.feature_label:
            return self.feature_label
        return self.feature_label[0:self.feature_label.index(self.RELATION_CHARACTER)]


class Matrix4x4(BaseModel):

    @staticmethod
    def _identity_values() -> list[float]:
        return \
            [1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0]
    values: list[float] = Field(default_factory=_identity_values)

    def as_numpy_array(self):
        a = self.values
        return numpy.asarray(
            [[a[0],  a[1],  a[2],  a[3]],
             [a[4],  a[5],  a[6],  a[7]],
             [a[8],  a[9],  a[10], a[11]],
             [a[12], a[13], a[14], a[15]]])

    def __getitem__(self, idx: tuple[int, int]) -> float:
        if isinstance(idx, tuple):
            return self.values[(idx[0]*4) + idx[1]]
        else:
            raise ValueError("Unexpected index. Expected tuple index [row,col].")

    def __mul__(self, other) -> 'Matrix4x4':
        if not isinstance(other, Matrix4x4):
            raise ValueError
        result_numpy_array = numpy.matmul(self.as_numpy_array(), other.as_numpy_array())
        return Matrix4x4(values=list(result_numpy_array.flatten()))

    def get_translation(self) -> list[float]:
        """
        Return a vector of [x,y,z] representing translation.
        """
        a = self.values
        return [a[3], a[7], a[11]]

    def inverse(self) -> 'Matrix4x4':
        inv_numpy_array = numpy.linalg.inv(self.as_numpy_array())
        return Matrix4x4.from_numpy_array(inv_numpy_array)

    @staticmethod
    def from_raw_values(
        v00, v01, v02, v03,
        v10, v11, v12, v13,
        v20, v21, v22, v23,
        v30, v31, v32, v33
    ) -> 'Matrix4x4':
        return Matrix4x4(values=[
            v00, v01, v02, v03,
            v10, v11, v12, v13,
            v20, v21, v22, v23,
            v30, v31, v32, v33])

    @staticmethod
    def from_list(
        value_list: list[float]
    ) -> 'Matrix4x4':
        if len(value_list) != 16:
            raise ValueError(f"Expected a list of 16 float. Got {str(value_list)}.")
        return Matrix4x4(values=list(value_list))

    @staticmethod
    def from_numpy_array(
        value_array: numpy.ndarray
    ) -> 'Matrix4x4':
        if len(value_array) != 4:
            raise ValueError(f"Expected input array to have 4 rows. Got {len(value_array)}.")
        for i in range(0, len(value_array)):
            if len(value_array[i]) != 4:
                raise ValueError(f"Expected input row {i} to have 4 col. Got {len(value_array[i])}.")
        return Matrix4x4(values=list(value_array.flatten()))


class Pose(BaseModel):
    target_id: str = Field()
    object_to_reference_matrix: Matrix4x4 = Field()
    solver_timestamp_utc_iso8601: str = Field()


class Ray:
    source_point: list[float]
    direction: list[float]

    def __init__(
        self,
        source_point: list[float],
        direction: list[float],
        epsilon: float = _DEFAULT_EPSILON
    ):
        direction_norm = numpy.linalg.norm(direction)
        if direction_norm < epsilon:
            raise ValueError("Direction cannot be zero.")
        self.source_point = source_point
        self.direction = direction


class Target(BaseModel):
    """
    A trackable object.
    """
    label: str
    landmarks: list[Landmark]

    def get_landmark_point(
        self,
        feature_label: str
    ) -> list[float]:
        for landmark in self.landmarks:
            if landmark.feature_label == feature_label:
                return landmark.as_float_list()
        raise ValueError


class MathUtils:
    """
    static class for reused math-related functions.
    """

    def __init__(self):
        raise RuntimeError("This class is not meant to be initialized.")

    @staticmethod
    def average_quaternion(
        quaternions: list[list[float]]
    ) -> list[float]:
        """
        Solution based on this link: https://stackoverflow.com/a/27410865
        based on Markley et al. "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30.4 (2007): 1193-1197.
        """
        quaternion_matrix = numpy.array(quaternions, dtype="float32").transpose()  # quaternions into columns
        quaternion_matrix /= len(quaternions)
        eigenvalues, eigenvectors = numpy.linalg.eig(numpy.matmul(quaternion_matrix, quaternion_matrix.transpose()))
        maximum_eigenvalue_index = numpy.argmax(eigenvalues)
        quaternion = eigenvectors[:, maximum_eigenvalue_index]
        if quaternion[3] < 0:
            quaternion *= -1
        return quaternion.tolist()

    @staticmethod
    def average_vector(
        translations: list[list[float]]
    ) -> list[float]:
        """
        This is a very simple function for averaging translations
        when it is not desired to use numpy (for whatever reason)
        """
        sum_translations: list[float] = [0.0, 0.0, 0.0]
        for translation in translations:
            for i in range(0, 3):
                sum_translations[i] += translation[i]
        translation_count = len(translations)
        return [
            sum_translations[0] / translation_count,
            sum_translations[1] / translation_count,
            sum_translations[2] / translation_count]

    class RayIntersection2Output:
        parallel: bool  # special case, mark it as such
        closest_point_1: numpy.ndarray
        closest_point_2: numpy.ndarray

        def __init__(
            self,
            parallel: bool,
            closest_point_1: numpy.ndarray,
            closest_point_2: numpy.ndarray
        ):
            self.parallel = parallel
            self.closest_point_1 = closest_point_1
            self.closest_point_2 = closest_point_2

        def centroid(self) -> numpy.ndarray:
            return (self.closest_point_1 + self.closest_point_2) / 2

        def distance(self) -> float:
            return float(numpy.linalg.norm(self.closest_point_2 - self.closest_point_1))

    @staticmethod
    def closest_intersection_between_two_lines(
        ray_1: Ray,
        ray_2: Ray,
        epsilon: float = _DEFAULT_EPSILON
    ) -> RayIntersection2Output:  # Returns data on intersection
        ray_1_direction_normalized = numpy.asarray(ray_1.direction) / numpy.linalg.norm(ray_1.direction)
        ray_2_direction_normalized = numpy.asarray(ray_2.direction) / numpy.linalg.norm(ray_2.direction)

        # ray 3 will be perpendicular to both rays 1 and 2,
        # and will intersect with both rays at the nearest point(s)

        ray_3_direction = numpy.cross(ray_2_direction_normalized, ray_1_direction_normalized)
        ray_3_direction_norm = numpy.linalg.norm(ray_3_direction)
        if ray_3_direction_norm < epsilon:
            return MathUtils.RayIntersection2Output(
                parallel=True,
                closest_point_1=numpy.asarray(ray_1.source_point),
                closest_point_2=numpy.asarray(ray_2.source_point))

        # system of equations Ax = b
        b = numpy.subtract(ray_2.source_point, ray_1.source_point)
        a = numpy.asarray(
            [ray_1_direction_normalized, -ray_2_direction_normalized, ray_3_direction], dtype="float32").transpose()
        x = numpy.linalg.solve(a, b)

        param_ray_1 = float(x[0])
        intersection_point_1 = ray_1.source_point + param_ray_1 * ray_1_direction_normalized

        param_ray_2 = float(x[1])
        intersection_point_2 = ray_2.source_point + param_ray_2 * ray_2_direction_normalized

        return MathUtils.RayIntersection2Output(
            parallel=False,
            closest_point_1=intersection_point_1,
            closest_point_2=intersection_point_2)

    class RayIntersectionNOutput:
        centroids: numpy.ndarray

        # How many rays were used.
        # Note that centroids might not use all possible intersections (e.g. parallel rays)
        ray_count: int

        def __init__(
            self,
            centroids: numpy.ndarray,
            ray_count: int
        ):
            self.centroids = centroids
            self.ray_count = ray_count

        def centroid(self) -> numpy.ndarray:
            sum_centroids = numpy.asarray([0, 0, 0], dtype="float32")
            for centroid in self.centroids:
                sum_centroids += centroid
            return sum_centroids / self.centroids.shape[0]

        def intersection_count(self) -> int:
            return int((self.ray_count * (self.ray_count - 1)) / 2)

    @staticmethod
    def closest_intersection_between_n_lines(
        rays: list[Ray],
        maximum_distance: float
    ) -> RayIntersectionNOutput:
        ray_count = len(rays)
        intersections: list[MathUtils.RayIntersection2Output] = list()
        for ray_1_index in range(0, ray_count):
            for ray_2_index in range(ray_1_index + 1, ray_count):
                intersections.append(MathUtils.closest_intersection_between_two_lines(
                    ray_1=rays[ray_1_index],
                    ray_2=rays[ray_2_index]))
        centroids: list[numpy.ndarray] = list()
        for intersection in intersections:
            if intersection.parallel:
                continue
            if intersection.distance() > maximum_distance:
                continue
            centroids.append(intersection.centroid())
        return MathUtils.RayIntersectionNOutput(
            centroids=numpy.asarray(centroids, dtype="float32"),
            ray_count=ray_count)

    @staticmethod
    def closest_point_on_ray(
        ray_source: list[float],
        ray_direction: list[float],
        query_point: list[float],
        forward_only: bool
    ):
        """
        Find the closest point on a ray in 3D.
        """
        # Let ray_point be the closest point between query_point and the ray.
        # (ray_point - query_point) will be perpendicular to ray_direction.
        # Let ray_distance be the distance along the ray where the closest point is.
        # So we have two equations:
        #     (1)    (ray_point - query_point) * ray_direction = 0
        #     (2)    ray_point = ray_source + ray_distance * ray_direction
        # If we substitute eq (2) into (1) and solve for ray_distance, we get:
        ray_distance: float = (
                (query_point[0] * ray_direction[0] + query_point[1] * ray_direction[1] + query_point[2] * ray_direction[
                    2]
                 - ray_source[0] * ray_direction[0] - ray_source[1] * ray_direction[1] - ray_source[2] * ray_direction[
                     2])
                /
                ((ray_direction[0] ** 2) + (ray_direction[1] ** 2) + (ray_direction[2] ** 2)))

        if ray_distance < 0 and forward_only:
            return ray_source  # point is behind the source, so the closest point is just the source

        ray_point = [0.0] * 3  # temporary values
        for i in range(0, 3):
            ray_point[i] = ray_source[i] + ray_distance * ray_direction[i]
        return ray_point

    @staticmethod
    def convert_detector_points_to_vectors(
        points: list[list[float]],  # [point_index][x/y/z]
        detector_intrinsics: IntrinsicParameters,
        detector_to_reference_matrix: Matrix4x4
    ) -> list[list[float]]:
        """
        Given a detector's matrix transform and its intrinsic properties,
        convert pixel coordinates to ray directions (with origin at the detector).
        """
        distorted_points: numpy.ndarray = numpy.asarray(points)
        distorted_points = numpy.reshape(
            a=distorted_points,
            newshape=(1, len(points), 2))
        undistorted_points: numpy.ndarray = cv2.undistortPoints(
            src=distorted_points,
            cameraMatrix=numpy.asarray(detector_intrinsics.get_matrix()),
            distCoeffs=numpy.asarray(detector_intrinsics.get_distortion_coefficients()))
        rays: list[list[float]] = list()
        origin_point_detector = [0, 0, 0, 1]  # origin
        detector_to_reference: numpy.ndarray = detector_to_reference_matrix.as_numpy_array()
        for undistorted_point in undistorted_points:
            target_point_image = [undistorted_point[0, 0], undistorted_point[0, 1], 1, 1]  # reverse perspective
            target_point_detector = MathUtils.image_to_opengl_vector(target_point_image)
            ray_direction_detector = numpy.subtract(target_point_detector, origin_point_detector)
            ray_direction_detector = ray_direction_detector / numpy.linalg.norm(ray_direction_detector)  # normalize
            ray_direction_reference = numpy.matmul(detector_to_reference, ray_direction_detector)
            rays.append(list(ray_direction_reference[0:3]))
        return rays

    @staticmethod
    def convex_quadrilateral_area(
        points: list[list[float]],  # 2D points in clockwise order
        epsilon: float = _DEFAULT_EPSILON
    ) -> float:
        """
        Compute the area of a quadrilateral, given 2D points in clockwise order.
        """

        # General approach:
        # Given points a, b, c, and d shown below,
        # and calculating points e and f shown below,
        # add areas defined by right triangles bea, ceb, dfc, and afd
        # b..................c
        # . ..              ...
        #  .  ...        ...  .
        #  .     ..   .f.      .
        #   .      .e.  ...    .
        #   .   ...        ..   .
        #    ...             ... .
        #    a...................d

        point_a = numpy.array(points[0], dtype="float32")
        point_b = numpy.array(points[1], dtype="float32")
        point_c = numpy.array(points[2], dtype="float32")
        point_d = numpy.array(points[3], dtype="float32")

        vector_ac = point_c - point_a
        vector_ac_norm = numpy.linalg.norm(vector_ac)
        vector_bd = point_d - point_b
        vector_bd_norm = numpy.linalg.norm(vector_bd)
        if vector_ac_norm <= epsilon or vector_bd_norm <= epsilon:
            return 0.0
        width_vector = vector_ac / numpy.linalg.norm(vector_ac)
        height_vector = numpy.array([width_vector[1], -width_vector[0]], dtype="float32")  # rotated 90 degrees

        sum_of_areas: float = 0.0
        point_pairs: list[tuple[numpy.ndarray, numpy.ndarray]] = [
            (point_a, point_b),
            (point_b, point_c),
            (point_c, point_d),
            (point_d, point_a)]
        for point_pair in point_pairs:
            line_vector = point_pair[1] - point_pair[0]
            width = numpy.dot(line_vector, width_vector)
            height = numpy.dot(line_vector, height_vector)
            sum_of_areas += numpy.abs(width * height / 2.0)

        return sum_of_areas

    @staticmethod
    def estimate_matrix_transform_to_detector(
        annotations: list[Annotation],
        target: Target,
        detector_intrinsics: IntrinsicParameters
    ) -> Matrix4x4:
        target_points: list[list[float]] = list()    # ordered points [point_index][x/y/z]
        detector_points: list[list[float]] = list()  # ordered points [point_index][x/y]
        annotations_dict: dict[str, Annotation] = {annotation.feature_label: annotation for annotation in annotations}
        for landmark in target.landmarks:
            if landmark.feature_label in annotations_dict.keys():
                annotation = annotations_dict[landmark.feature_label]
                target_points.append([landmark.x, landmark.y, landmark.z])
                detector_points.append([annotation.x_px, annotation.y_px])
        rotation_vector: numpy.ndarray
        translation_vector: numpy.ndarray
        _, rotation_vector, translation_vector = cv2.solvePnP(
            objectPoints=numpy.asarray(target_points),
            imagePoints=numpy.asarray(detector_points),
            cameraMatrix=numpy.asarray(detector_intrinsics.get_matrix()),
            distCoeffs=numpy.asarray(detector_intrinsics.get_distortion_coefficients()))
        rotation_vector = rotation_vector.flatten()
        translation_vector = translation_vector.flatten()
        object_to_camera_matrix = numpy.identity(4, dtype="float32")
        object_to_camera_matrix[0:3, 0:3] = Rotation.from_rotvec(rotation_vector).as_matrix()
        object_to_camera_matrix[0:3, 3] = translation_vector[0:3]
        object_to_detector_matrix = MathUtils.image_to_opengl_transformation_matrix(object_to_camera_matrix)
        return Matrix4x4.from_numpy_array(object_to_detector_matrix)

    class IterativeClosestPointOutput:
        source_to_target_matrix: Matrix4x4
        iteration_count: int
        mean_point_distance: float
        rms_point_distance: float

        def __init__(
            self,
            source_to_target_matrix: Matrix4x4,
            iteration_count: int,
            mean_point_distance: float,
            rms_point_distance: float
        ):
            self.source_to_target_matrix = source_to_target_matrix
            self.iteration_count = iteration_count
            self.mean_point_distance = mean_point_distance
            self.rms_point_distance = rms_point_distance

    @staticmethod
    def iterative_closest_point_for_points_and_rays(
        source_known_points: list[list[float]],
        target_known_points: list[list[float]],
        source_ray_points: list[list[float]],
        target_rays: list[Ray],
        parameters: IterativeClosestPointParameters = None,
        initial_transformation_matrix: numpy.ndarray = None
    ) -> IterativeClosestPointOutput:
        """
        Algorithm is based on ICP: Besl and McKay. Method for registration of 3-D shapes. 1992.
        This is customized, adapted to the problem of registering a set of points to
        a set of points and rays where the correspondence is known.
        :param source_known_points: points with known corresponding positions in both source and target coordinate frames
        :param target_known_points: points with known corresponding positions in both source and target coordinate frames
        :param source_ray_points: points with known position in the source coordinate frame, but NOT in target
        :param target_rays: rays along which the remaining target points lie (1:1 correspondence with source_ray_points)
        :param parameters:
        :param initial_transformation_matrix:
        """

        def _transform_points(
            original_points: list[list[float]],
            transformation: numpy.ndarray
        ):
            transformed_points: list[list[float]] = list()
            for point in original_points:
                transformed_point = list(numpy.matmul(
                    transformation,
                    numpy.array([point[0], point[1], point[2], 1])))
                transformed_points.append([transformed_point[0], transformed_point[1], transformed_point[2]])
            return transformed_points

        if len(source_known_points) != len(target_known_points):
            raise ValueError(
                "source_known_points and target_known_points must be of equal length (1:1 correspondence).")

        if len(source_known_points) != len(target_known_points):
            raise ValueError("source_ray_points and target_rays must be of equal length (1:1 correspondence).")

        # Initial transformation
        source_to_transformed_matrix: numpy.ndarray
        if initial_transformation_matrix is not None:
            source_to_transformed_matrix = numpy.array(initial_transformation_matrix, dtype="float32")
        else:
            source_to_transformed_matrix = numpy.identity(4, dtype="float32")

        if parameters is None:
            parameters = IterativeClosestPointParameters(
                termination_iteration_count=50,
                termination_delta_translation=0.1,
                termination_delta_rotation_radians=0.001,
                termination_mean_point_distance=0.1,
                termination_rms_point_distance=0.1)

        transformed_known_points: list[list[float]] = _transform_points(
            original_points=source_known_points,
            transformation=source_to_transformed_matrix)
        transformed_ray_points: list[list[float]] = _transform_points(
            original_points=source_ray_points,
            transformation=source_to_transformed_matrix)

        iteration_count: int = 0
        mean_point_distance: float
        rms_point_distance: float
        while True:
            target_ray_points: list[list[float]] = list()
            for i, transformed_ray_point in enumerate(transformed_ray_points):
                target_ray_points.append(MathUtils.closest_point_on_ray(
                    ray_source=target_rays[i].source_point,
                    ray_direction=target_rays[i].direction,
                    query_point=transformed_ray_point,
                    forward_only=True))

            transformed_all_points = transformed_known_points + transformed_ray_points
            target_points = target_known_points + target_ray_points
            transformed_to_target_matrix = MathUtils.register_corresponding_points(
                point_set_from=transformed_all_points,
                point_set_to=target_points,
                collinearity_do_check=False)

            # update transformation & transformed points
            source_to_transformed_matrix = numpy.matmul(transformed_to_target_matrix, source_to_transformed_matrix)
            transformed_known_points: list[list[float]] = _transform_points(
                original_points=source_known_points,
                transformation=source_to_transformed_matrix)
            transformed_ray_points: list[list[float]] = _transform_points(
                original_points=source_ray_points,
                transformation=source_to_transformed_matrix)

            iteration_count += 1

            transformed_all_points = transformed_known_points + transformed_ray_points
            point_offsets = numpy.subtract(target_points, transformed_all_points).tolist()
            sum_point_distances = 0.0
            sum_square_point_distances = 0.0
            for delta_point_offset in point_offsets:
                delta_point_distance: float = numpy.linalg.norm(delta_point_offset)
                sum_point_distances += delta_point_distance
                sum_square_point_distances += numpy.square(delta_point_distance)
            mean_point_distance = sum_point_distances / len(point_offsets)
            rms_point_distance = numpy.sqrt(sum_square_point_distances / len(point_offsets))

            # Check if termination criteria are met
            # Note that transformed_to_target_matrix describes the change since last iteration, so we often operate on it
            delta_translation = numpy.linalg.norm(transformed_to_target_matrix[0:3, 3])
            delta_rotation_radians = \
                numpy.linalg.norm(Rotation.from_matrix(transformed_to_target_matrix[0:3, 0:3]).as_rotvec())
            if delta_translation < parameters.termination_delta_translation and \
                    delta_rotation_radians < parameters.termination_delta_rotation_radians:
                break
            if mean_point_distance < parameters.termination_mean_point_distance:
                break
            if rms_point_distance < parameters.termination_rms_point_distance:
                break
            if iteration_count >= parameters.termination_iteration_count:
                break

        return MathUtils.IterativeClosestPointOutput(
            source_to_target_matrix=Matrix4x4.from_numpy_array(source_to_transformed_matrix),
            iteration_count=iteration_count,
            mean_point_distance=mean_point_distance,
            rms_point_distance=rms_point_distance)

    @staticmethod
    def image_to_opengl_transformation_matrix(
        transformation_matrix_image: numpy.ndarray
    ) -> numpy.ndarray:
        transformation_matrix_detector = numpy.array(transformation_matrix_image)
        transformation_matrix_detector[1:3, :] = -transformation_matrix_detector[1:3, :]
        return transformation_matrix_detector
        # Equivalent to:
        # transformation_matrix_180 = numpy.identity(4, dtype="float")
        # transformation_matrix_180[1, 1] *= -1
        # transformation_matrix_180[2, 2] *= -1
        # return numpy.matmul(transformation_matrix_180, transformation_matrix_image)

    @staticmethod
    def image_to_opengl_vector(
        vector_image: numpy.ndarray | list[float]
    ) -> numpy.ndarray:
        vector_detector = numpy.array(vector_image)
        vector_detector[1:3] = -vector_detector[1:3]
        return vector_detector
        # Equivalent to:
        # transformation_matrix_180 = numpy.identity(4, dtype="float")
        # transformation_matrix_180[1, 1] *= -1
        # transformation_matrix_180[2, 2] *= -1
        # return numpy.matmul(transformation_matrix_180, vector_image)

    @staticmethod
    def register_corresponding_points(
        point_set_from: list[list[float]],
        point_set_to: list[list[float]],
        collinearity_do_check: bool = True,
        collinearity_zero_threshold: float = 0.0001,
        use_oomori_mirror_fix: bool = True
    ) -> numpy.array:  # 4x4 transformation matrix, indexed by [row,col]
        """
        Solution based on: Arun et al. Least square fitting of two 3D point sets (1987)
        https://stackoverflow.com/questions/66923224/rigid-registration-of-two-point-clouds-with-known-correspondence
        Use mirroring solution proposed by Oomori et al.
        Oomori et al. Point cloud matching using singular value decomposition. (2016)
        :param point_set_from:
        :param point_set_to:
        :param collinearity_do_check: Do a (naive) collinearity check. May be computationally expensive.
        :param collinearity_zero_threshold: Threshold considered zero for cross product and norm comparisons
        :param use_oomori_mirror_fix: Use the mirroring solution proposed in Oomori et al 2016.
        """
        if len(point_set_from) != len(point_set_to):
            raise ValueError("Input point sets must be of identical length.")
        if len(point_set_from) < 3:
            raise ValueError("Input point sets must be of length 3 or higher.")
        if collinearity_do_check:
            for point_set in (point_set_from, point_set_to):
                collinear = True  # assume true until shown otherwise
                p1: numpy.ndarray = numpy.asarray(point_set[0])
                vec1: numpy.ndarray
                i: int = 1
                for i in range(i, len(point_set)):
                    p2: numpy.ndarray = numpy.asarray(point_set[1])
                    vec1 = p2 - p1
                    vec1_length: float = float(numpy.linalg.norm(vec1))
                    if vec1_length > collinearity_zero_threshold:
                        break  # points are distinct, move to next phase
                for i in range(i, len(point_set)):
                    p3: numpy.ndarray = numpy.asarray(point_set[2])
                    vec2: numpy.ndarray = p3 - p1
                    # noinspection PyUnboundLocalVariable
                    cross_product_norm: float = float(numpy.linalg.norm(numpy.cross(vec1, vec2)))
                    if cross_product_norm > collinearity_zero_threshold:
                        collinear = False
                        break
                if collinear:
                    raise ValueError("Input points appear to be collinear - please check the input.")

        # for consistency, points are in rows
        point_count = len(point_set_from)
        sums_from = numpy.array([0, 0, 0], dtype="float32")
        sums_to = numpy.array([0, 0, 0], dtype="float32")
        for point_index in range(0, point_count):
            sums_from += numpy.array(point_set_from[point_index])
            sums_to += numpy.array(point_set_to[point_index])
        centroid_from = (sums_from / point_count)
        centroid_to = (sums_to / point_count)
        points_from = numpy.array(point_set_from)
        points_to = numpy.array(point_set_to)
        centered_points_from = points_from - numpy.hstack(centroid_from)
        centered_points_to = points_to - numpy.hstack(centroid_to)
        covariance = numpy.matmul(centered_points_from.T, centered_points_to)
        u, _, vh = numpy.linalg.svd(covariance)
        s = numpy.identity(3, dtype="float32")  # s will be the Oomori mirror fix
        if use_oomori_mirror_fix:
            s[2, 2] = numpy.linalg.det(numpy.matmul(u, vh))
        rotation = numpy.matmul(u, numpy.matmul(s, vh)).transpose()
        translation = centroid_to - numpy.matmul(rotation, centroid_from)
        matrix = numpy.identity(4, dtype="float32")
        matrix[0:3, 0:3] = rotation
        matrix[0:3, 3] = translation[0:3].reshape(3)
        return matrix

    @staticmethod
    def square_marker_corner_points(
        marker_size: float
    ) -> list[list[float]]:  #[corner_index][dimension_index], 3D
        half_width = marker_size / 2.0
        corner_points = [
            [-half_width, half_width, 0., 1.],  # Top-left
            [half_width, half_width, 0., 1.],  # Top-right
            [half_width, -half_width, 0., 1.],  # Bottom-right
            [-half_width, -half_width, 0., 1.]]  # Bottom-left
        return corner_points
