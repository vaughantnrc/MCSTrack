from .corner_refinement import CornerRefinementMethod
import numpy
from pydantic import BaseModel, Field


class DetectionParameters(BaseModel):

    adaptive_thresh_win_size_min: int | None = Field(default=None)
    adaptive_thresh_win_size_max: int | None = Field(default=None)
    adaptive_thresh_win_size_step: int | None = Field(default=None)
    adaptive_thresh_constant: float | None = Field(default=None)

    min_marker_perimeter_rate: float | None = Field(default=None)
    max_marker_perimeter_rate: float | None = Field(default=None)
    polygonal_approx_accuracy_rate: float | None = Field(default=None)
    min_corner_distance_rate: float | None = Field(default=None)
    min_marker_distance_rate: float | None = Field(default=None)
    min_distance_to_border: int | None = Field(default=None)

    marker_border_bits: int | None = Field(default=None)
    min_otsu_std_dev: float | None = Field(default=None)
    perspective_remove_pixel_per_cell: int | None = Field(default=None)
    perspective_remove_ignored_margin_per_cell: float | None = Field(default=None)

    max_erroneous_bits_in_border_rate: float | None = Field(default=None)
    error_correction_rate: float | None = Field(default=None)
    detect_inverted_marker: bool | None = Field(default=None)

    corner_refinement_method: CornerRefinementMethod | None = Field(default=None)
    corner_refinement_win_size: int | None = Field(default=None)
    corner_refinement_max_iterations: int | None = Field(default=None)
    corner_refinement_min_accuracy: float | None = Field(default=None)

    april_tag_critical_rad: float | None = Field(default=None)
    april_tag_deglitch: int | None = Field(default=None)
    april_tag_max_line_fit_mse: float | None = Field(default=None)
    april_tag_max_nmaxima: int | None = Field(default=None)
    april_tag_min_cluster_pixels: int | None = Field(default=None)
    april_tag_min_white_black_diff: int | None = Field(default=None)
    april_tag_quad_decimate: float | None = Field(default=None)
    april_tag_quad_sigma: float | None = Field(default=None)

    use_aruco_3_detection: bool | None = Field(default=None)
    min_marker_length_ratio_original_img: float | None = Field(default=None)
    min_side_length_canonical_img: int | None = Field(default=None)

    @staticmethod
    def default_values():
        """
        Values from aruco_detector.hpp
        """
        return DetectionParameters(
            adaptive_thresh_win_size_min=3,
            adaptive_thresh_win_size_max=23,
            adaptive_thresh_win_size_step=10,
            adaptive_thresh_constant=7,
            min_marker_perimeter_rate=0.03,
            max_marker_perimeter_rate=4.0,
            polygonal_approx_accuracy_rate=0.03,
            min_corner_distance_rate=0.05,
            min_marker_distance_rate=0.05,
            min_distance_to_border=3,
            marker_border_bits=1,
            min_otsu_std_dev=5.0,
            perspective_remove_pixel_per_cell=4,
            perspective_remove_ignored_margin_per_cell=0.13,
            max_erroneous_bits_in_border_rate=0.35,
            error_correction_rate=0.6,
            detect_inverted_marker=False,
            corner_refinement_method="NONE",
            corner_refinement_win_size=5,
            corner_refinement_max_iterations=30,
            corner_refinement_min_accuracy=0.1,
            april_tag_critical_rad=numpy.pi * 10.0 / 180.0,
            april_tag_deglitch=0,
            april_tag_max_line_fit_mse=10.0,
            april_tag_max_nmaxima=10,
            april_tag_min_cluster_pixels=5,
            april_tag_min_white_black_diff=5,
            april_tag_quad_decimate=0.0,
            april_tag_quad_sigma=0.0,
            use_aruco_3_detection=False,
            min_marker_length_ratio_original_img=32,
            min_side_length_canonical_img=0.0)
