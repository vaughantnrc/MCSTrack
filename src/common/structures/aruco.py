import base64
import cv2
import cv2.aruco
import numpy
from pydantic import BaseModel, Field
from typing import Any, Final, Literal, Tuple


CornerRefinementMethod = Literal["NONE", "SUBPIX", "CONTOUR", "APRILTAG"]
CORNER_REFINEMENT_METHOD_NONE: Final[str] = 'NONE'
CORNER_REFINEMENT_METHOD_SUBPIX: Final[str] = 'SUBPIX'
CORNER_REFINEMENT_METHOD_CONTOUR: Final[str] = 'CONTOUR'
CORNER_REFINEMENT_METHOD_APRILTAG: Final[str] = 'APRILTAG'


CORNER_REFINEMENT_METHOD_DICTIONARY_TEXT_TO_INT: dict[CornerRefinementMethod, int] = {
    "NONE": cv2.aruco.CORNER_REFINE_NONE,
    "SUBPIX": cv2.aruco.CORNER_REFINE_SUBPIX,
    "CONTOUR": cv2.aruco.CORNER_REFINE_CONTOUR,
    "APRILTAG": cv2.aruco.CORNER_REFINE_APRILTAG}


CORNER_REFINEMENT_METHOD_DICTIONARY_INT_TO_TEXT: dict[int, CornerRefinementMethod] = {
    cv2.aruco.CORNER_REFINE_NONE: "NONE",
    cv2.aruco.CORNER_REFINE_SUBPIX: "SUBPIX",
    cv2.aruco.CORNER_REFINE_CONTOUR: "CONTOUR",
    cv2.aruco.CORNER_REFINE_APRILTAG: "APRILTAG"}


class CharucoBoardSpecification(BaseModel):
    dictionary_name: str = Field(default="DICT_4X4_100")
    square_count_x: int = Field(default=8)
    square_count_y: int = Field(default=10)
    square_size_px: int = Field(default=800)
    marker_size_px: int = Field(default=400)
    px_per_mm: float = Field(default=40)

    def aruco_dictionary(self) -> Any:  # type cv2.aruco.Dictionary
        if self.dictionary_name != "DICT_4X4_100":
            raise NotImplementedError("Only DICT_4X4_100 is currently implemented")
        aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        return aruco_dictionary

    def size_px(self) -> Tuple[float, float]:
        board_size_x_px = self.square_count_x * self.square_size_px
        board_size_y_px = self.square_count_y * self.square_size_px
        return board_size_x_px, board_size_y_px

    def size_mm(self) -> Tuple[float, float]:
        board_size_x_mm = self.square_count_x * self.square_size_px / self.px_per_mm
        board_size_y_mm = self.square_count_y * self.square_size_px / self.px_per_mm
        return board_size_x_mm, board_size_y_mm

    def create_board(self) -> Any:  # type cv2.aruco.CharucoBoard
        charuco_board = cv2.aruco.CharucoBoard(
            size=(self.square_count_x, self.square_count_y),
            squareLength=self.square_size_px,
            markerLength=self.marker_size_px,
            dictionary=self.aruco_dictionary())
        return charuco_board

    def get_marker_center_points(self) -> list[list[float]]:
        """
        Note that the coordinates assume (based on portrait orientation):
        origin: at bottom-left of board
        x-axis: goes right
        y-axis: goes up the page
        z-axis: comes out of the image and toward the viewer
        """
        points = []
        for y in range(self.square_count_y):
            for x in range(self.square_count_x):
                if (x + y) % 2 == 1:  # Only add the points for the white squares
                    point_x = (x + 0.5) * self.square_size_px / self.px_per_mm
                    point_y = (self.square_count_y - y - 0.5) * self.square_size_px / self.px_per_mm
                    points.append([point_x, point_y, 0.0])
        return points

    def get_marker_corner_points(self) -> list[list[float]]:
        """
        Note that the coordinates assume the same axes as get_marker_center_points,
        but the origin is in the center of the board, not the bottom-left corner.
        """
        points = []
        marker_size_mm: float = self.marker_size_px / self.px_per_mm
        square_size_mm: float = self.square_size_px / self.px_per_mm
        for y_sq in range(self.square_count_y):
            for x_sq in range(self.square_count_x):
                if (x_sq + y_sq) % 2 == 1:  # Only add the points for the white squares
                    x_sq_centered: float = x_sq - (self.square_count_x / 2.0)
                    y_sq_centered: float = y_sq - (self.square_count_y / 2.0)
                    for corner_index in range(0, 4):
                        x_mm: float = (x_sq_centered + 0.5) * square_size_mm
                        if corner_index == 0 or corner_index == 3:
                            x_mm -= (marker_size_mm / 2.0)
                        else:
                            x_mm += (marker_size_mm / 2.0)
                        y_mm: float = (-(y_sq_centered + 0.5)) * square_size_mm
                        if corner_index == 0 or corner_index == 1:
                            y_mm += (marker_size_mm / 2.0)
                        else:
                            y_mm -= (marker_size_mm / 2.0)
                        z_mm: float = 0.0
                        points.append([x_mm, y_mm, z_mm])
        return points

    def get_marker_ids(self) -> list[int]:
        num_markers = self.square_count_x * self.square_count_y // 2
        return list(range(num_markers))


class MarkerDefinition(BaseModel):
    # TODO: This is unused at the time of writing 2025-07-09, deletion should be assessed

    label: str = Field()
    representation_single_base64: str = Field()  # representation from a single rotation only

    def representation_all_base64(self):
        """
        OpenCV ArUco expects to receive all possible rotations of a marker. We generate these programmatically.
        """
        representation_single_bytes: bytes = base64.b64decode(self.representation_single_base64)
        representation_single_list: list[bool] = list(representation_single_bytes)
        representation_single_matrix: numpy.ndarray = numpy.asarray(
            a=representation_single_list,
            dtype=bool)
        marker_side_length_bits: int = int(numpy.sqrt(len(representation_single_list)))
        representation_single_matrix = numpy.reshape(
            a=representation_single_matrix,
            newshape=(marker_side_length_bits, marker_side_length_bits))
        representation_all_list: list[bool] = list(representation_single_matrix.flatten())
        for i in range(3):
            representation_single_matrix = numpy.rot90(representation_single_matrix)
            representation_all_list += list(representation_single_matrix.flatten())
        representation_all_bytes: bytes = bytes(representation_all_list)
        return base64.b64encode(representation_all_bytes)
