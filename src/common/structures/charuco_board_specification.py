from pydantic import BaseModel, Field
from typing import Any, Tuple
import cv2.aruco


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
        charuco_board = cv2.aruco.CharucoBoard_create(
            self.square_count_x,
            self.square_count_y,
            self.square_size_px,
            self.marker_size_px,
            self.aruco_dictionary())
        return charuco_board

    def get_marker_center_points(self) -> list[list[float]]:
        points = []
        for y in range(self.square_count_y):
            for x in range(self.square_count_x):
                if (x + y) % 2 == 1:  # Only add the points for the white squares
                    point_x = (x + 0.5) * self.square_size_px / self.px_per_mm
                    point_y = (self.square_count_y - y - 0.5) * self.square_size_px / self.px_per_mm
                    points.append([point_x, point_y, 0.0])
        return points

    def get_marker_ids(self) -> list[int]:
        num_markers = self.square_count_x * self.square_count_y // 2
        return list(range(num_markers))
