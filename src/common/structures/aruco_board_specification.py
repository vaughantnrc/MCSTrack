from typing import Any, Dict, List
from pydantic import BaseModel, validator
from .vec3 import Vec3
import json, hjson

class BoardMarker(BaseModel):
    marker_id: int
    points: List[Vec3]

    @validator('points', pre=True, each_item=True)
    def validate_points(cls, value):
        if isinstance(value, dict):
            return Vec3(**value)
        if isinstance(value, Vec3):
            return value
        raise ValueError("Invalid type for point")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "marker_id": self.marker_id,
            "points": [point.as_dict() for point in self.points]
        }


class ArucoBoardSpecification(BaseModel):
    board_markers: List[BoardMarker]

    @validator('board_markers', pre=True, each_item=True)
    def validate_board_markers(cls, value):
        if isinstance(value, dict):
            return BoardMarker(**value)
        if isinstance(value, BoardMarker):
            return value
        raise ValueError("Invalid type for board marker")

    def as_dict(self) -> Dict[str, Any]:
        return {"board_markers": [board_marker.as_dict() for board_marker in self.board_markers]}


def read_file(input_filepath: str) -> ArucoBoardSpecification:
    with open(input_filepath, 'r') as file:
        data = hjson.load(file)
    return ArucoBoardSpecification(**data)


def write_file(output_filepath: str, output_board: ArucoBoardSpecification) -> None:
    with open(output_filepath, 'w') as file:
        json.dump(output_board.as_dict(), file, indent=4)
