from src.implementations.common_aruco_opencv import ArucoOpenCVCommon
from src.common.structures import Annotation, Landmark, Target


board: ArucoOpenCVCommon.CharucoBoard = ArucoOpenCVCommon.CharucoBoard()
points: list[list[float]] = board.get_marker_corner_points()
landmarks: list[Landmark] = list()
POINTS_PER_MARKER: int = 4
marker_count: int = round(int(len(points) / POINTS_PER_MARKER))
for marker_index in range(marker_count):
    point_start_index: int = marker_index * POINTS_PER_MARKER
    marker_points = points[point_start_index: point_start_index + POINTS_PER_MARKER]
    for corner_index, corner_point in enumerate(marker_points):
        landmarks.append(Landmark(
            feature_label=f"{marker_index}{Annotation.RELATION_CHARACTER}{corner_index}",
            x=corner_point[0], y=corner_point[1], z=corner_point[2]))
target: Target = Target(
    label="CharucoBoard",
    landmarks=landmarks)
with open("temp.json", 'w') as outfile:
    outfile.write(target.model_dump_json(exclude_unset=True, indent=2))
