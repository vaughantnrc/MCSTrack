from src.common.structures import Matrix4x4
import datetime
from pydantic import BaseModel, Field


class MarkerRaySet(BaseModel):
    marker_id: int = Field()
    image_points: list[list[float]] = Field()  # image positions of marker corners. Size 4.
    image_timestamp: datetime.datetime = Field()
    ray_origin_reference: list[float] = Field()  # Shared origin for all rays (same detector)
    ray_directions_reference: list[list[float]] = Field()  # Size 4 (one for each image point)
    detector_label: str = Field()
    detector_to_reference_matrix: Matrix4x4 = Field()

    @staticmethod
    def age_seconds(
        marker_ray_set,
        query_timestamp: datetime.datetime
    ):
        return (query_timestamp - marker_ray_set.image_timestamp).total_seconds()

    @staticmethod
    def newest_timestamp_in_list(marker_ray_set_list: list) -> datetime.datetime:
        return_value = datetime.datetime.now()
        for ray_set in marker_ray_set_list:
            if ray_set.image_timestamp > return_value:
                return_value = ray_set.image_timestamp
        return return_value

    @staticmethod
    def oldest_timestamp_in_list(marker_ray_set_list: list) -> datetime.datetime:
        return_value = datetime.datetime.utcfromtimestamp(0)
        for ray_set in marker_ray_set_list:
            if ray_set.image_timestamp > return_value:
                return_value = ray_set.image_timestamp
        return return_value
