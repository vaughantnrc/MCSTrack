from .marker_ray_set import MarkerRaySet
from src.common.structures import Matrix4x4
import datetime
from pydantic import BaseModel, Field


# TODO: Merge/replace this with pose under common data structures
class PoseData(BaseModel):
    target_id: str = Field()
    object_to_reference_matrix: Matrix4x4 = Field()
    ray_sets: list[MarkerRaySet]

    def newest_timestamp(self) -> datetime.datetime:
        return MarkerRaySet.newest_timestamp_in_list(self.ray_sets)

    def oldest_timestamp(self) -> datetime.datetime:
        return MarkerRaySet.oldest_timestamp_in_list(self.ray_sets)

    @staticmethod
    def age_seconds(
        pose,
        query_timestamp: datetime.datetime
    ) -> float:
        return (query_timestamp - pose.oldest_timestamp()).total_seconds()
