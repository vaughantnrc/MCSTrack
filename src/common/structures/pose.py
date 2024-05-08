from .matrix4x4 import Matrix4x4
import datetime
from pydantic import BaseModel, Field
import uuid


class Pose(BaseModel):
    target_id: str = Field()
    object_to_reference_matrix: Matrix4x4 = Field()
    solver_timestamp_utc_iso8601: str = Field()
