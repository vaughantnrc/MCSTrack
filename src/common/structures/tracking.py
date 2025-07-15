from .image import Annotation, ImageResolution
from .linear_algebra import Pose
import datetime
from pydantic import BaseModel, Field


class DetectorFrame(BaseModel):
    annotations: list[Annotation] = Field(default_factory=list)
    timestamp_utc_iso8601: str = Field()
    image_resolution: ImageResolution = Field()

    @property
    def annotations_identified(self):
        return [
            annotation
            for annotation in self.annotations
            if annotation.feature_label != Annotation.UNIDENTIFIED_LABEL]

    @property
    def annotations_unidentified(self):
        return [
            annotation
            for annotation in self.annotations
            if annotation.feature_label == Annotation.UNIDENTIFIED_LABEL]

    @property
    def timestamp_utc(self):
        return datetime.datetime.fromisoformat(self.timestamp_utc_iso8601)


class PoseSolverFrame(BaseModel):
    detector_poses: list[Pose] | None = Field()
    target_poses: list[Pose] | None = Field()
    timestamp_utc_iso8601: str = Field()

    def timestamp_utc(self):
        return datetime.datetime.fromisoformat(self.timestamp_utc_iso8601)
