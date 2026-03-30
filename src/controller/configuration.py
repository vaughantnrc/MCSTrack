from src.common import \
    DetectorPoseMode, \
    KeyValueSimpleAny, \
    Matrix4x4, \
    SinkConfiguration, \
    Target
from pydantic import BaseModel, Field, SerializeAsAny


class MCTComponentConfig(BaseModel):
    label: str = Field()
    ip_address: str = Field()
    port: int = Field()


class DetectorComponentConfig(MCTComponentConfig):
    camera_parameters: list[SerializeAsAny[KeyValueSimpleAny]] = Field(default_factory=list)
    annotator_parameters: list[SerializeAsAny[KeyValueSimpleAny]] = Field(default_factory=list)


class MixerDetectorConfig(BaseModel):
    """
    The per-detector configuration that is stored for and will be applied for the Mixer.
    """
    detector_label: str = Field()
    pose_mode: DetectorPoseMode = Field(default_factory=DetectorPoseMode.default_mode)
    detector_to_reference: Matrix4x4 | None = Field(default=None)


class MixerConfig(MCTComponentConfig):
    solver_parameters: list[SerializeAsAny[KeyValueSimpleAny]] | None = Field(default=None)
    detectors: list[MixerDetectorConfig] | None = Field(default=None)
    targets: list[Target] | None = Field(default=None)


class MCTConfiguration(BaseModel):
    detectors: list[DetectorComponentConfig] = Field(default_factory=list)
    mixers: list[MixerConfig] = Field(default_factory=list)
    sinks: list[SinkConfiguration] = Field(default_factory=list)
