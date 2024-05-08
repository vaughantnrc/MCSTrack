from .base_component_connection_dynamic import BaseComponentConnectionDynamic


class DetectorComponentConnectionDynamic(BaseComponentConnectionDynamic):

    stream_image: bool
    stream_markers: bool

    def __init__(self):
        super().__init__()
        self.stream_image = False
        self.stream_markers = False
