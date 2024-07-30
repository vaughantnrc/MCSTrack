import datetime

# TODO: Remove this in favour of DetectorFrame or another data structure
class MarkerCorners:
    detector_label: str
    marker_id: int
    points: list[list[float]]
    timestamp: datetime.datetime

    def __init__(
        self,
        detector_label: str,
        marker_id: int,
        points: list[list[float]],
        timestamp: datetime.datetime
    ):
        self.detector_label = detector_label
        self.marker_id = marker_id
        self.points = points
        self.timestamp = timestamp
