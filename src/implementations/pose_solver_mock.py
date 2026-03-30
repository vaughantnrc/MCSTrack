from src.common import \
    Matrix4x4, \
    PoseSolver
import datetime


class MockPoseSolver(PoseSolver):
    """
    The Mock classes are relatively simple implementations made for testing connectivity functionality.
    """

    def update(self) -> None:
        if self._last_updated_timestamp_utc >= self._last_change_timestamp_utc:
            return
        self._last_updated_timestamp_utc = datetime.datetime.now(tz=datetime.timezone.utc)
        self._poses_by_detector_label.clear()
        for detector_label in self._detector_records_by_detector_label.keys():
            self._poses_by_target_label[detector_label] = Matrix4x4()  # Default, identity
        self._poses_by_target_label.clear()
        for target in self._targets:
            target_label: str = target.label
            self._poses_by_target_label[target_label] = Matrix4x4()  # Default, identity
