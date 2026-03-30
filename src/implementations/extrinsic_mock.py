from src.common import \
    ExtrinsicCalibration, \
    ExtrinsicDetectorCalibration, \
    ExtrinsicCalibrator, \
    Matrix4x4
import datetime


class MockExtrinsicCalibrator(ExtrinsicCalibrator):
    """
    The Mock classes are relatively simple implementations made for testing connectivity functionality.
    """

    def _calculate_implementation(
        self,
        image_metadata_list: list[ExtrinsicCalibrator.ImageMetadata]
    ) -> tuple[ExtrinsicCalibration, list[ExtrinsicCalibrator.ImageMetadata]]:
        detector_set: set[str] = set()
        for image_metadata in image_metadata_list:
            detector_set.add(image_metadata.detector_label)
        extrinsic_calibration: ExtrinsicCalibration = ExtrinsicCalibration(
            timestamp_utc=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            calibrated_values=[
                ExtrinsicDetectorCalibration(
                    detector_label=detector_label,
                    detector_to_reference=Matrix4x4())
                for detector_label in detector_set],
            supplemental_data=dict())
        return extrinsic_calibration, image_metadata_list
