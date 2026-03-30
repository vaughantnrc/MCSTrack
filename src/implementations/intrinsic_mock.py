from src.common import \
    ImageResolution, \
    IntrinsicCalibration, \
    IntrinsicCalibrator, \
    IntrinsicParameters
import datetime


class MockIntrinsicCalibrator(IntrinsicCalibrator):
    """
    The Mock classes are relatively simple implementations made for testing connectivity functionality.
    """

    def _calculate_implementation(
        self,
        image_resolution: ImageResolution,
        image_metadata_list: list[IntrinsicCalibrator.ImageMetadata]
    ) -> tuple[IntrinsicCalibration, list[IntrinsicCalibrator.ImageMetadata]]:
        intrinsic_calibration: IntrinsicCalibration = IntrinsicCalibration(
            timestamp_utc=datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
            image_resolution=image_resolution,
            calibrated_values=IntrinsicParameters.generate_zero_parameters(
                resolution_x_px=image_resolution.x_px,
                resolution_y_px=image_resolution.y_px),
            supplemental_data=dict())
        return intrinsic_calibration, image_metadata_list
