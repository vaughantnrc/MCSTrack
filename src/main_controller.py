from .controller import MCTController
import datetime
import logging
import os
from typing import Final


class MainController:
    _image_identifier: str

    def main(self):

        CONFIGURATION_FILEPATH_ENV_VAR: Final[str] = "MCSTRACK_CONTROLLER_CONFIGURATION_FILEPATH"
        configuration_filepath: str = os.path.join(
            os.path.dirname(__file__), "..", "data", "configuration", "controller", "mock.json")
        configuration_filepath = os.getenv(CONFIGURATION_FILEPATH_ENV_VAR, configuration_filepath)

        logging.basicConfig(level=logging.INFO)
        controller: MCTController = MCTController(send_status_messages_to_logger=True)
        controller.configure_from_filepath(configuration_filepath)

        start_time: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
        now_time: datetime.datetime = start_time

        logging.info("startup")
        controller.start_up()
        logging.info("startup updates")
        while (now_time - start_time).total_seconds() < 1:
            controller.update()
            if controller.get_controller_state() == MCTController.State.RUNNING:
                break
            now_time = datetime.datetime.now(tz=datetime.timezone.utc)

        if controller.get_controller_state() != MCTController.State.RUNNING:
            exit(-1)

        logging.info("run updates")
        frame_count: int = 5
        for _ in range(0,frame_count):
            controller.update()

        logging.info("image add")
        def _on_image_add_callback(component_label: str, image_identifier: str):
            logging.info(f"  Retrieved from {component_label} result identifier {image_identifier}")
            self._image_identifier = image_identifier
        controller.calibrate_intrinsic_image_add(
            detector_label="det",
            callback=_on_image_add_callback)
        while (now_time - start_time).total_seconds() < 1:
            controller.update()
            if not controller.is_user_task_running():
                break
            now_time = datetime.datetime.now(tz=datetime.timezone.utc)

        logging.info("image get")
        def _on_image_get_callback(component_label: str, image_base64: str):
            logging.info(f"  Retrieved from {component_label} image string of length {len(image_base64)}")
        controller.calibrate_intrinsic_image_get(
            detector_label="det",
            image_identifier=self._image_identifier,
            callback=_on_image_get_callback)
        while (now_time - start_time).total_seconds() < 1:
            controller.update()
            if not controller.is_user_task_running():
                break
            now_time = datetime.datetime.now(tz=datetime.timezone.utc)

        logging.info("shutdown")
        controller.shut_down()
        logging.info("shutdown updates")
        while (now_time - start_time).total_seconds() < 1:
            controller.update()
            if controller.get_controller_state() == MCTController.State.CONFIGURED:
                break
            now_time = datetime.datetime.now(tz=datetime.timezone.utc)

        if controller.get_controller_state() != MCTController.State.CONFIGURED:
            exit(-1)


if __name__ == "__main__":
    MainController().main()
