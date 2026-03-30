from .controller import MCTController
import datetime
import logging


logging.basicConfig(level=logging.DEBUG)
controller: MCTController = MCTController(send_status_messages_to_logger=True)
controller.configure_from_filepath("/workspace/dev/MCSTrack/data/configuration/controller/mock.json")

start_time: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
now_time: datetime.datetime = start_time

logging.debug("startup")
controller.start_up()
logging.debug("startup updates")
while (now_time - start_time).total_seconds() < 1:
    controller.update()
    if controller.get_controller_state() == MCTController.State.RUNNING:
        break
    now_time = datetime.datetime.now(tz=datetime.timezone.utc)

if controller.get_controller_state() != MCTController.State.RUNNING:
    exit(-1)

logging.debug("run updates")
frame_count: int = 5
for _ in range(0,frame_count):
    controller.update()

logging.debug("shutdown")
controller.shut_down()
logging.debug("shutdown updates")
while (now_time - start_time).total_seconds() < 1:
    controller.update()
    if controller.get_controller_state() == MCTController.State.IDLE:
        break
    now_time = datetime.datetime.now(tz=datetime.timezone.utc)

if controller.get_controller_state() != MCTController.State.IDLE:
    exit(-1)
