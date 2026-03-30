from .controller import MCTController
import datetime


controller: MCTController = MCTController()
controller.configure_from_filepath("/workspace/dev/MCSTrack/data/configuration/controller/mock.json")
controller.start_up()
start_time: datetime.datetime = datetime.datetime.now(tz=datetime.timezone.utc)
now_time: datetime.datetime = start_time
while (now_time - start_time).total_seconds() < 5:
    controller.update()
    now_time = datetime.datetime.now(tz=datetime.timezone.utc)
controller.shut_down()
while (now_time - start_time).total_seconds() < 1:
    controller.update()
    now_time = datetime.datetime.now(tz=datetime.timezone.utc)
