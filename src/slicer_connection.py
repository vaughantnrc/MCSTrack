import asyncio
import sys
import hjson
import numpy as np
import pyigtl
import time as t
import logging

from src.common.api import MCTRequestSeries
from src.common.structures.mct_component import COMPONENT_ROLE_LABEL_DETECTOR, COMPONENT_ROLE_LABEL_POSE_SOLVER
from src.common.structures import TargetBase
from src.controller.mct_controller import MCTController
from ipaddress import IPv4Address

from src.pose_solver.api import PoseSolverAddTargetMarkerRequest
from src.pose_solver.api import TargetMarker
from src.pose_solver.api import PoseSolverSetReferenceRequest

from src.controller import Connection

# Input filepath is specified by command line arguments
if len(sys.argv) < 2:
    raise Exception("No input filepath specified")
input_filepath = sys.argv[1]

with open(input_filepath, 'r') as file:
    data = hjson.load(file)

server = pyigtl.OpenIGTLinkServer(port=18944,local_server=False)

while not server.is_connected():
    t.sleep(0.1)

logging.basicConfig(level=logging.INFO)

controller = MCTController(
    serial_identifier="controller",
    send_status_messages_to_logger=True)

controller.add_status_message(
        severity="info",
        message=f"Slicer client connected")

controller.start_from_configuration_filepath(input_filepath)

while True:
    controller.update()

    done_transitioning: bool = (not controller.is_transitioning())
    if done_transitioning:
        ps_frame = controller.get_live_pose_solver_frame("sol")
        timestamp = ps_frame.timestamp_utc_iso8601

        if len(ps_frame.target_poses) > 0:
            target_poses = ps_frame.target_poses[0]
            message = pyigtl.TransformMessage(
                matrix=target_poses.object_to_reference_matrix.as_numpy_array(),
                # timestamp=target_poses.solver_timestamp_utc_iso8601,
                device_name=target_poses.target_id
            )
            server.send_message(message,wait=True)
            break

