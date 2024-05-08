from typing import Final
from enum import IntEnum


class PoseSolverStatus:

    class Solve(IntEnum):
        STOPPED: Final[int] = 0
        RUNNING: Final[int] = 1
        FAILURE: Final[int] = 2

    solve_status: Solve
    solve_errors: list[str]

    def __init__(self):
        self.solve_status = PoseSolverStatus.Solve.STOPPED
        self.solve_errors = list()

    def in_runnable_state(self):
        return self.solve_status == PoseSolverStatus.Solve.RUNNING
