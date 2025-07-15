from src.common import MCTError


class PoseSolverException(MCTError):
    message: str

    def __init__(self, message: str, *args, **kwargs):
        super().__init__(args, kwargs)
        self.message = message
