class MCTError(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class MCTParsingError(MCTError):
    message: str

    def __init__(self, message: str, *args):
        super().__init__(args)
        self.message = message
