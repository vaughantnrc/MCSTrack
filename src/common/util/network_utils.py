from fastapi import Request, WebSocket


class NetworkUtils:
    """
    A "class" to group related static functions, like in a namespace.
    The class itself is not meant to be instantiated.
    """

    def __init__(self):
        raise RuntimeError(f"{__class__.__name__} is not meant to be instantiated.")

    @staticmethod
    def client_identifier_from_connection(
        connection: Request | WebSocket
    ) -> str:
        return f"{connection.client.host}:{connection.client.port}"
