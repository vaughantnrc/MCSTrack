from fastapi import Request, WebSocket


def client_identifier_from_connection(
    connection: Request | WebSocket
) -> str:
    return f"{connection.client.host}:{connection.client.port}"
