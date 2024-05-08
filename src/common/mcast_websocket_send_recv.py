from src.common import MCastRequestSeries, MCastResponseSeries
import json
from pydantic import BaseModel
from typing import Callable
from websockets import WebSocketClientProtocol


async def mcast_websocket_send_recv(
    websocket: WebSocketClientProtocol,
    request_series: MCastRequestSeries,
    response_series_type: type[MCastResponseSeries | dict] | None = dict,
    response_series_converter: Callable[[dict], MCastResponseSeries] | None = None
) -> MCastResponseSeries | dict | None:
    """
    Send data via a websocket and get the response (if specified).
    The response can be a subclass of pydantic's BaseModel, a dict, or None, according to response_type.
    """
    request_series_dict: dict
    if isinstance(request_series, BaseModel):
        request_series_dict = request_series.dict()
    else:
        request_series_dict = dict(request_series)
    request_series_str: str = json.dumps(request_series_dict)
    await websocket.send(request_series_str)
    if response_series_type is None:
        return None
    response_series_str: str = await websocket.recv()
    response_series_dict: dict = json.loads(response_series_str)
    if response_series_type is dict:
        return response_series_dict
    assert isinstance(response_series_type, type)
    response_series: MCastResponseSeries
    if response_series_converter:
        response_series = response_series_converter(response_series_dict)
    else:
        response_series = response_series_type(**response_series_dict)
    return response_series
