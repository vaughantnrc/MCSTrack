from pydantic import BaseModel, Field


class CameraConnection(BaseModel):
    """
    Information used to connect to a camera on the current device.
    """
    usb_id: int = Field()
