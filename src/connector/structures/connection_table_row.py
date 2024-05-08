from pydantic import BaseModel, Field


class ConnectionTableRow(BaseModel):
    label: str = Field()
    role: str = Field()
    ip_address: str = Field()
    port: int = Field()
    status: str = Field()

