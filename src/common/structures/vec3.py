from pydantic import BaseModel, Field


class Vec3(BaseModel):
    x: float = Field()
    y: float = Field()
    z: float = Field()
