from pydantic import BaseModel, Field


class ImageRecorderConfiguration(BaseModel):
    image_path: str = Field()
    min_disk_byte_count: int = Field(default=100000000)  # 100 MB
