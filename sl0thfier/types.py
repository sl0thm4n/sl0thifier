from typing import Literal

from pydantic import BaseModel


class ImageTask(BaseModel):
    path: str
    background: Literal["transparent", "black", "white", "#00FF00"] = "transparent"
    tone_correction: bool = True
