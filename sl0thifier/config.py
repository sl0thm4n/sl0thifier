from pathlib import Path

from pydantic import BaseModel


class ModelPaths(BaseModel):
    birefnet: Path
    realesrgan_exe: Path
    realesrgan_models: Path

class Config(BaseModel):
    log_dir: Path = Path("logs")
    thumb_size: int = 128
    pad: int = 16
    grid_cols: int = 5
    model_paths: ModelPaths

    class Config:
        arbitrary_types_allowed = True

def load_config() -> Config:
    base_dir = Path(__file__).resolve().parent
    return Config(
        model_paths=ModelPaths(
            birefnet=base_dir / "models" / "birefnet.onnx",
            realesrgan_exe=base_dir / "realesrgan" / "realesrgan-ncnn-vulkan.exe",
            realesrgan_models=base_dir / "realesrgan" / "models"
        )
    )
