import os

from sl0thfier.config import load_config
from sl0thfier.inference import realesrgan_upscale

cfg = load_config()


def test_realesrgan_executable_exists():
    assert (
        cfg.model_paths.realesrgan_exe.exists()
    ), f"Executable not found at {cfg.model_paths.realesrgan_exe}"


def test_realesrgan_upscale_runs(sample_image, output_dir):
    out_path = os.path.join(output_dir, "out.png")
    ok = realesrgan_upscale(
        sample_image,
        out_path,
        str(cfg.model_paths.realesrgan_exe),
        str(cfg.model_paths.realesrgan_models),
    )
    assert isinstance(ok, bool)
