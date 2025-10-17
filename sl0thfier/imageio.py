import os
from typing import Tuple

from PIL import Image

from sl0thfier.logger import setup_logger

log = setup_logger()

def save_resized_all(img: Image.Image, src_path: str) -> Tuple[str, str]:
    src_dir = os.path.abspath(os.path.dirname(src_path))
    base = os.path.splitext(os.path.basename(src_path))[0]

    out512_dir = os.path.join(src_dir, "512x512")
    out1024_dir = os.path.join(src_dir, "1024x1024")
    os.makedirs(out512_dir, exist_ok=True)
    os.makedirs(out1024_dir, exist_ok=True)

    out512 = os.path.join(out512_dir, f"{base}_512.png")
    out1024 = os.path.join(out1024_dir, f"{base}_1024.png")

    img.resize((512, 512), Image.Resampling.LANCZOS).save(out512, format="PNG")
    img.resize((1024, 1024), Image.Resampling.LANCZOS).save(out1024, format="PNG")

    log.info(f"✅ Saved: {out512}")
    log.info(f"✅ Saved: {out1024}")
    return out512, out1024
