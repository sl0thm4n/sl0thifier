# sl0thfier/utils.py

from typing import Tuple

from PIL import Image


def ensure_png_alpha(img: Image.Image) -> Image.Image:
    """Ensure image is in RGBA mode."""
    return img.convert("RGBA")

def parse_hex_color(s: str) -> Tuple[int, int, int]:
    """Convert a hex string or known color name to RGB tuple."""
    s = s.strip().lower()
    if s in ("black", "white"):
        return (0, 0, 0) if s == "black" else (255, 255, 255)
    if s.startswith("#") and len(s) == 7:
        return int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)
    return (0, 255, 0)  # default fallback: green
