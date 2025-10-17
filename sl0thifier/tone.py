# sl0thfier/tone.py

import traceback

from PIL import Image, ImageEnhance

from sl0thfier.logger import setup_logger

log = setup_logger()


def tone_correction(img: Image.Image) -> Image.Image:
    try:
        img = ImageEnhance.Color(img).enhance(1.08)
        img = ImageEnhance.Brightness(img).enhance(1.02)
        return img
    except Exception:
        log.warning("Tone correction failed:\n%s", traceback.format_exc())
        return img
