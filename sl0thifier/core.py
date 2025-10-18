# sl0thifier/core.py

import cv2
import numpy as np
from PIL import Image


def read_image(path: str) -> np.ndarray:
    """Read image from disk and return as RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))


def save_image(img: np.ndarray, path: str) -> None:
    """Save RGB numpy array to disk."""
    Image.fromarray(img).save(path)


def resize_image(img: np.ndarray, scale: int = 2) -> np.ndarray:
    """Resize image using bicubic interpolation."""
    height, width = img.shape[:2]
    return cv2.resize(
        img, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC
    )


def normalize(img: np.ndarray) -> np.ndarray:
    """Normalize to 0.0–1.0 range (float32)."""
    return img.astype(np.float32) / 255.0


def denormalize(img: np.ndarray) -> np.ndarray:
    """Convert float32 0–1 image to uint8 RGB."""
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)
