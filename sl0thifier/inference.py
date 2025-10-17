import os
import subprocess
import traceback

from PIL import Image

from sl0thfier.logger import setup_logger
from sl0thfier.utils import ensure_png_alpha, parse_hex_color

log = setup_logger()


def realesrgan_upscale(
    in_path: str, out_path: str, exe_path: str, model_dir: str
) -> bool:
    # ────────────────────────────────
    # RealESRGAN (Vulkan)
    # ────────────────────────────────
    if not os.path.exists(exe_path):
        log.warning("RealESRGAN binary not found: %s", exe_path)
        return False
    if not os.path.isdir(model_dir):
        log.warning("RealESRGAN model directory not found: %s", model_dir)
        return False

    cmd = [
        exe_path,
        "-i",
        in_path,
        "-o",
        out_path,
        "-m",
        model_dir,
        "-n",
        "realesrgan-x4plus",
        "-s",
        "4",
        "-g",
        "0",
        "-j",
        "1:1:1",
        "-f",
        "png",
    ]
    log.info("Running RealESRGAN command: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, timeout=900)
        ok = os.path.exists(out_path)
        log.info("Upscale successful? %s -> %s", ok, out_path)
        return ok
    except subprocess.TimeoutExpired:
        log.error("RealESRGAN process timed out")
        return False
    except Exception:
        log.error("RealESRGAN error:\n%s", traceback.format_exc())
        return False


# ────────────────────────────────
# BiRefNet (ONNX) Background Removal
# ────────────────────────────────
bg_session = None

try:
    import onnxruntime as ort

    av = ort.get_available_providers()
    log.info("ONNX available providers: %s", av)
    if "CUDAExecutionProvider" in av:
        PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif "DmlExecutionProvider" in av:
        PROVIDERS = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        PROVIDERS = ["CPUExecutionProvider"]
except Exception:
    log.error("Failed to import onnxruntime:\n%s", traceback.format_exc())
    ort = None
    PROVIDERS = ["CPUExecutionProvider"]


def load_birefnet(model_path: str):
    global bg_session
    if ort and os.path.exists(model_path):
        try:
            log.info("Loading BiRefNet: %s", model_path)
            bg_session = ort.InferenceSession(model_path, providers=PROVIDERS)
            log.info("BiRefNet loaded. Providers: %s", bg_session.get_providers())
        except Exception:
            log.error("Failed to initialize BiRefNet:\n%s", traceback.format_exc())
            bg_session = None
    else:
        log.warning("BiRefNet model not found: %s", model_path)


def birefnet_remove_bg(img: Image.Image, bg_choice: str) -> Image.Image:
    log.info("BiRefNet background removal started")

    if bg_session is None:
        log.warning("BiRefNet session is not initialized. Returning original image.")
        return ensure_png_alpha(img)

    import cv2
    import numpy as np

    try:
        input_name = bg_session.get_inputs()[0].name
        output_name = bg_session.get_outputs()[0].name
        expected_shape = bg_session.get_inputs()[0].shape  # [1, 3, H, W]
        target_w = int(expected_shape[3])
        target_h = int(expected_shape[2])
        log.info("Model input size: %dx%d", target_w, target_h)

        np_img = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        h, w = np_img.shape[:2]

        np_in = cv2.resize(np_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        np_in = np.transpose(np_in, (2, 0, 1))[None, :, :, :]

        result = bg_session.run([output_name], {input_name: np_in})[0]

        mask = result.squeeze()
        mask = (mask > 0.5).astype("uint8") * 255
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        rgba = img.convert("RGBA")
        np_rgba = np.array(rgba)

        if bg_choice.lower() == "transparent":
            np_rgba[:, :, 3] = mask
            log.info("Applied transparent background.")
        else:
            r, g, b = parse_hex_color(bg_choice)
            bg = np.full_like(np_rgba, (r, g, b, 255))
            alpha = (mask.astype("float32") / 255.0)[..., None]
            np_rgba = (
                np_rgba.astype("float32") * alpha + bg.astype("float32") * (1 - alpha)
            ).astype("uint8")
            log.info("Applied solid background color: %s", bg_choice)

        out = Image.fromarray(np_rgba, "RGBA")
        log.info("Background removal completed successfully.")
        return out

    except Exception:
        log.error("BiRefNet inference failed:\n%s", traceback.format_exc())
        return ensure_png_alpha(img)
