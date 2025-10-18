import subprocess
import tempfile
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from pydantic import BaseModel

from sl0thifier.exceptions import ModelNotInstalled
from sl0thifier.logger import logger

WORKING_DIR = Path(__file__).parent.resolve() / ".."


class Sl0thifierBaseClass(BaseModel):
    """Abstract base class for all image models."""

    @staticmethod
    def _create_tmp(name: str) -> Path:
        """Get temporary file path in the system temp directory."""
        return Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}-{name}"

    @staticmethod
    def _cleanup_tmp(path: Path) -> None:
        """Remove temporary file if it exists."""
        try:
            path.unlink()
        except FileNotFoundError:
            logger.warning("Temp file %s not found for cleanup.", path)

    @abstractmethod
    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        """Process the input image.

        Args:
            img (Image.Image): Image instance to process.

        Raises:
            sl0thifier.exceptions.ModelNotInstalled: Raised when required model(s) are not installed.

        Returns:
            Image.Image: Image instance after processing.
        """
        pass


class ImageUpscaler(Sl0thifierBaseClass):
    """Image upscaler class."""

    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        model_name = kwargs.get("model_name", "realesrgan-x4plus")  # 🧼 Default preset
        scale = kwargs.get("scale", 4)

        exe = WORKING_DIR / "realesrgan" / "realesrgan-ncnn-vulkan.exe"
        if not exe.exists():
            raise ModelNotInstalled("Real-ESRGAN binary not found. Run `slth install`.")

        input_path = self._create_tmp("slth_in.png")
        output_path = self._create_tmp("slth_out.png")

        try:
            img.save(input_path)

            cmd = [
                str(exe),
                "-i",
                str(input_path),
                "-o",
                str(output_path),
                "-n",
                model_name,
                "-s",
                str(scale),
            ]

            subprocess.run(cmd, check=True)
            result = Image.open(output_path).convert("RGB")

            logger.info("🦥 Upscaled ×%s using '%s'", scale, model_name)
            return result

        except subprocess.CalledProcessError as e:
            logger.error("❌ Real-ESRGAN failed: %s", e)
            raise RuntimeError("Real-ESRGAN execution failed") from e

        finally:
            self._cleanup_tmp(input_path)
            self._cleanup_tmp(output_path)


class ImageEnhancer(Sl0thifierBaseClass):
    """Image enhancer class using CLAHE (OpenCV)."""

    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        """Enhance the input image using CLAHE.

        Args:
            img (Image.Image): Image instance to enhance.

        Returns:
            Image.Image: Enhanced image instance.
        """
        # Convert PIL image to NumPy array (RGB)
        rgb_array = np.array(img.convert("RGB"))

        try:
            # Convert to LAB color space
            lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
            lightness_channel, a_channel, b_channel = cv2.split(lab_array)

            # Apply CLAHE to the lightness channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_lightness = clahe.apply(lightness_channel)

            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge((enhanced_lightness, a_channel, b_channel))
            enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)

        except cv2.error as e:
            logger.error("❌ OpenCV error during CLAHE enhancement: %s", e)
            raise RuntimeError("CLAHE enhancement failed due to OpenCV error") from e

        logger.info("🦥 Enhanced image using CLAHE")

        return Image.fromarray(enhanced_rgb)


class ImageBackgroundRemover(Sl0thifierBaseClass):
    """Image background remover using BiRefNet ONNX model.

    Download backend model from:
    https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-resolution_512x512-fp16-epoch_216.onnx


    Args:
        Sl0thifierBaseClass (_type_): _description_
    """

    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        bg_color = kwargs.get("bg_color", "None")

        model_path = WORKING_DIR / "birefnet" / "birefnet.onnx"
        if not model_path.exists():
            raise ModelNotInstalled(
                "BiRefNet ONNX model not found. Run `slth install`."
            )

        # Resize input image to (512, 512)
        resized_img = img.convert("RGB").resize((512, 512))
        np_img = np.array(resized_img).astype(np.float32) / 255.0
        input_tensor = np.transpose(np_img, (2, 0, 1))[None, ...]  # (1, 3, 512, 512)

        try:
            ort_session = ort.InferenceSession(str(model_path))
            outputs = ort_session.run(["output_image"], {"input_image": input_tensor})
            alpha_mask = outputs[0][0, 0]  # shape: (512, 512)
        except Exception as e:
            logger.error("❌ BiRefNet ONNX inference failed: %s", e)
            raise RuntimeError("BiRefNet inference failed") from e

        # Resize alpha mask to match original image size
        alpha_mask_resized = cv2.resize(
            alpha_mask, img.size, interpolation=cv2.INTER_LINEAR
        )
        alpha_uint8 = (alpha_mask_resized * 255).clip(0, 255).astype(np.uint8)

        rgba_array = np.dstack((np.array(img.convert("RGB")), alpha_uint8))
        logger.info("🦥 Removed background using BiRefNet")

        # Add background color handling here if bg_color is provided. If color is None,
        # keep transparency.

        final_img = Image.fromarray(rgba_array, "RGBA")

        if bg_color not in (
            "none",
            "None",
            None,
        ):
            try:
                # Create background image
                bg = Image.new("RGBA", final_img.size, bg_color)
                # Composite foreground and background
                final_img = Image.alpha_composite(bg, final_img).convert("RGB")
                logger.info("🦥 Applied background color: %s", bg_color)
                return final_img
            except ValueError:
                logger.error("❌ Error applying alpha mask for background removal.")
                raise
        return final_img


class KingSl0th(Sl0thifierBaseClass):
    """Composite model that applies upscaling and enhancement."""

    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        output_width: Optional[int] = kwargs.get("output_width", 512)
        output_height: Optional[int] = kwargs.get("output_height", 512)
        remove_bg = kwargs.get("remove_bg", False)

        upscaler = ImageUpscaler()
        enhancer = ImageEnhancer()

        upscaled_img = upscaler.sl0thify(img, **kwargs)
        enhanced_img = enhancer.sl0thify(upscaled_img, **kwargs)

        if remove_bg:
            bg_remover = ImageBackgroundRemover()
            enhanced_img = bg_remover.sl0thify(enhanced_img, **kwargs)

        # Resize to output dimensions if specified
        if output_width and output_height:
            enhanced_img = enhanced_img.resize(
                (output_width, output_height), Image.Resampling.LANCZOS
            )

        return enhanced_img
