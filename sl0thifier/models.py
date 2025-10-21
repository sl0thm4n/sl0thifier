import platform
import shutil
import subprocess
import tempfile
import uuid
from abc import abstractmethod
from pathlib import Path
from typing import ClassVar, Optional
import zipfile

import cv2
import httpx
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
    """Image upscaler using Real-ESRGAN NCNN-Vulkan executable."""

    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        model_name = kwargs.get("model_name", "realesrgan-x4plus")
        scale = kwargs.get("scale", 4)

        exe = self.ensure_model()

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
            logger.error("❌ Real-ESRGAN execution failed: %s", e)
            raise RuntimeError("Real-ESRGAN execution failed") from e

        finally:
            self._cleanup_tmp(input_path)
            self._cleanup_tmp(output_path)

    def ensure_model(self) -> Path:
        """Ensure the Real-ESRGAN binary is downloaded and ready to use."""
        system = platform.system()
        realesrgan_dir = WORKING_DIR / "realesrgan"

        match system:
            case "Windows":
                exe_name = "realesrgan-ncnn-vulkan.exe"
                zip_url = "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v20220424/realesrgan-ncnn-vulkan-20220424-windows.zip"
            case "Linux":
                exe_name = "realesrgan-ncnn-vulkan"
                zip_url = "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v20220424/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
            case "Darwin":
                exe_name = "realesrgan-ncnn-vulkan"
                zip_url = "https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan/releases/download/v20220424/realesrgan-ncnn-vulkan-20220424-macos.zip"
            case _:
                raise RuntimeError(f"Unsupported OS: {system}")

        exe = realesrgan_dir / exe_name
        if exe.exists():
            return exe

        logger.warning("Real-ESRGAN binary not found. Downloading...")

        realesrgan_dir.mkdir(parents=True, exist_ok=True)
        zip_path = realesrgan_dir / "realesrgan.zip"

        # Download zip via httpx
        try:
            with httpx.stream("GET", zip_url, timeout=60.0) as response:
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
            logger.info("✅ Downloaded Real-ESRGAN package.")
        except Exception as e:
            logger.error("❌ Failed to download Real-ESRGAN: %s", e)
            raise ModelNotInstalled("Could not download Real-ESRGAN package.") from e

        # Extract & find binary
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(realesrgan_dir)
            logger.info("✅ Extracted Real-ESRGAN package.")

            inner_exe = next(realesrgan_dir.rglob(exe_name), None)
            if inner_exe and inner_exe != exe:
                shutil.move(str(inner_exe), str(exe))

            if system != "Windows" and exe.exists():
                exe.chmod(exe.stat().st_mode | 0o111)
                logger.info("✅ Set execute permissions.")
        except Exception as e:
            logger.error("❌ Failed to extract or prepare Real-ESRGAN: %s", e)
            raise ModelNotInstalled("Failed to extract or prepare Real-ESRGAN.") from e
        finally:
            if zip_path.exists():
                zip_path.unlink()

        if not exe.exists():
            raise ModelNotInstalled("Real-ESRGAN executable not found after install.")

        return exe


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
        clip_limit = kwargs.get("clip_limit", 1.0)
        tile_size = kwargs.get("tile_size", 4)
        rgb_array = np.array(img.convert("RGB"))

        try:
            # Convert to LAB color space
            lab_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
            lightness_channel, a_channel, b_channel = cv2.split(lab_array)

            # Apply CLAHE to the lightness channel
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)
            )
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

    Downloads the model automatically if not found.

    Uses BiRefNet:
    https://github.com/ZhengPeng7/BiRefNet
    """

    MODEL_URL: ClassVar[str] = (
        "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-resolution_512x512-fp16-epoch_216.onnx"
    )
    MODEL_FILENAME: ClassVar[str] = "birefnet.onnx"
    MODEL_DIR: ClassVar[Path] = WORKING_DIR / "birefnet"
    MODEL_PATH: ClassVar[Path] = MODEL_DIR / MODEL_FILENAME

    def ensure_model(self):
        if self.MODEL_PATH.exists():
            return

        logger.warning("BiRefNet model not found. Downloading...")
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        try:
            response = httpx.get(self.MODEL_URL, timeout=30.0, follow_redirects=True)
            response.raise_for_status()

            with open(self.MODEL_PATH, "wb") as f:
                f.write(response.content)

            logger.info(
                "✅ BiRefNet model downloaded successfully to %s", self.MODEL_PATH
            )

        except Exception as e:
            logger.error("❌ Failed to download BiRefNet ONNX model: %s", e)
            raise ModelNotInstalled(
                "Could not download BiRefNet model. Check your internet connection."
            ) from e

    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        bg_color = kwargs.get("bg_color", "None")

        self.ensure_model()

        # Resize image to 512x512 for model input
        resized_img = img.convert("RGB").resize((512, 512))
        np_img = np.array(resized_img).astype(np.float32) / 255.0
        input_tensor = np.transpose(np_img, (2, 0, 1))[None, ...]  # (1, 3, 512, 512)

        try:
            ort_session = ort.InferenceSession(str(self.MODEL_PATH))
            outputs = ort_session.run(["output_image"], {"input_image": input_tensor})
            alpha_mask = outputs[0][0, 0]  # (512, 512)
        except Exception as e:
            logger.error("❌ BiRefNet ONNX inference failed: %s", e)
            raise RuntimeError("BiRefNet inference failed") from e

        # Resize alpha mask to original image size
        alpha_mask_resized = cv2.resize(
            alpha_mask, img.size, interpolation=cv2.INTER_LINEAR
        )
        alpha_uint8 = (alpha_mask_resized * 255).clip(0, 255).astype(np.uint8)

        rgba_array = np.dstack((np.array(img.convert("RGB")), alpha_uint8))
        final_img = Image.fromarray(rgba_array, "RGBA")

        logger.info("🦥 Removed background using BiRefNet")

        # Background color compositing if needed
        if bg_color not in ("none", "None", None):
            try:
                bg = Image.new("RGBA", final_img.size, bg_color)
                final_img = Image.alpha_composite(bg, final_img).convert("RGB")
                logger.info("🦥 Applied background color: %s", bg_color)
            except ValueError:
                logger.error("❌ Error applying background color.")
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
