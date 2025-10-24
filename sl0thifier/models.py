import os
import cv2
import torch
import httpx
import zipfile
import shutil
import platform
import subprocess
import tempfile
import uuid
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Literal
from gfpgan import GFPGANer
import mediapipe as mp
import onnxruntime as ort

# Increase PIL decompression bomb limit for large upscaled images
Image.MAX_IMAGE_PIXELS = None  # Disable limit completely

from sl0thifier.logger import logger
from sl0thifier.exceptions import ModelNotInstalled

# =========================================
# 📂 PATHS
# =========================================
WORKING_DIR: Path = Path(__file__).resolve().parent.parent

# =========================================
# 🧩 BASE CLASS
# =========================================
class Sl0thifierBaseClass:
    """Common interface for all sl0thifier modules."""
    
    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        raise NotImplementedError("Each subclass must implement sl0thify().")

# =========================================
# 🧠 FACE REFOCUSER (GFPGAN)
# =========================================
class FaceRefocuser(Sl0thifierBaseClass):
    """Face refocus and restoration using GFPGAN v1.4."""
    
    MODEL_URLS = {
        "GFPGANv1.4": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    }
    MODEL_PATH = WORKING_DIR / "gfpgan" / "GFPGANv1.4.pth"

    def ensure_model(self):
        if self.MODEL_PATH.exists():
            return
        url = self.MODEL_URLS["GFPGANv1.4"]
        logger.warning("GFPGAN model not found. Downloading from: %s", url)
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.stream("GET", url, timeout=60.0, follow_redirects=True) as r:
                r.raise_for_status()
                with open(self.MODEL_PATH, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
        except Exception as e:
            logger.error("❌ Failed to download GFPGAN model: %s", e)
            raise ModelNotInstalled("Failed to download GFPGAN model.") from e
        if not self.MODEL_PATH.exists():
            raise ModelNotInstalled("GFPGAN model file missing after download.")
        logger.info("✅ GFPGAN model downloaded to %s", self.MODEL_PATH)

    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        self.ensure_model()
        logger.info("🧠 Running FaceRefocuser (GFPGAN v1.4)...")

        img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

        # Check if faces exist using MediaPipe
        mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        results = mp_face.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        if not results.detections:
            logger.warning("No face detected; skipping GFPGAN.")
            return img

        # Process entire image with GFPGAN (it will find and restore faces automatically)
        restorer = GFPGANer(
            model_path=str(self.MODEL_PATH),
            upscale=2,  # 2x upscale for better quality
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )

        try:
            # Let GFPGAN handle the entire image with maximum quality settings
            _, _, restored = restorer.enhance(
                img_cv,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=1.0  # Full strength restoration for best quality
            )
            logger.info("🦥 Successfully refocused %d face(s) with high quality", len(results.detections))
            torch.cuda.empty_cache()
            return Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error("❌ Error restoring faces: %s", e)
            logger.warning("Returning original image without face restoration")
            torch.cuda.empty_cache()
            return img

# =========================================
# 🧼 IMAGE UPSCALER (Real-ESRGAN NCNN-Vulkan)
# =========================================
class ImageUpscaler(Sl0thifierBaseClass):
    """Image upscaler using Real-ESRGAN NCNN-Vulkan executable."""

    @classmethod
    def ensure_model(cls) -> Path:
        system = platform.system()
        realesrgan_dir = WORKING_DIR / "realesrgan"

        if system == "Windows":
            exe_name = "realesrgan-ncnn-vulkan.exe"
            zip_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
        elif system == "Linux":
            exe_name = "realesrgan-ncnn-vulkan"
            zip_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-ubuntu.zip"
        elif system == "Darwin":
            exe_name = "realesrgan-ncnn-vulkan"
            zip_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip"
        else:
            raise RuntimeError("Unsupported OS: %s" % system)

        exe = realesrgan_dir / exe_name
        if exe.exists():
            return exe

        logger.warning("Real-ESRGAN binary not found. Downloading...")
        realesrgan_dir.mkdir(parents=True, exist_ok=True)
        zip_path = realesrgan_dir / "realesrgan.zip"

        try:
            with httpx.stream("GET", zip_url, timeout=60.0, follow_redirects=True) as response:
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
            logger.info("✅ Downloaded Real-ESRGAN package.")
        except Exception as e:
            logger.error("❌ Failed to download Real-ESRGAN: %s", e)
            raise ModelNotInstalled("Could not download Real-ESRGAN package.") from e

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
                try:
                    zip_path.unlink()
                except Exception:
                    pass

        if not exe.exists():
            raise ModelNotInstalled("Real-ESRGAN executable not found after install.")
        return exe

    @classmethod
    def list_models(cls) -> list[str]:
        """Return available Real-ESRGAN model names for UI dropdown."""
        model_dir = WORKING_DIR / "realesrgan" / "models"
        if not model_dir.exists():
            return ["realesrgan-x4plus", "realesrgan-x4plus-anime"]
        files = [f.stem for f in model_dir.glob("*.bin")]
        return sorted(files or ["realesrgan-x4plus", "realesrgan-x4plus-anime"])

    def _run_realesrgan(self, img: Image.Image, exe: Path, model_name: str, scale: int) -> Image.Image:
        """Call realesrgan-ncnn-vulkan via subprocess and return PIL Image."""
        tmpdir = Path(tempfile.gettempdir()) / ("sl0thifier_" + uuid.uuid4().hex)
        tmpdir.mkdir(parents=True, exist_ok=True)
        in_path = tmpdir / "in.png"
        out_path = tmpdir / "out.png"
        try:
            img.convert("RGB").save(in_path)

            cmd = [
                str(exe),
                "-i", str(in_path),
                "-o", str(out_path),
                "-s", str(scale),
                "-n", model_name,
            ]
            logger.info("🧼 Real-ESRGAN cmd: %s", " ".join(['"%s"' % c if " " in c else c for c in cmd]))

            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                text=True,
            )
            if completed.returncode != 0 or not out_path.exists():
                logger.error("❌ Real-ESRGAN failed (code=%s): %s", completed.returncode, completed.stdout)
                raise RuntimeError("Real-ESRGAN execution failed.")

            up = Image.open(out_path).convert("RGB")
            return up
        finally:
            try:
                if in_path.exists(): in_path.unlink()
                if out_path.exists(): out_path.unlink()
                tmpdir.rmdir()
            except Exception:
                pass

    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        exe = self.ensure_model()
        model_name = kwargs.get("model_name", "realesrgan-x4plus")
        scale = int(kwargs.get("scale", 4))
        logger.info("🧼 Upscaling using %s (×%d)...", model_name, scale)
        return self._run_realesrgan(img, exe, model_name, scale)

# =========================================
# 🌈 IMAGE ENHANCER (CLAHE)
# =========================================
class ImageEnhancer(Sl0thifierBaseClass):
    """Apply CLAHE and color enhancement."""
    
    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        clip_limit = kwargs.get("clip_limit", 1.0)
        tile_size = kwargs.get("tile_size", 4)
        logger.info("✨ Applying CLAHE (clip=%.2f, tile=%d)", clip_limit, tile_size)
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(img_cv)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l = clahe.apply(l)
        enhanced = cv2.merge((l, a, b))
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced_rgb)

# =========================================
# 🎨 BACKGROUND REMOVER (BiRefNet ONNX)
# =========================================
class ImageBackgroundRemover(Sl0thifierBaseClass):
    """Remove image background using BiRefNet ONNX model."""
    
    MODEL_URL = "https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-resolution_512x512-fp16-epoch_216.onnx"
    MODEL_PATH = WORKING_DIR / "birefnet" / "birefnet.onnx"

    @classmethod
    def ensure_model(cls):
        if cls.MODEL_PATH.exists():
            return
        logger.warning("BiRefNet model not found. Downloading...")
        cls.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = httpx.get(cls.MODEL_URL, timeout=60.0, follow_redirects=True)
            response.raise_for_status()
            with open(cls.MODEL_PATH, "wb") as f:
                f.write(response.content)
            logger.info("✅ BiRefNet model downloaded to %s", cls.MODEL_PATH)
        except Exception as e:
            logger.error("❌ Failed to download BiRefNet model: %s", e)
            raise ModelNotInstalled("Could not download BiRefNet model.") from e

    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        self.ensure_model()
        bg_color = kwargs.get("bg_color", "none")
        
        logger.info("🎨 Removing background using BiRefNet...")
        
        # Resize for model input (512x512)
        original_size = img.size
        resized_img = img.convert("RGB").resize((512, 512))
        np_img = np.array(resized_img).astype(np.float32) / 255.0
        input_tensor = np.transpose(np_img, (2, 0, 1))[None, ...]

        try:
            ort_session = ort.InferenceSession(str(self.MODEL_PATH))
            outputs = ort_session.run(["output_image"], {"input_image": input_tensor})
            alpha_mask = outputs[0][0, 0]
        except Exception as e:
            logger.error("❌ BiRefNet inference failed: %s", e)
            raise RuntimeError("BiRefNet inference failed") from e

        # Resize mask back to original size
        alpha_mask_resized = cv2.resize(alpha_mask, original_size, interpolation=cv2.INTER_LINEAR)
        alpha_uint8 = (alpha_mask_resized * 255).clip(0, 255).astype(np.uint8)

        # Create RGBA image
        rgba_array = np.dstack((np.array(img.convert("RGB")), alpha_uint8))
        result = Image.fromarray(rgba_array, "RGBA")

        # Apply background color if specified
        if bg_color not in ("none", "None", None):
            bg = Image.new("RGBA", result.size, bg_color)
            result = Image.alpha_composite(bg, result).convert("RGB")
            logger.info("🦥 Applied background color: %s", bg_color)

        return result

# =========================================
# 🦥 KING SL0TH (PIPELINE)
# =========================================
class KingSl0th(Sl0thifierBaseClass):
    """Composite model combining refocuser, upscaler, enhancer, and optional bg remover."""
    
    def __init__(self):
        self.refocuser = FaceRefocuser()
        self.upscaler = ImageUpscaler()
        self.enhancer = ImageEnhancer()
        self.bg_remover = ImageBackgroundRemover()

    def sl0thify(self, img: Image.Image, **kwargs) -> Image.Image:
        output_width = kwargs.get("output_width", 512)
        output_height = kwargs.get("output_height", 512)
        remove_bg = kwargs.get("remove_bg", False)
        
        logger.info("🧩 [KingSl0th] Starting high-quality pipeline...")
        
        # Step 1: Face refocus & restoration (2x upscale + full restoration)
        img = self.refocuser.sl0thify(img, **kwargs)
        
        # Step 2: Upscale (4x upscale with Real-ESRGAN)
        img = self.upscaler.sl0thify(img, **kwargs)
        
        # Step 3: Enhance contrast and color (CLAHE)
        img = self.enhancer.sl0thify(img, **kwargs)
        
        # Step 4: Remove background (optional)
        if remove_bg:
            img = self.bg_remover.sl0thify(img, **kwargs)
        
        # Step 5: Resize to target dimensions with high-quality resampling
        if output_width and output_height:
            img = img.resize((output_width, output_height), Image.Resampling.LANCZOS)
            logger.info("🦥 Resized to %dx%d", output_width, output_height)
        
        logger.info("✅ [KingSl0th] High-quality pipeline complete.")
        return img