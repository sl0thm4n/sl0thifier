import subprocess
from pathlib import Path

import cv2
import pytest
from PIL import Image

import sl0thifier.models as models
from sl0thifier.exceptions import ModelNotInstalled
from sl0thifier.models import ImageBackgroundRemover, ImageEnhancer, ImageUpscaler


def create_dummy_image(size=(128, 128), color=(255, 0, 0)):
    return Image.new("RGB", size, color)


class TestImageEnhancer:
    def test_sl0thify_success(self):
        img = create_dummy_image()
        enhancer = ImageEnhancer()
        out = enhancer.sl0thify(img)
        assert isinstance(out, Image.Image)
        assert out.size == img.size

    def test_sl0thify_opencv_error(self, monkeypatch):
        img = create_dummy_image()
        enhancer = ImageEnhancer()

        def raise_cv2_error(*args, **kwargs):
            raise cv2.error("Mock OpenCV failure")

        monkeypatch.setattr(cv2, "cvtColor", raise_cv2_error)
        with pytest.raises(RuntimeError):
            enhancer.sl0thify(img)


class TestImageBackgroundRemover:
    def test_sl0thify_model_not_installed(self, monkeypatch):
        img = create_dummy_image()
        remover = ImageBackgroundRemover()

        monkeypatch.setattr(Path, "exists", lambda self: False)
        with pytest.raises(ModelNotInstalled):
            remover.sl0thify(img)

    def test_sl0thify_inference_error(self, monkeypatch):
        img = create_dummy_image()
        remover = ImageBackgroundRemover()

        monkeypatch.setattr(Path, "exists", lambda self: True)

        class MockSession:
            def run(self, *args, **kwargs):
                raise RuntimeError("Mock ONNX inference failure")

        monkeypatch.setattr(models.ort, "InferenceSession", lambda path: MockSession())

        with pytest.raises(RuntimeError):
            remover.sl0thify(img)


class TestImageUpscaler:
    def test_sl0thify_model_not_installed(self, monkeypatch):
        img = create_dummy_image()
        upscaler = ImageUpscaler()

        monkeypatch.setattr(Path, "exists", lambda self: False)
        with pytest.raises(ModelNotInstalled):
            upscaler.sl0thify(img)

    def test_sl0thify_subprocess_failure(self, monkeypatch):
        img = create_dummy_image()
        upscaler = ImageUpscaler()

        monkeypatch.setattr(Path, "exists", lambda self: True)

        def raise_subprocess_error(*args, **kwargs):
            raise subprocess.CalledProcessError(returncode=1, cmd="mock")

        monkeypatch.setattr(subprocess, "run", raise_subprocess_error)

        with pytest.raises(RuntimeError):
            upscaler.sl0thify(img)
