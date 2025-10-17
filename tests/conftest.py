import pytest
from PIL import Image


@pytest.fixture(scope="session")
def sample_image(tmp_path_factory):
    """Creates a small RGB test image."""
    tmp_dir = tmp_path_factory.mktemp("data")
    img_path = tmp_dir / "sample.png"
    img = Image.new("RGB", (128, 128), color=(128, 128, 128))
    img.save(img_path)
    return str(img_path)


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    """Creates an output directory for tests."""
    return tmp_path_factory.mktemp("out")
