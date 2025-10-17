from PIL import Image

from sl0thfier.inference import birefnet_remove_bg


def test_birefnet_handles_missing_session(sample_image):
    img = Image.open(sample_image).convert("RGBA")
    result = birefnet_remove_bg(img, "transparent")
    assert result.mode == "RGBA"
    assert result.size == img.size
