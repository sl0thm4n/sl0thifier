from PIL import Image

from sl0thfier.tone import tone_correction


def test_tone_correction_returns_image(sample_image):
    img = Image.open(sample_image).convert("RGB")
    result = tone_correction(img)
    assert result.size == img.size
    assert result.mode == img.mode
