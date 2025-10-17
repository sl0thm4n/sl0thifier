from PIL import Image


def resize_image(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return img.resize(size, Image.Resampling.LANCZOS)

def make_thumbnail(img: Image.Image, max_size: int = 128) -> Image.Image:
    img = img.copy()
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img
