import argparse
import os
from pathlib import Path
from typing import Literal

from PIL import Image
from sl0thifier.models import KingSl0th
from sl0thifier.logger import logger

try:
    resample = Image.Resampling.LANCZOS
except AttributeError:
    resample = getattr(Image, "LANCZOS")


def process_image(
    img_path: Path,
    output_dir: Path,
    model_name: str,
    clip_limit: float = 1.0,
    tile_size: int = 4,
    target_size: tuple[int, int] = (512, 512),
    remove_bg: bool = False,
    bg_color: Literal["none", "white", "black", "green"] = "none",
    shared_model: KingSl0th | None = None,
):
    """Process a single image using a shared KingSl0th instance."""
    try:
        img = Image.open(img_path).convert("RGB")
        ks = shared_model or KingSl0th()

        logger.info("🦥 Sl0thifying %s...", img_path.name)
        final = ks.sl0thify(
            img,
            output_width=target_size[0],
            output_height=target_size[1],
            model_name=model_name,
            clip_limit=clip_limit,
            tile_size=tile_size,
            remove_bg=remove_bg,
            bg_color=bg_color,
        )

        output_path = output_dir / img_path.name
        final.save(output_path)
        logger.info("✅ Processed %s -> %s", img_path.name, output_path)
    except Exception as e:
        logger.error("❌ Failed to process %s: %s", img_path.name, str(e))


def collect_images(images_arg: str) -> list[Path]:
    """Collect image paths (single file or folder)."""
    p = Path(images_arg)
    if not p.exists():
        raise FileNotFoundError("Path '%s' not found" % images_arg)
    if p.is_file():
        return [p]
    return list(p.rglob("*.jpg")) + list(p.rglob("*.jpeg")) + list(p.rglob("*.png"))


def main():
    parser = argparse.ArgumentParser(
        description="🦥 Sl0thify: batch refocus + upscale + enhance + resize"
    )
    parser.add_argument("--images", required=True, help="Path to image or folder")
    parser.add_argument(
        "--model-name", default="realesrgan-x4plus", help="Upscaler model name"
    )
    parser.add_argument(
        "--clip-limit", type=float, default=1.0, help="CLAHE clip limit (brightness contrast)"
    )
    parser.add_argument(
        "--tile-size", type=int, default=4, help="CLAHE tile size for local contrast"
    )
    parser.add_argument("--width", type=int, required=True, help="Final image width")
    parser.add_argument("--height", type=int, required=True, help="Final image height")
    parser.add_argument("--output-dir", help="Directory to save processed images")
    parser.add_argument(
        "--remove-bg", action="store_true", help="Remove background after enhancement"
    )
    parser.add_argument(
        "--bg-color",
        type=str,
        choices=["none", "white", "black", "green"],
        default="none",
        help="Background color to use if removing background",
    )

    args = parser.parse_args()

    img_paths = collect_images(args.images)
    target_size = (args.width, args.height)

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        input_path = Path(args.images)
        if input_path.is_dir():
            output_dir = input_path / "sl0thified"
        else:
            output_dir = input_path.parent / "sl0thified"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "🧪 Processing %d image(s) with model '%s' to size %s...",
        len(img_paths),
        args.model_name,
        target_size,
    )
    logger.info("📁 Output directory: %s", output_dir)

    # 🧠 Shared model instance
    ks = KingSl0th()
    logger.info("🔧 Model initialized once (shared across all images).")

    for img_path in img_paths:
        process_image(
            img_path,
            output_dir,
            args.model_name,
            args.clip_limit,
            args.tile_size,
            target_size,
            args.remove_bg,
            args.bg_color,
            shared_model=ks,
        )


if __name__ == "__main__":
    main()
