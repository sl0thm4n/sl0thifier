import argparse
import os
from concurrent.futures import ProcessPoolExecutor
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
):
    try:
        img = Image.open(img_path).convert("RGB")

        ks = KingSl0th(model_name=model_name)
        logger.info(f"🦥 Sl0thifying {img_path.name}...")
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

        # Save
        output_path = output_dir / img_path.name
        final.save(output_path)
        print(f"✅ Processed {img_path.name} -> {output_path}")
    except Exception as e:
        print(f"❌ Failed to process {img_path.name}: {e}")


def collect_images(images_arg: str) -> list[Path]:
    p = Path(images_arg)
    if not p.exists():
        raise FileNotFoundError(f"Path '{images_arg}' not found")
    if p.is_file():
        return [p]
    return list(p.rglob("*.jpg")) + list(p.rglob("*.jpeg")) + list(p.rglob("*.png"))


def main():
    parser = argparse.ArgumentParser(
        description="🦥 Sl0thify: batch upscale + enhance + resize"
    )
    parser.add_argument("--images", required=True, help="Path to image or folder")
    parser.add_argument(
        "--model-name", default="realesrgan-x4plus", help="Upscaler model"
    )
    parser.add_argument(
        "--clip-limit", type=float, default=1.0, help="CLAHE clip limit"
    )
    parser.add_argument("--tile-size", type=int, default=4, help="CLAHE tile size")
    parser.add_argument("--width", type=int, required=True, help="Final image width")
    parser.add_argument("--height", type=int, required=True, help="Final image height")
    parser.add_argument("--output-dir", help="Directory to save processed images")
    parser.add_argument(
        "--remove-bg", type=bool, default=False, help="Remove background"
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
    size = (args.width, args.height)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path.cwd() / f"sl0thified_{args.width}-{args.height}"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"🧪 Processing {len(img_paths)} image(s) with model '{args.model_name}' to size {size}..."
    )
    print(f"📁 Output directory: {output_dir}")

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for img_path in img_paths:
            executor.submit(
                process_image,
                img_path,
                output_dir,
                args.model_name,
                clip_limit=clip_limit,
                tile_size=tile_size,
                size=size,
                remove_bg=args.remove_bg,
                bg_color=args.bg_color,
            )


if __name__ == "__main__":
    main()
