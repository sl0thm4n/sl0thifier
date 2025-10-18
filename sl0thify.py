import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from sl0thifier.models import ImageEnhancer, ImageUpscaler

try:
    resample = Image.Resampling.LANCZOS
except AttributeError:
    resample = getattr(Image, "LANCZOS")


def process_image(
    img_path: Path, output_dir: Path, model_name: str, target_size: tuple[int, int]
):
    try:
        img = Image.open(img_path).convert("RGB")

        # Step 1: Upscale
        upscaled = ImageUpscaler().sl0thify(img, model_name=model_name, scale=4)

        # Step 2: Enhance
        enhanced = ImageEnhancer().sl0thify(upscaled)

        # Step 3: Resize to final size
        final = enhanced.resize(target_size, resample)

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
    parser.add_argument("--width", type=int, required=True, help="Final image width")
    parser.add_argument("--height", type=int, required=True, help="Final image height")
    parser.add_argument("--output-dir", help="Directory to save processed images")

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
            executor.submit(process_image, img_path, output_dir, args.model_name, size)


if __name__ == "__main__":
    main()
