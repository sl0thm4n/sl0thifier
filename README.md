# sl0thifier 🦥 – High-Performance Image Preprocessor (Real-ESRGAN + ONNX)

**sl0thifier** is a Python-based high-resolution image preprocessing tool designed to deliver efficient and high-quality results using ONNX inference and Real-ESRGAN super-resolution models.

---

## ✨ Features

- ✅ Fast inference using ONNX runtime  
- ✅ Super-resolution enhancement via Real-ESRGAN  
- ✅ Modular image processing (`imageio`, `resize`, `tone`, etc.)  
- ✅ Type-safe configuration using Pydantic v2  
- ✅ Clean, test-driven codebase with full coverage  
- ✅ Semantic versioning and automated release process  
- ✅ 🧠 One-command smart CLI with parallel processing  

---

## 📊 Requirements

- Python 3.10+
- CUDA Toolkit (11.2 or later recommended) ✨
- cuDNN (matching CUDA version)
- Vulkan-compatible GPU for RealESRGAN NCNN executable (Windows only)

### 🔗 Required model files
- [`realesrgan/realesrgan-ncnn-vulkan.exe`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip) – Windows-only NCNN executable
- `realesrgan/models/realesrgan-x4plus.bin`
- [`birefnet/birefnet.onnx`](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-resolution_512x512-fp16-epoch_216.onnx) – 918,483KB version with weights included

> If using background removal or super-resolution, ensure your system supports CUDA acceleration for best performance.

---

## 📊 Requirements

### 1. Create environment (Python 3.10+)

```bash
uv venv .venv --python=3.10 --seed
source .venv/bin/activate

### 🔗 Required model files
- [`realesrgan/realesrgan-ncnn-vulkan.exe`](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip) – Windows-only NCNN executable
- `realesrgan/models/realesrgan-x4plus.bin`
- `realesrgan/models/realesrgan-x4plus.param`
- [`birefnet/birefnet.onnx`](https://github.com/ZhengPeng7/BiRefNet/releases/download/v1/BiRefNet-general-resolution_512x512-fp16-epoch_216.onnx) – 918,483KB version with weights included

> After installation, the `sl0thify` CLI will be available in your environment.

### 2. Download model files manually

Place the following files in the project directory structure:
- `realesrgan/realesrgan-ncnn-vulkan.exe`
- `realesrgan/models/realesrgan-x4plus.bin`
- `birefnet/birefnet.onnx`

See [Required model files](#-required-model-files) for direct links.

### 3. (Optional) Run the GUI

```bash
python main.py
```

---

## ⚙️ CLI Usage

```bash
sl0thify --images=PATH [--model-name=MODEL] --width=WIDTH --height=HEIGHT [--output-dir=OUTDIR]
```

### ✅ Example

```bash
sl0thify --images=./cats --width=512 --height=512
sl0thify --images=./cats/cat1.jpg --model-name=realesrnet-x4plus --width=256 --height=256 --output-dir=./out
```

---

## 🖼 GUI Usage (Tkinter)

A simple GUI is available via `main.py` using `tkinter` and `tkinterdnd2`. Features include:

- 💾 Drag & Drop support for files and folders
- ⚖️ Options:
  - Remove Background (optional)
  - New Background Color (None / White / Black / Green)
- ⏳ Progress bar display
- 📂 Output saved to same directory with `_sl0thified_WIDTHxHEIGHT` suffix
- ✨ Automatic processing upon file drop

---

## 🦠 Image Processing Pipeline

Each image passes through the following steps:

1. 🔹 **Upscaling** — 4x upscaling with denoising using RealESRGAN (via `ImageUpscaler`)
2. 🔹 **Enhancement** — Contrast Limited Adaptive Histogram Equalization (CLAHE) via OpenCV (`ImageEnhancer`)
3. 🔹 **Background Removal** (Optional) — Using BiRefNet ONNX model (`ImageBackgroundRemover`)
4. 🔹 **Resize** — Final resize to user-defined dimensions

---

## 💥 Recent Bugfixes & Improvements

- Fixed missing `img` argument in `KingSl0th.sl0thify()` call
- Corrected faulty `ImageUpscaler` initialization
- Replaced faulty 470MB `BiRefNet.onnx` with 930MB full model
- UI simplified: removed thumbnails, added plain text feedback
- Background removal options UI cleaned up
- Drop zone and progress bar layout improved

---

## ⚠️ OS Compatibility Matrix

| Feature               | Windows | Linux          | macOS          |
|----------------------|---------|----------------|----------------|
| Tkinter + DnD        | ✅ Full  | ⚠️ Limited       | ⚠️ Limited       |
| RealESRGAN Executable| ✅ Yes   | ❌ Needs binary | ❌ Not available |
| ONNX Runtime         | ✅ Yes   | ✅ Yes         | ✅ Yes         |
| Full Pipeline        | ✅ Works| ⚠️ Minor fixes   | ⚠️ Minor fixes   |

---

## 🔪 Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=sl0thifier

# Code style checks
black . --check
ruff check .
flake8
```

---

## 🚀 Versioning & Releases

This project uses [bump-my-version](https://github.com/callowayproject/bump-my-version) for automated semantic versioning.

```bash
# Example: Bump patch version (e.g., 0.1.0 → 0.1.1)
bump-my-version bump patch --commit --tag

git push && git push --tags
```

Current version: `0.1.0`  
Author: [sl0thm4n](https://github.com/sl0thm4n)

---

## 📁 Project Structure
```
sl0thifier/
├── __init__.py
├── assets/
│   └── sl0thm4n.ico
├── core.py
├── exceptions.py
├── logger.py
├── models.py
├── preprocess.py
├── utils.py
sl0thify.py         # CLI entrypoint
main.py             # GUI entrypoint
tests/
├── __init__.py
├── conftest.py
└── test_models.py

birefnet/
└── birefnet.onnx

realesrgan/
├── realesrgan-ncnn-vulkan.exe
└── models/
    └── realesrgan-x4plus.bin
```

---

## 🙌 Contribution Guide

Pull requests and contributions are welcome!

Before submitting a PR:

- Format code with `black`
- Pass all checks: `ruff`, `flake8`, `pytest`
- Add relevant tests
- Follow consistent commit messages

*A full CONTRIBUTING guide will be added soon.*

---

## 🧠 Tech Stack

- Python 3.10+  
- ONNX Runtime  
- Real-ESRGAN  
- NumPy, OpenCV, Pillow  
- Pydantic v2  
- Pytest, Coverage, Ruff, Black  
- `uv` for dependency and environment management  

---

## 📜 License

MIT License  
Copyright (c) 2025 [sl0thm4n](https://github.com/sl0thm4n)
