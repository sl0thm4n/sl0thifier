# sl0thifier — AI-Powered Image Refinement

**sl0thifier** is a high-performance image preprocessing tool designed to deliver professional-quality results using state-of-the-art AI models. Slow, precise, and deadly clean.

---

## 🖼️ Example Output

<p align="center">
  <table>
    <tr>
      <td align="center">
        <img src="./examples/input.jpg" alt="Input image" width="220"/><br/>
        <sub><strong>Original Input</strong></sub>
      </td>
      <td align="center">
        <img src="./examples/input_sl0thified.png" alt="Sl0thified output" width="300"/><br/>
        <sub><strong>Sl0thified, 1024x1024</strong></sub>
      </td>
    </tr>
  </table>
</p>

---

## 🦥 Image Processing Pipeline

Each image passes through the following high-quality enhancement steps:

1. **🧠 Face Refocus** — Face restoration and enhancement using GFPGAN v1.4 (2x upscale, full restoration)
2. **🧼 Upscaling** — 4x super-resolution with Real-ESRGAN NCNN-Vulkan
3. **✨ Enhancement** — Contrast Limited Adaptive Histogram Equalization (CLAHE) for color and contrast
4. **🎨 Background Removal** (Optional) — Using BiRefNet ONNX model
5. **📐 Resize** — Final resize to target dimensions with high-quality LANCZOS resampling

---

## 📋 Requirements

- **Python**: 3.10 or later
- **GPU**: CUDA-compatible GPU recommended (NVIDIA)
- **CUDA Toolkit**: 11.2 or later (for GPU acceleration)
- **Vulkan**: Required for Real-ESRGAN (Windows/Linux)

---

## 📦 Installation

### Windows

#### 1. Install Python 3.10+
Download from [python.org](https://www.python.org/downloads/)

#### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

#### 3. Install dependencies
```bash
pip install -e .
```

#### 4. Verify installation
```bash
sl0thify --help
```

**Note**: Real-ESRGAN executable will be downloaded automatically on first run.

---

### Linux (Ubuntu/Debian)

#### 1. Install Python 3.10+ and system dependencies
```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
sudo apt install libvulkan1 vulkan-utils  # For Real-ESRGAN
```

#### 2. Create virtual environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install -e .
```

#### 4. Verify installation
```bash
sl0thify --help
```

**Note**: Real-ESRGAN binary will be downloaded automatically on first run.

---

### macOS

#### 1. Install Python 3.10+ via Homebrew
```bash
brew install python@3.10
```

#### 2. Create virtual environment
```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install -e .
```

#### 4. Verify installation
```bash
sl0thify --help
```

**Note**: macOS support is experimental. Real-ESRGAN binary will be downloaded automatically on first run.

---

## ⚙️ CLI Usage

### Basic Usage

```bash
sl0thify --images=PATH --width=WIDTH --height=HEIGHT
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--images` | ✅ Yes | - | Path to image file or folder |
| `--width` | ✅ Yes | - | Output image width |
| `--height` | ✅ Yes | - | Output image height |
| `--model-name` | ❌ No | `realesrgan-x4plus` | Real-ESRGAN model name |
| `--clip-limit` | ❌ No | `1.0` | CLAHE clip limit (contrast) |
| `--tile-size` | ❌ No | `4` | CLAHE tile size |
| `--output-dir` | ❌ No | `./sl0thified` | Output directory |
| `--remove-bg` | ❌ No | `False` | Remove background |
| `--bg-color` | ❌ No | `none` | Background color (`none`, `white`, `black`, `green`) |

### Examples

**Process a single image:**
```bash
sl0thify --images=./photo.jpg --width=1024 --height=1024
```

**Process entire folder:**
```bash
sl0thify --images=./photos --width=512 --height=512
```

**Custom model and output directory:**
```bash
sl0thify --images=./cats --model-name=realesrgan-x4plus-anime --width=768 --height=768 --output-dir=./output
```

**With background removal:**
```bash
sl0thify --images=./portraits --width=1024 --height=1024 --remove-bg --bg-color=white
```

---

## 🖼️ GUI Usage

A simple GUI is available via `main.py`:

```bash
python main.py
```

### GUI Features:
- 📂 Drag & Drop support for files and folders
- ⚙️ Adjustable parameters (width, height, model)
- 🎨 Background removal options
- ⏳ Progress bar display
- 📁 Output saved with `_sl0thified` suffix

---

## 🔧 Advanced Configuration

### Available Real-ESRGAN Models

- `realesrgan-x4plus` (default) — Best for general photos
- `realesrgan-x4plus-anime` — Optimized for anime/illustration

Models are downloaded automatically to `./realesrgan/models/` on first use.

### CLAHE Parameters

**Clip Limit** (`--clip-limit`):
- Range: 0.1 - 5.0
- Lower = natural, Higher = more contrast
- Default: 1.0

**Tile Size** (`--tile-size`):
- Range: 2 - 16
- Smaller = local adjustment, Larger = global adjustment
- Default: 4

---

## 🗂️ Project Structure

```
sl0thifier/
├── __init__.py
├── models.py           # Core AI models
├── logger.py           # Logging utilities
├── exceptions.py       # Custom exceptions
sl0thify.py             # CLI entrypoint
main.py                 # GUI entrypoint
tests/                  # Test suite
├── __init__.py
├── conftest.py
└── test_models.py
gfpgan/                 # GFPGAN model files (auto-downloaded)
└── GFPGANv1.4.pth
realesrgan/             # Real-ESRGAN binaries (auto-downloaded)
├── realesrgan-ncnn-vulkan.exe (Windows)
└── models/
    └── realesrgan-x4plus.bin
birefnet/               # BiRefNet ONNX model (auto-downloaded)
└── birefnet.onnx
```

---

## ⚠️ OS Compatibility

| Feature | Windows | Linux | macOS |
|---------|---------|-------|-------|
| Face Refocus (GFPGAN) | ✅ Full | ✅ Full | ✅ Full |
| Upscaling (Real-ESRGAN) | ✅ Full | ✅ Full | ⚠️ Experimental |
| Enhancement (CLAHE) | ✅ Full | ✅ Full | ✅ Full |
| Background Removal | ✅ Full | ✅ Full | ✅ Full |
| GUI (Tkinter) | ✅ Full | ⚠️ Limited | ⚠️ Limited |

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=sl0thifier

# Code style checks
black . --check
ruff check .
```

---

## 🚀 Versioning & Releases

This project uses [bump-my-version](https://github.com/callowayproject/bump-my-version) for automated semantic versioning.

```bash
# Bump version (patch: 0.1.0 → 0.1.1)
bump-my-version bump patch --commit --tag

# Bump minor version (0.1.0 → 0.2.0)
bump-my-version bump minor --commit --tag

# Push with tags
git push && git push --tags
```

**Current version**: `0.1.0`

---

## 🤝 Contributing

Pull requests and contributions are welcome!

**Before submitting:**
- Format code with `black`
- Pass all checks: `ruff`, `pytest`
- Add relevant tests
- Follow conventional commit messages

---

## 🧠 Tech Stack

- **Python** 3.10+
- **GFPGAN** — Face restoration
- **Real-ESRGAN** — Super-resolution
- **BiRefNet** — Background removal
- **ONNX Runtime** — GPU acceleration
- **MediaPipe** — Face detection
- **OpenCV** — Image processing
- **NumPy** — Numerical operations
- **Pillow** — Image I/O

---

## 📜 License

MIT License

Copyright (c) 2025 [sl0thm4n](https://github.com/sl0thm4n)

---

## 🙏 Acknowledgments

- [GFPGAN](https://github.com/TencentARC/GFPGAN) by Tencent ARC Lab
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) by ZhengPeng7
- [MediaPipe](https://github.com/google/mediapipe) by Google

---

## 📧 Contact

For issues, questions, or contributions, please open an issue on [GitHub](https://github.com/sl0thm4n/sl0thifier).

---

**Made with 🦥 by sl0thm4n**