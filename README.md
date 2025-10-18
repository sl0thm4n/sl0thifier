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

## 📦 Installation

```bash
# Python 3.10+ recommended
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install --editable .
```

> `sl0thify` command will now be available globally within the environment.

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

### 💡 How it works

Each image passes through the following pipeline:

1. ✅ 4x Upscale (Real-ESRGAN)
2. ✅ Denoise (inherent to Real-ESRGAN)
3. ✅ CLAHE Enhancement
4. ✅ Final resize to target `--width` × `--height`

Multiple images are processed **in parallel** using all available CPU cores.

---

## 🧪 Running Tests

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
├── config/         # Pydantic configuration models
├── core/
│   ├── imageio.py  # Image loading and saving
│   ├── resize.py   # Image resizing operations
│   └── tone.py     # Color/tone adjustments
├── models/
│   ├── birefnet.py     # ONNX model handler
│   ├── realesrgan.py   # Real-ESRGAN interface
├── utils/          # Logging, helpers, etc.
sl0thify.py         # CLI entrypoint
tests/              # pytest test suite
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