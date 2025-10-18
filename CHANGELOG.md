# Changelog

All notable changes to this project will be documented in this file.

---

## [0.1.0] - 2025-10-18

### Added

* Initial CLI pipeline: Upscaling, CLAHE enhancement, resizing
* Real-ESRGAN integration for 4x super-resolution
* BiRefNet ONNX model for background removal
* Type-safe configuration using Pydantic v2
* Parallel image processing via CLI

### GUI

* Tkinter-based GUI with drag & drop support
* Optional background removal and color selection
* Progress bar and automatic processing on drop

### Fixed

* CLI errors with missing `img` argument in `KingSl0th`
* `ImageUpscaler` misinitialization issue
* BiRefNet 470MB model crash (replaced with 930MB version)

### Changed

* GUI simplified: thumbnails removed, plain-text UI added
* Background removal UI alignment improved
* Progress bar and drop zone layout adjusted

### Compatibility

* Full support for Windows
* Partial support for Linux/macOS (tkinter DnD and RealESRGAN issues)
