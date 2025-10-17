# Contributing to sl0thifier 🦥

First of all, thank you for your interest in contributing to **sl0thifier**!

This project aims to provide a clean, modular, and high-performance image preprocessing toolkit based on Python, ONNX, and Real-ESRGAN.

---

## 🧰 Development Setup

### Install `uv`

We use [`uv`](https://github.com/astral-sh/uv) to manage the virtual environment and dependencies.

If you don't have it installed yet:

```bash
curl -Ls https://astral.sh/uv/install.sh | bash
```

Check that it's installed:

```bash
uv --version
```

---

### Clone and Setup

```bash
git clone https://github.com/sl0thm4n/sl0thifier.git
cd sl0thifier

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

---

## 🧪 Testing & Linting

Before committing any code, please make sure all tests and style checks pass.

```bash
# Run all tests
pytest

# Check test coverage
pytest --cov=sl0thifier

# Format check
black . --check

# Linting
ruff check .
flake8
```

You can auto-format the codebase using:

```bash
black .
```

---

## 🚀 Versioning & Releases

We follow [Semantic Versioning (SemVer)](https://semver.org/) and automate it with [bump-my-version](https://github.com/callowayproject/bump-my-version).

To bump a version and tag it:

```bash
bump-my-version bump [major|minor|patch] --commit --tag
```

Then push:

```bash
git push && git push --tags
```

---

## 💡 Contribution Workflow

1. **Fork** the repo and create a feature branch:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure:
   - Tests are added/updated
   - Code passes format and lint checks
   - Code is modular and clearly documented

3. **Commit** your changes:

   ```bash
   git commit -m "feat: short description of your change"
   ```

4. **Push** to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request** against `main`.

---

## 📏 Code Style

We follow a minimal, consistent code style:

- `black` for formatting
- `ruff` and `flake8` for linting
- `pydantic v2` for type-validated config
- Prefer pure functions and clean module structure

---

## 🙏 Thank You!

We appreciate all kinds of contributions — bug fixes, new features, documentation, ideas, and more!

If you're not sure where to begin, feel free to open an [Issue](https://github.com/sl0thm4n/sl0thifier/issues) or start a [Discussion](https://github.com/sl0thm4n/sl0thifier/discussions).

Happy hacking! 🦥
