# TinyTorch Installation Guide

Complete installation guide for TinyTorch using modern Python package management with `uv`.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Recommended)](#quick-start-recommended)
3. [Installation Methods](#installation-methods)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Development Setup](#development-setup)

---

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher (3.10+ recommended)
- **Operating System**: macOS, Linux, or Windows (WSL2 recommended for Windows)
- **Disk Space**: ~2GB for full installation with dependencies


## Quick Start (Recommended)

The fastest way to get started with TinyTorch using `uv`:

```bash
# 1. Clone the repository
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch

# 2. Run the setup script (installs everything automatically)
./setup-uv.sh

# 3. Activate the virtual environment
source activate.sh

# 4. Verify installation
tito --version

# 5. Run system diagnostics
tito system doctor

# 6. Start your learning journey!
tito module start 01
```

That's it! You're ready to build ML systems from scratch.

---


## Verification

After installation, verify everything is working:

### 1. Check `tito` Command

```bash
# Should show the TinyTorch CLI help
tito

# Check version
tito --version

# Run system diagnostics
tito system doctor
```

### 2. Test Python Imports

```bash
python -c "import tinytorch; print(tinytorch.__version__)"
python -c "from tito.main import main; print('tito module loaded')"
```

### 3. Run Basic Tests

```bash
# Run a simple test to verify installation
pytest tests/01_tensor/test_tensor_core.py -v
```

### 4. Open Your First Module

```bash
# Start Module 01 in Jupyter
tito module start 01
```

---

## Troubleshooting

### Issue: `tito: command not found`

**Solution 1: Activate virtual environment**
```bash
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

**Solution 2: Reinstall in editable mode**
```bash
uv pip install -e .
# OR
pip install -e .
```

**Solution 3: Add to PATH manually**
```bash
export PATH="$PATH:$(pwd)/.venv/bin"
```

### Issue: Import errors for `nbdev` or `jupytext`

**Solution:**
```bash
# Install development dependencies
uv sync --extra dev

# OR with pip
pip install -e ".[dev]"
```

### Issue: Apple Silicon (M1/M2/M3) Architecture Issues

**Solution:**
```bash
# Re-run the setup script
rm -rf .venv
./setup-uv.sh
```

### Issue: Permission errors during export

**Solution:**
```bash
# Fix file permissions
chmod -R u+w tinytorch/

# Re-run export
tito module complete 01
```

### Issue: `ModuleNotFoundError` for tinytorch

**Solution:**
```bash
# Reinstall in development mode
uv pip install -e .

# Verify
python -c "import tinytorch; print('Success!')"
```

---

## Development Setup

### For Contributors and Advanced Users

```bash
# 1. Clone and navigate
git clone https://github.com/mlsysbook/TinyTorch.git
cd TinyTorch

# 2. Install with all development tools
uv sync --extra all

# 3. Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install

# 4. Verify development environment
tito system doctor

# 5. Run all tests
pytest tests/ -v

# 6. Build documentation
tito book build
```

### Development Workflow

```bash
# Work on a module
tito module start 01

# Make changes in modules/01_tensor/tensor.py

# Test your changes
python modules/01_tensor/tensor.py

# Export to package
tito module complete 01

# Run integration tests
pytest tests/01_tensor/ -v
```

### Optional Tools

```bash
# Code formatting
uv pip install black ruff

# Type checking
uv pip install mypy

# Documentation
uv pip install sphinx sphinx-rtd-theme
```

---

## Environment Variables

Optional environment variables for customization:

```bash
# Set custom cache directory
export TINYTORCH_CACHE_DIR=~/.cache/tinytorch

# Enable debug logging
export TINYTORCH_LOG_LEVEL=DEBUG

# Disable color output
export TINYTORCH_NO_COLOR=1
```

Add to your shell profile (~/.bashrc, ~/.zshrc):

```bash
# TinyTorch configuration
export TINYTORCH_CACHE_DIR=~/.cache/tinytorch
alias tt='tito'  # Shortcut for tito command
```

---

## Next Steps

After successful installation:

1. **Read the Quick Start Guide**: `docs/STUDENT_QUICKSTART.md`
2. **Start Module 01**: `tito module start 01`
3. **Join the Community**: See `docs/README.md` for community resources
4. **Explore Milestones**: `tito milestone list`
5. **Track Your Progress**: `tito checkpoint status`

---

## Getting Help

- **CLI Help**: Run `tito --help` or `tito <command> --help`
- **System Diagnostics**: `tito system doctor`
- **Documentation**: https://mlsysbook.github.io/TinyTorch/
- **Issues**: https://github.com/mlsysbook/TinyTorch/issues
- **Quick Reference**: `docs/STUDENT_QUICKSTART.md`

---

## Architecture-Specific Notes

### macOS Apple Silicon (M1/M2/M3)

```bash
# Ensure arm64 Python
arch -arm64 /usr/bin/python3 -m venv .venv
source .venv/bin/activate
uv sync
```

### Windows WSL2

```bash
# Install Python and git first
sudo apt update
sudo apt install python3 python3-venv python3-pip git

# Then follow standard installation
curl -LsSf https://astral.sh/uv/install.sh | sh
cd TinyTorch
uv sync
```

### Linux

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Standard installation
cd TinyTorch
uv sync
```

---

## Uninstallation

To completely remove TinyTorch:

```bash
# Remove virtual environment
rm -rf .venv

# Remove cache (optional)
rm -rf ~/.cache/tinytorch

# Remove repository
cd ..
rm -rf TinyTorch
```

---

## Version Information

- **TinyTorch**: v0.1.0
- **Python**: >=3.9, <4.0
- **UV**: >=0.6.0
- **Documentation**: https://mlsysbook.github.io/TinyTorch/

---

**Happy Learning! 🔥**

Build production ML systems from scratch, one module at a time.
