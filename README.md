# TinyTorch

This repository is a copy of the TinyTorch project from the ML Systems Book:
https://github.com/mlsysbook/TinyTorch

Notes:

-   **Provenance:** Forked/derived from `mlsysbook/TinyTorch` and adapted for
    the "Machine learning systems from scratch" module used at HAW Landshut.
-   **Setup:** This copy uses a `uv`-based setup (see `setup-uv.sh`) and includes a few small fixes to integrate with the
    coursework and local environment.

## Quick Start

Get started with TinyTorch in under 2 minutes using UV, the fastest Python package manager.

### One-Command Setup

```bash
# Install everything at once
./setup-uv.sh

# Activate the virtual environment
source activate.sh
```

### Start Learning

```bash
# Start your first module
tito module start 01

# Work in Jupyter, then complete it
tito module complete 01

# Check your progress
tito module status

# Continue to next module
tito module start 02
```

Note that your ipynb files are converted to .py files in the `modules/` directory for easier editing. You can run them directly or open them in Jupyter. The py files are the canonical source for grading and testing. The conversion is done automatically when you run `tito module complete <module_number>`.

### Common Commands

```bash
# View all commands
tito

# Get help for any command
tito <command> --help

# Module workflow
tito module start 01      # Open module 01 with Jupyter or VSCode (via --code)
tito module complete 01   # Test + export + track progress
tito module status        # View all progress

# Progress tracking
tito checkpoint status    # Capabilities unlocked
tito milestone list       # Major achievements

# Troubleshooting
tito system doctor        # Run diagnostics
```

## Development Workflow

TinyTorch uses **Jupyter notebooks (.ipynb) as the source of truth** for all module development.

### Working with Modules

Each module folder (e.g., `modules/01_tensor/`) contains:

-   `XX.ipynb` - The notebook where you do your work (source of truth)
-   `XX_solution.py` - Reference solution code
-   `XX.py` - Auto-generated Python file (created from the notebook during `tito module complete XX`)

### Development Process

1. **Work in the notebook**: Edit and develop your code in the `.ipynb` file

    ```bash
    tito module start 01  # Opens Jupyter Lab in the module directory or use --code to open in VSCode
    ```

2. **Generate the Python file**: When finished, run:

    ```bash
    tito module complete 01
    ```

    This command:

    - Converts the notebook (`.ipynb`) to a Python file (`.py`)
    - Processes the code for testing
    - Exports code to the `tinytorch` package
    - Runs the test suite

3. **The `.py` file is used for**:
    - Running automated tests
    - Importing into the main `tinytorch` package
    - Integration with the overall system

**Important**: Always edit the `.ipynb` notebook, never the generated `.py` file directly. The `.py` file will be overwritten when you run `tito module complete`.
