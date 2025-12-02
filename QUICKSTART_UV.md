# TinyTorch Quick Start with UV

Get started with TinyTorch in under 2 minutes using UV, the fastest Python package manager.

## One-Command Setup

```bash
# Install everything at once
./setup-uv.sh

# Activate the virtual environment
source activate.sh
```

## Start Learning

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

## Common Commands

```bash
# View all commands
tito

# Get help for any command
tito <command> --help

# Module workflow
tito module start 01      # Open module 01
tito module complete 01   # Test + export + track progress
tito module status        # View all progress

# Progress tracking
tito checkpoint status    # Capabilities unlocked
tito milestone list       # Major achievements

# Troubleshooting
tito system doctor        # Run diagnostics
```
