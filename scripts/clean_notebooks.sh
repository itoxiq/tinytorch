#!/usr/bin/env bash
# Generate Jupyter notebooks from Python module files
#
# Usage:
#   ./scripts/clean_notebooks.sh              # Convert all modules
#   ./scripts/clean_notebooks.sh 19_benchmarking  # Convert specific module

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv/bin" ]; then
    source .venv/bin/activate
fi

# Run the Python script
.venv/bin/python scripts/clean_notebooks.py "$@"
