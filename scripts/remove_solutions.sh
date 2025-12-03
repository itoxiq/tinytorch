#!/bin/bash

# Wrapper script for remove_solutions.py
# Usage:
#   ./remove_solutions.sh           # Process all folders
#   ./remove_solutions.sh 01        # Process folder 01
#   ./remove_solutions.sh 01 03 07  # Process folders 01, 03, and 07
#   ./remove_solutions.sh --range 01-05  # Process folders 01-05

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# If no arguments provided, use --all
if [ $# -eq 0 ]; then
    python3 "$SCRIPT_DIR/remove_solutions.py" --all --base-path modules
else
    python3 "$SCRIPT_DIR/remove_solutions.py" "$@" --base-path modules
fi
