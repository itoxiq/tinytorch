#!/bin/bash
# TinyTorch activation helper (UV edition)
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "🔥 TinyTorch environment activated"
    echo "💡 Try: tito --version"
else
    echo "❌ Virtual environment not found. Run ./setup-uv.sh first."
    return 1
fi
