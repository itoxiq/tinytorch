#!/bin/bash
# TinyTorch Setup with UV
# Fast, modern Python package management for TinyTorch

set -e  # Exit on error

echo "🔥 TinyTorch Setup (UV Edition)"
echo "================================"
echo ""

# Detect system
OS=$(uname -s)
ARCH=$(uname -m)

echo "📋 System Info:"
echo "   OS: $OS"
echo "   Architecture: $ARCH"
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "⚠️  UV not found. Installing UV..."
    echo ""

    if [ "$OS" = "Darwin" ] || [ "$OS" = "Linux" ]; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        echo ""
        echo "✅ UV installed!"
        echo "💡 You may need to restart your shell or run: source ~/.bashrc"
        echo ""

        # Try to source common profile files
        if [ -f "$HOME/.bashrc" ]; then
            source "$HOME/.bashrc" 2>/dev/null || true
        fi
        if [ -f "$HOME/.zshrc" ]; then
            source "$HOME/.zshrc" 2>/dev/null || true
        fi
    else
        echo "❌ Automatic UV installation not supported on this platform."
        echo "💡 Please install UV manually: pip install uv"
        exit 1
    fi
fi

# Verify UV installation
echo "🔍 Verifying UV installation..."
UV_VERSION=$(uv --version || echo "not found")
echo "   UV version: $UV_VERSION"
echo ""

if [ "$UV_VERSION" = "not found" ]; then
    echo "❌ UV installation failed. Please install manually:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Remove old venv if it exists
if [ -d ".venv" ]; then
    echo "🗑️  Removing old virtual environment..."
    rm -rf .venv
    echo "✅ Old environment removed"
    echo ""
fi

# Install with UV
echo "📦 Installing TinyTorch with UV..."
echo ""

# Sync dependencies (creates venv automatically)
uv sync --extra all

echo ""
echo "✅ Installation complete!"
echo ""

# Verify installation
echo "🔍 Verifying installation..."
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    VENV_ARCH=$(.venv/bin/python3 -c "import platform; print(platform.machine())")
    echo "   Python architecture: $VENV_ARCH"
fi

echo "   Python location: $(which python3)"
PYTHON_VERSION=$(.venv/bin/python3 --version)
echo "   Python version: $PYTHON_VERSION"
echo ""

# Create activation helper
cat > activate.sh << 'EOF'
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
EOF

chmod +x activate.sh

echo "✅ Setup complete!"
echo ""
echo "🚀 Next steps:"
echo "   1. source activate.sh           # Activate environment"
echo "   2. tito --version               # Verify tito command"
echo "   3. tito system doctor           # Run diagnostics"
echo "   4. tito module start 01         # Start learning!"
echo ""
echo "💡 Quick commands:"
echo "   tito                   # Show all available commands"
echo "   tito module status     # View your progress"
echo "   tito checkpoint status # See capabilities unlocked"
echo ""
