#!/bin/bash
# AndesCode macOS build script
# Produces AndesCode.app in dist/
# Requirements: pip install pyinstaller pywebview

set -e
echo "🏔️  Building AndesCode for macOS..."

# Check pyinstaller
if ! python3 -c "import PyInstaller" 2>/dev/null; then
    echo "Installing PyInstaller..."
    pip3 install pyinstaller
fi

# Check pywebview
if ! python3 -c "import webview" 2>/dev/null; then
    echo "Installing pywebview..."
    pip3 install pywebview
fi

# Clean previous build
rm -rf dist/AndesCode.app build/ AndesCode.spec 2>/dev/null || true

# Build
pyinstaller \
    --name "AndesCode" \
    --windowed \
    --onedir \
    --icon "assets/icon.icns" \
    --add-data "static:static" \
    --add-data "requirements.txt:." \
    --hidden-import "webview" \
    --hidden-import "webview.platforms.cocoa" \
    --collect-all "webview" \
    --noconfirm \
    app.py

echo ""
echo "✅  Build complete: dist/AndesCode.app"
echo ""
echo "To install: drag dist/AndesCode.app to /Applications"
echo "To test:    open dist/AndesCode.app"
