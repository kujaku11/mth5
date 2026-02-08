#!/bin/bash
# Build MTH5 Validator Standalone Executable (Linux/Mac)
# Quick build script using PyInstaller

set -e  # Exit on error

echo "==============================================================================="
echo "MTH5 Validator - Standalone Executable Builder (Unix)"
echo "==============================================================================="
echo ""

# Check if PyInstaller is installed
if ! python -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller not found. Installing..."
    python -m pip install pyinstaller
fi

echo "Building executable..."
echo ""

pyinstaller \
    --onefile \
    --name mth5-validator \
    --console \
    --clean \
    --hidden-import=h5py.defs \
    --hidden-import=h5py.utils \
    --hidden-import=h5py._proxy \
    --hidden-import=h5py.h5ac \
    --collect-all=mt_metadata \
    --collect-all=mth5 \
    --copy-metadata=mth5 \
    --copy-metadata=mt_metadata \
    --copy-metadata=h5py \
    --exclude-module=matplotlib \
    --exclude-module=tkinter \
    --exclude-module=PyQt5 \
    --exclude-module=PyQt6 \
    --exclude-module=jupyter \
    --exclude-module=notebook \
    --log-level=WARN \
    mth5/utils/cli.py

echo ""
echo "==============================================================================="
echo "BUILD SUCCESSFUL"
echo "==============================================================================="
echo ""
echo "Executable location: dist/mth5-validator"
echo ""
echo "Test it:"
echo "  ./dist/mth5-validator --help"
echo "  ./dist/mth5-validator validate your_file.mth5"
echo ""
echo "==============================================================================="
